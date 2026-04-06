"""
Scripted Lift / Stack demo with Panda + touch_grid validation and plots.

This demo:
1) Builds a temporary Panda gripper XML with configurable touch_grid sensors.
2) Runs a selected task (Lift or Stack) with default controller behavior.
3) Executes pre-grasp -> grasp -> close -> lift phases.
4) Logs touch_grid observations, validates them, and saves plots at the end.
"""

import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure local checkout imports when script is run as:
# python robosuite/demos/demo_lift_touch_grid_trajectory.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robosuite.models.grippers import register_gripper  # noqa: E402
from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402
from robosuite.models.grippers.panda_gripper import PandaGripper  # noqa: E402
from robosuite.utils.mjcf_utils import xml_path_completion  # noqa: E402

import robosuite as suite  # noqa: E402


def _validate_touch_config(args):
    if not (1 <= args.touch_nchannel <= 6):
        raise ValueError("--touch-nchannel must be in [1, 6]")
    if args.touch_size_x <= 0 or args.touch_size_y <= 0:
        raise ValueError("--touch-size-x and --touch-size-y must be > 0")
    if not (0 < args.touch_fov_x <= 180):
        raise ValueError("--touch-fov-x must be in (0, 180]")
    if not (0 < args.touch_fov_y <= 90):
        raise ValueError("--touch-fov-y must be in (0, 90]")
    if not (0 <= args.touch_gamma <= 1):
        raise ValueError("--touch-gamma must be in [0, 1]")


def _read_touch_defaults_from_xml(base_xml_path):
    root = ET.parse(base_xml_path).getroot()
    sensor = root.find("sensor")
    if sensor is None:
        raise ValueError(f"Missing <sensor> block in {base_xml_path}")

    plugin = sensor.find("./plugin[@name='touch_grid_left']")
    if plugin is None:
        plugin = sensor.find("./plugin[@plugin='mujoco.sensor.touch_grid']")
    if plugin is None:
        raise ValueError(f"No touch_grid plugin defaults found in {base_xml_path}")

    cfg = {node.get("key"): node.get("value") for node in plugin.findall("config")}
    for req in ("nchannel", "size", "fov", "gamma"):
        if req not in cfg:
            raise ValueError(f"Missing touch_grid config key '{req}' in {base_xml_path}")

    size_parts = cfg["size"].split()
    fov_parts = cfg["fov"].split()
    if len(size_parts) != 2 or len(fov_parts) != 2:
        raise ValueError(f"Invalid touch_grid size/fov in {base_xml_path}")

    return {
        "touch_nchannel": int(cfg["nchannel"]),
        "touch_size_x": int(size_parts[0]),
        "touch_size_y": int(size_parts[1]),
        "touch_fov_x": float(fov_parts[0]),
        "touch_fov_y": float(fov_parts[1]),
        "touch_gamma": float(cfg["gamma"]),
    }


def _resolve_touch_config(args, base_xml_path):
    defaults = _read_touch_defaults_from_xml(base_xml_path)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def _ensure_touch_grid_xml(base_xml_path, dst_xml_path, args):
    tree = ET.parse(base_xml_path)
    root = tree.getroot()
    base_dir = Path(base_xml_path).resolve().parent

    extension = root.find("extension")
    if extension is None:
        extension = ET.Element("extension")
        root.insert(0, extension)
    if extension.find("./plugin[@plugin='mujoco.sensor.touch_grid']") is None:
        extension.append(ET.Element("plugin", {"plugin": "mujoco.sensor.touch_grid"}))

    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.Element("sensor")
        root.append(sensor)

    def add_touch_sensor(name, site_name):
        existing = sensor.find(f"./plugin[@name='{name}']")
        if existing is not None:
            sensor.remove(existing)
        plugin = ET.Element(
            "plugin",
            {
                "name": name,
                "plugin": "mujoco.sensor.touch_grid",
                "objtype": "site",
                "objname": site_name,
            },
        )
        plugin.append(ET.Element("config", {"key": "nchannel", "value": str(args.touch_nchannel)}))
        plugin.append(ET.Element("config", {"key": "size", "value": f"{args.touch_size_x} {args.touch_size_y}"}))
        plugin.append(ET.Element("config", {"key": "fov", "value": f"{args.touch_fov_x} {args.touch_fov_y}"}))
        plugin.append(ET.Element("config", {"key": "gamma", "value": str(args.touch_gamma)}))
        sensor.append(plugin)

    add_touch_sensor("touch_grid_left", "finger1_touch_grid_site")
    add_touch_sensor("touch_grid_right", "finger2_touch_grid_site")

    # Keep all mesh references valid when writing this XML to a temp directory.
    for node in root.findall("./asset/*[@file]"):
        rel = node.get("file")
        if rel is not None:
            node.set("file", str((base_dir / rel).resolve()))

    tree.write(dst_xml_path, encoding="unicode")


def _register_touch_grid_gripper(xml_path):
    class PandaTouchGridLiftGripper(PandaGripper):
        def __init__(self, idn=0):
            GripperModel.__init__(self, xml_path, idn=idn)

    register_gripper(PandaTouchGridLiftGripper)
    return PandaTouchGridLiftGripper.__name__


def _get_touch_sensor_meta(sim):
    meta = []
    for sensor_name in sim.model.sensor_names:
        if "touch_grid" not in sensor_name:
            continue
        sensor_id = sim.model.sensor_name2id(sensor_name)
        sensor_dim = int(sim.model.sensor_dim[sensor_id])
        if hasattr(sim.model, "sensor_adr"):
            sensor_adr = int(sim.model.sensor_adr[sensor_id])
        else:
            sensor_adr = int(np.sum(sim.model.sensor_dim[:sensor_id]))
        meta.append((sensor_name, sensor_adr, sensor_dim))
    return meta


def _infer_action_parts(robot):
    split = robot._action_split_indexes
    dims = {name: end - start for name, (start, end) in split.items()}
    gripper_part = next((k for k in dims if "gripper" in k and dims[k] > 0), None)
    arm_candidates = [k for k in dims if dims[k] >= 6 and "gripper" not in k]
    arm_part = "right" if "right" in arm_candidates else (arm_candidates[0] if arm_candidates else None)
    if arm_part is None or gripper_part is None:
        raise RuntimeError(f"Unable to infer action parts from split indexes: {split}")
    return arm_part, gripper_part


def _pick_obs_key(obs, suffix):
    exact = f"robot0_{suffix}"
    if exact in obs:
        return exact
    matches = [k for k in obs if k.endswith(suffix)]
    if matches:
        return matches[0]
    raise KeyError(f"Could not find observation key ending with '{suffix}'. Available keys: {list(obs.keys())}")


def _get_touch_site_debug_meta(sim):
    meta = []
    for site_name in sim.model.site_names:
        if "touch_grid_site" in site_name:
            site_id = sim.model.site_name2id(site_name)
            meta.append((site_name, site_id))
    return meta


class _LiveTouchGridPlot:
    """Live matplotlib plot with paper-style tactile visualization.

    Layout:
    - Top row: timeseries of max/mean magnitude per sensor.
    - Bottom row: one paper-style tactile image per sensor (red=normal, green arrows=shear).
    """

    def __init__(self, sensor_meta, args):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.args = args
        self.sensor_names = [name for name, _, _ in sensor_meta]
        self.sample_steps = []
        self.max_hist = {name: [] for name in self.sensor_names}
        self.mean_hist = {name: [] for name in self.sensor_names}
        self._ts_lines = {}

        n_sensors = len(self.sensor_names)
        # Rows: 1 for timeseries + 1 for tactile images.  Cols: n_sensors.
        plt.ion()
        self.fig, all_axes = plt.subplots(
            2, max(n_sensors, 1), figsize=(6 * max(n_sensors, 1), 8),
            squeeze=False,
        )
        # Timeseries axes (span full top row)
        self.ax_ts = all_axes[0, 0]
        for c in range(1, n_sensors):
            all_axes[0, c].set_visible(False)

        # Tactile image axes
        res = args.tactile_resolution
        H, W = args.touch_size_y, args.touch_size_x
        zero_img = np.zeros((H * res, W * res, 3), dtype=np.uint8)
        self.ax_tac = []
        self.tac_im = []
        for s, name in enumerate(self.sensor_names):
            ax = all_axes[1, s]
            im = ax.imshow(zero_img)
            ax.set_title(f"{name}\n(red=normal, arrows=shear)", fontsize=8)
            ax.axis("off")
            self.ax_tac.append(ax)
            self.tac_im.append(im)

        # Pre-create timeseries line objects
        for name in self.sensor_names:
            ln_max, = self.ax_ts.plot([], [], label=f"{name} max|x|")
            ln_mean, = self.ax_ts.plot([], [], linestyle="--", alpha=0.75, label=f"{name} mean|x|")
            self._ts_lines[name] = (ln_max, ln_mean)
        self.ax_ts.set_xlabel("Step")
        self.ax_ts.set_ylabel("Magnitude")
        self.ax_ts.grid(True, alpha=0.3)
        self.ax_ts.legend(loc="upper right", fontsize=7)

        self.fig.suptitle("Live touch_grid")
        self.fig.tight_layout()
        self.fig.show()
        plt.show(block=False)

    def update(self, step_idx, phase_name, sensor_values):
        self.sample_steps.append(step_idx)
        x = np.array(self.sample_steps)
        n_ch = self.args.touch_nchannel
        H, W = self.args.touch_size_y, self.args.touch_size_x

        # Update timeseries
        for name in self.sensor_names:
            arr = sensor_values[name]
            self.max_hist[name].append(float(np.max(np.abs(arr))))
            self.mean_hist[name].append(float(np.mean(np.abs(arr))))
            ln_max, ln_mean = self._ts_lines[name]
            ln_max.set_data(x, self.max_hist[name])
            ln_mean.set_data(x, self.mean_hist[name])
        self.ax_ts.relim()
        self.ax_ts.autoscale_view()
        self.ax_ts.set_title(f"Live timeseries (phase={phase_name}, step={step_idx})")

        # Update paper-style tactile images
        for s, name in enumerate(self.sensor_names):
            grid = sensor_values[name].reshape(n_ch, H, W)
            normal = grid[0]
            shear = np.stack([grid[1], grid[2]], axis=-1) if n_ch >= 3 else np.zeros((H, W, 2))
            tac_img = _compute_tactile_shear_image(
                normal, shear,
                normal_force_threshold=self.args.tactile_nf_threshold,
                shear_force_threshold=self.args.tactile_sf_threshold,
                resolution=self.args.tactile_resolution,
            )
            # Convert BGR to RGB for matplotlib
            self.tac_im[s].set_data(tac_img[:, :, ::-1])

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self, hold=False):
        if hold:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.close(self.fig)


def _print_contact_debug(env, step_idx, limit):
    ncon = int(env.sim.data.ncon)
    if ncon == 0:
        print(f"[step {step_idx}] contacts: none")
        return
    print(f"[step {step_idx}] contacts: {ncon}")
    for i in range(min(ncon, limit)):
        con = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(int(con.geom1))
        g2 = env.sim.model.geom_id2name(int(con.geom2))
        print(f"  - {i}: {g1} <-> {g2}, dist={float(con.dist):.6f}")


def _print_stack_contact_summary(env, step_idx):
    ncon = int(env.sim.data.ncon)
    cube_a_hits = 0
    cube_b_hits = 0
    for i in range(ncon):
        con = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(int(con.geom1))
        g2 = env.sim.model.geom_id2name(int(con.geom2))
        pair = f"{g1} {g2}"
        if "cubeA" in pair:
            cube_a_hits += 1
        if "cubeB" in pair:
            cube_b_hits += 1
    print(f"[step {step_idx}] stack-contact-summary: cubeA_pairs={cube_a_hits}, cubeB_pairs={cube_b_hits}")


def _print_site_debug(env, step_idx, site_meta):
    print(f"[step {step_idx}] touch site poses:")
    for site_name, site_id in site_meta:
        pos = env.sim.data.site_xpos[site_id]
        rot = env.sim.data.site_xmat[site_id].reshape(3, 3)
        z_axis = rot[:, 2]
        forward = -z_axis
        print(
            f"  - {site_name}: pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) "
            f"-z=({forward[0]:.4f},{forward[1]:.4f},{forward[2]:.4f})"
        )


def _phase_step(
    env,
    obs,
    robot,
    arm_part,
    gripper_part,
    target_pos,
    gripper_cmd,
    steps,
    pos_gain,
    pos_scale,
    arm_clip,
    low,
    high,
    render,
    phase_name,
    traces,
    sensor_meta,
    eef_key,
    step_callback=None,
):
    for _ in range(steps):
        eef_pos = obs[eef_key]
        delta = pos_gain * (target_pos - eef_pos) / max(pos_scale, 1e-8)
        arm_action = np.zeros(6)
        arm_action[:3] = np.clip(delta, -arm_clip, arm_clip)
        gripper_action = np.array([gripper_cmd], dtype=np.float64)
        action = robot.create_action_vector({arm_part: arm_action, gripper_part: gripper_action})
        action = np.clip(action, low, high)

        obs, _, _, _ = env.step(action)
        if render:
            env.render()

        traces["phase"].append(phase_name)
        traces["eef_pos"].append(obs[eef_key].copy())
        latest = {}
        for name, adr, dim in sensor_meta:
            val = env.sim.data.sensordata[adr : adr + dim].copy()
            traces["sensors"][name].append(val)
            latest[name] = val

        if step_callback is not None:
            step_callback(len(traces["phase"]) - 1, phase_name, latest)

    return obs


def _validate_traces(traces, sensor_meta, args, interaction_phases):
    if not sensor_meta:
        raise RuntimeError("No touch_grid sensors found in simulation model.")

    expected_dim = args.touch_nchannel * args.touch_size_x * args.touch_size_y
    for name, _, dim in sensor_meta:
        if dim != expected_dim:
            raise RuntimeError(f"Sensor {name} dim={dim}, expected={expected_dim} from configured touch-grid spec.")

    phase_arr = np.array(traces["phase"])
    baseline_mask = phase_arr == "pre_grasp"
    interaction_mask = np.isin(phase_arr, interaction_phases)

    if not baseline_mask.any() or not interaction_mask.any():
        raise RuntimeError("Missing baseline or interaction phase samples for validation.")

    any_contact = False
    stats = {}
    for name, _, _ in sensor_meta:
        values = np.array(traces["sensors"][name])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values found in sensor {name}.")

        baseline_vals = values[baseline_mask]
        interaction_vals = values[interaction_mask]
        baseline_max = float(np.max(np.abs(baseline_vals)))
        interaction_max = float(np.max(np.abs(interaction_vals)))
        std_all = float(np.std(values))

        if baseline_max > args.baseline_eps:
            raise RuntimeError(
                f"Baseline too high for {name}: {baseline_max:.6f} > baseline eps {args.baseline_eps:.6f}"
            )
        if std_all <= 0:
            raise RuntimeError(f"Sensor {name} appears constant (std={std_all:.6f}).")
        if interaction_max >= args.contact_threshold:
            any_contact = True

        stats[name] = {
            "baseline_max_abs": baseline_max,
            "interaction_max_abs": interaction_max,
            "std": std_all,
        }

    if not any_contact:
        raise RuntimeError(
            f"No touch_grid sensor exceeded contact threshold {args.contact_threshold:.6f} during interaction phases {interaction_phases}."
        )
    return stats


def _plot_results(traces, sensor_meta, args, interaction_phases):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(args.plot_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    suffix = out_path.suffix or ".png"
    parent = out_path.parent

    steps = np.arange(len(traces["phase"]))
    phase_arr = np.array(traces["phase"])
    interaction_idx = np.where(np.isin(phase_arr, interaction_phases))[0]

    # Timeseries plot
    ts_fig, ts_ax = plt.subplots(figsize=(12, 5))
    for name, _, _ in sensor_meta:
        vals = np.array(traces["sensors"][name])
        max_abs = np.max(np.abs(vals), axis=1)
        mean_abs = np.mean(np.abs(vals), axis=1)
        ts_ax.plot(steps, max_abs, label=f"{name} max|x|")
        ts_ax.plot(steps, mean_abs, linestyle="--", alpha=0.8, label=f"{name} mean|x|")
    ts_ax.set_title("touch_grid timeseries")
    ts_ax.set_xlabel("Step")
    ts_ax.set_ylabel("Magnitude")
    ts_ax.legend(loc="upper right", fontsize=8)
    ts_ax.grid(True, alpha=0.3)
    ts_fig.tight_layout()
    ts_path = parent / f"{stem}_timeseries{suffix}"
    ts_fig.savefig(ts_path, dpi=160)
    plt.close(ts_fig)

    # Paper-style tactile snapshots (red=normal force, green arrows=shear)
    n_ch = args.touch_nchannel
    for name, _, _ in sensor_meta:
        vals = np.array(traces["sensors"][name])
        all_grids = vals.reshape(len(vals), n_ch, args.touch_size_y, args.touch_size_x)

        if len(interaction_idx) > 0:
            int_vals = vals[interaction_idx]
            peak_rel = int(np.argmax(np.max(np.abs(int_vals), axis=1)))
            peak_idx = interaction_idx[peak_rel]
        else:
            peak_idx = len(vals) - 1

        for label, idx in [("final", len(vals) - 1), (f"peak@{peak_idx}", peak_idx)]:
            grid = all_grids[idx]
            normal = grid[0]
            shear = np.stack([grid[1], grid[2]], axis=-1) if n_ch >= 3 else np.zeros((args.touch_size_y, args.touch_size_x, 2))
            tac_img = _compute_tactile_shear_image(
                normal, shear,
                normal_force_threshold=args.tactile_nf_threshold,
                shear_force_threshold=args.tactile_sf_threshold,
                resolution=args.tactile_resolution,
            )
            # Convert BGR to RGB for matplotlib
            tac_img_rgb = tac_img[:, :, ::-1]
            tac_fig, tac_ax = plt.subplots(figsize=(4, 5))
            tac_ax.imshow(tac_img_rgb)
            tac_ax.set_title(f"{name} {label}\n(red=normal, arrows=shear)", fontsize=9)
            tac_ax.axis("off")
            tac_fig.tight_layout()
            safe_name = name.replace("/", "_")
            safe_label = label.replace("@", "_at_")
            tac_path = parent / f"{stem}_{safe_name}_tactile_{safe_label}{suffix}"
            tac_fig.savefig(tac_path, dpi=160)
            plt.close(tac_fig)

    print(f"Saved plot artifacts to {parent}")


def _compute_tactile_shear_image(
    tactile_normal_force: np.ndarray,   # (H, W)
    tactile_shear_force: np.ndarray,    # (H, W, 2)
    normal_force_threshold: float = 0.00008,
    shear_force_threshold: float = 0.0005,
    resolution: int = 30,
) -> np.ndarray:
    """Paper-style tactile visualization: black background, red = normal force, green arrows = shear.

    Matches ``compute_tactile_shear_image`` from Han et al. (CoRL 2025).
    Returns BGR image of shape (H * resolution, W * resolution, 3).
    """
    import cv2

    H, W = tactile_normal_force.shape
    img = np.zeros((H * resolution, W * resolution, 3), dtype=np.uint8)

    nf_max = np.max(np.abs(tactile_normal_force))
    if nf_max < normal_force_threshold:
        nf_max = 1.0  # no meaningful contact — keep black

    sf_max = np.max(np.abs(tactile_shear_force))
    if sf_max < shear_force_threshold:
        sf_max = 1.0

    for r in range(H):
        for c in range(W):
            nf = abs(tactile_normal_force[r, c])
            # Normalise to [0, 1] — black at 0, red at max
            intensity = min(nf / nf_max, 1.0)

            red = int(255 * intensity)

            cy = (H - 1 - r) * resolution + resolution // 2  # flip Y for display
            cx = c * resolution + resolution // 2
            half = resolution // 2 - 1

            # Fill cell background: black -> red
            cv2.rectangle(
                img,
                (cx - half, cy - half),
                (cx + half, cy + half),
                (0, 0, red),  # BGR: black-to-red
                cv2.FILLED,
            )

            # Draw shear arrow — scale by shear magnitude relative to shear max
            sx, sy = tactile_shear_force[r, c]
            shear_mag = np.sqrt(sx ** 2 + sy ** 2)
            if shear_mag > shear_force_threshold:
                arrow_len = min(shear_mag / sf_max, 1.0) * (resolution * 0.4)
                dx = int(arrow_len * sx / shear_mag)
                dy = int(-arrow_len * sy / shear_mag)  # flip Y
                cv2.arrowedLine(
                    img,
                    (cx, cy),
                    (cx + dx, cy + dy),
                    (0, 255, 0),  # green in BGR
                    thickness=max(1, resolution // 10),
                    tipLength=0.3,
                )

    return img


def _apply_task_tuning(args, parser):
    """Apply Stack-specific defaults only when user kept parser defaults."""
    if args.task != "Stack":
        return

    tuned_defaults = {
        "pre_grasp_height": 0.14,
        "grasp_height": 0.015,
        "close_descend": 0.01,
        "lift_height": 0.22,
        "contact_threshold": 8e-4,
        "stack_approach_height": 0.12,
        "stack_place_height": 0.045,
        "stack_release_steps": 30,
        "stack_retreat_height": 0.14,
    }
    for field, value in tuned_defaults.items():
        if getattr(args, field) == parser.get_default(field):
            setattr(args, field, value)


def _interaction_phases(task):
    phases = ["grasp", "close", "lift"]
    if task == "Stack":
        phases.extend(["move_above_cubeB", "place_on_cubeB", "release_on_cubeB", "retreat_after_release"])
    return phases


@dataclass
class TouchGridDemoConfig:
    args: argparse.Namespace

    @classmethod
    def from_parser(cls, parser):
        args = parser.parse_args()
        base_xml = xml_path_completion("grippers/panda_gripper_tactile.xml")
        _resolve_touch_config(args, base_xml)
        _validate_touch_config(args)
        _apply_task_tuning(args, parser)
        np.random.seed(args.seed)
        return cls(args=args)


class TrajectoryPlanner:
    def __init__(self, cfg: TouchGridDemoConfig):
        self.args = cfg.args

    def build_lift_like_phases(self, cube_pos):
        args = self.args
        pre_grasp = cube_pos + np.array([0.0, 0.0, args.pre_grasp_height])
        grasp = cube_pos + np.array([0.0, 0.0, args.grasp_height])
        close_target = cube_pos + np.array([0.0, 0.0, args.grasp_height - args.close_descend])
        lift = cube_pos + np.array([0.0, 0.0, args.lift_height])
        return [
            {"name": "pre_grasp", "target": pre_grasp, "gripper_cmd": -1.0, "steps": args.pre_grasp_steps},
            {"name": "grasp", "target": grasp, "gripper_cmd": -1.0, "steps": args.grasp_steps},
            {"name": "close", "target": close_target, "gripper_cmd": 1.0, "steps": args.close_steps},
            {"name": "lift", "target": lift, "gripper_cmd": 1.0, "steps": args.lift_steps},
        ]

    def build_stack_post_lift_phases(self, cube_b_pos):
        args = self.args
        above_b = cube_b_pos + np.array([0.0, 0.0, args.stack_approach_height])
        place_on_b = cube_b_pos + np.array([0.0, 0.0, args.stack_place_height])
        retreat = cube_b_pos + np.array([0.0, 0.0, args.stack_retreat_height])
        return [
            {"name": "move_above_cubeB", "target": above_b, "gripper_cmd": 1.0, "steps": args.stack_approach_steps},
            {"name": "place_on_cubeB", "target": place_on_b, "gripper_cmd": 1.0, "steps": args.stack_place_steps},
            {"name": "release_on_cubeB", "target": place_on_b, "gripper_cmd": -1.0, "steps": args.stack_release_steps},
            {"name": "retreat_after_release", "target": retreat, "gripper_cmd": -1.0, "steps": args.stack_retreat_steps},
        ]


class TouchGridRecorder:
    def __init__(self, env, args, sensor_meta, site_meta, live_plot=None):
        self.env = env
        self.args = args
        self.site_meta = site_meta
        self.live_plot = live_plot
        self.traces = {
            "phase": [],
            "eef_pos": [],
            "sensors": {name: [] for name, _, _ in sensor_meta},
            "agentview_frames": [],
        }

    def record_step(self, step_idx, phase_name, latest_sensor_values):
        args = self.args
        if args.save_video:
            try:
                frame = self.env.sim.render(
                    width=args.video_camera_width,
                    height=args.video_camera_height,
                    camera_name=args.video_camera_name,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to render camera '{args.video_camera_name}' for video capture."
                ) from exc
            self.traces["agentview_frames"].append(np.flipud(frame).copy())
        if args.debug_progress_every > 0 and (step_idx % args.debug_progress_every == 0):
            print(f"[step {step_idx}] phase={phase_name}")
        if args.debug_contacts_every > 0 and (step_idx % args.debug_contacts_every == 0):
            _print_contact_debug(self.env, step_idx, args.debug_contact_limit)
            if args.task == "Stack" and args.stack_contact_summary:
                _print_stack_contact_summary(self.env, step_idx)
        if args.debug_sites_every > 0 and (step_idx % args.debug_sites_every == 0):
            _print_site_debug(self.env, step_idx, self.site_meta)
        if self.live_plot is not None and (step_idx % max(args.live_plot_every, 1) == 0):
            self.live_plot.update(step_idx, phase_name, latest_sensor_values)


class TouchGridDemoRunner:
    def __init__(self, cfg: TouchGridDemoConfig):
        self.cfg = cfg
        self.args = cfg.args
        self.planner = TrajectoryPlanner(cfg)
        self.env = None
        self.live_plot = None

    def setup_env(self):
        args = self.args
        base_xml = xml_path_completion("grippers/panda_gripper_tactile.xml")
        self._tmpdir_ctx = tempfile.TemporaryDirectory(prefix="lift_touch_grid_demo_")
        tmpdir = self._tmpdir_ctx.__enter__()
        patched_xml = str(Path(tmpdir) / "panda_gripper_touch_grid.xml")
        _ensure_touch_grid_xml(base_xml, patched_xml, args)
        gripper_name = _register_touch_grid_gripper(patched_xml)
        self.env = suite.make(
            args.task,
            robots="Panda",
            gripper_types=gripper_name,
            controller_configs=None,
            has_renderer=args.render,
            has_offscreen_renderer=args.save_video,
            use_camera_obs=False,
            control_freq=args.control_freq,
            horizon=2000,
            ignore_done=True,
        )

    def _execute_phase(
        self, obs, robot, arm_part, gripper_part, low, high, eef_key, sensor_meta, recorder, phase
    ):
        args = self.args
        for _ in range(phase["steps"]):
            eef_pos = obs[eef_key]
            delta = args.pos_gain * (phase["target"] - eef_pos) / max(args.pos_scale, 1e-8)
            arm_action = np.zeros(6)
            arm_action[:3] = np.clip(delta, -args.arm_clip, args.arm_clip)
            gripper_action = np.array([phase["gripper_cmd"]], dtype=np.float64)
            action = robot.create_action_vector({arm_part: arm_action, gripper_part: gripper_action})
            action = np.clip(action, low, high)

            obs, _, _, _ = self.env.step(action)
            if args.render:
                self.env.render()

            recorder.traces["phase"].append(phase["name"])
            recorder.traces["eef_pos"].append(obs[eef_key].copy())
            latest = {}
            for name, adr, dim in sensor_meta:
                val = self.env.sim.data.sensordata[adr : adr + dim].copy()
                recorder.traces["sensors"][name].append(val)
                latest[name] = val
            recorder.record_step(len(recorder.traces["phase"]) - 1, phase["name"], latest)
        return obs

    def run(self):
        args = self.args
        print(
            "touch_grid config: "
            f"nchannel={args.touch_nchannel}, size=({args.touch_size_x},{args.touch_size_y}), "
            f"fov=({args.touch_fov_x},{args.touch_fov_y}), gamma={args.touch_gamma}"
        )
        print(f"selected task: {args.task}")

        self.setup_env()
        try:
            obs = self.env.reset()
            low, high = self.env.action_spec
            robot = self.env.robots[0]
            arm_part, gripper_part = _infer_action_parts(robot)
            eef_key = _pick_obs_key(obs, "eef_pos")
            target_obs_suffix = "cube_pos" if args.task == "Lift" else "cubeA_pos"
            object_key = _pick_obs_key(obs, target_obs_suffix)
            cube_pos = obs[object_key].copy()

            sensor_meta = _get_touch_sensor_meta(self.env.sim)
            site_meta = _get_touch_site_debug_meta(self.env.sim)
            self.live_plot = _LiveTouchGridPlot(sensor_meta, args) if args.live_plot else None
            recorder = TouchGridRecorder(self.env, args, sensor_meta, site_meta, self.live_plot)

            phases = self.planner.build_lift_like_phases(cube_pos)
            for phase in phases:
                obs = self._execute_phase(
                    obs, robot, arm_part, gripper_part, low, high, eef_key, sensor_meta, recorder, phase
                )

            if args.task == "Stack":
                cube_b_key = _pick_obs_key(obs, "cubeB_pos")
                cube_b_pos = obs[cube_b_key].copy()
                for phase in self.planner.build_stack_post_lift_phases(cube_b_pos):
                    obs = self._execute_phase(
                        obs, robot, arm_part, gripper_part, low, high, eef_key, sensor_meta, recorder, phase
                    )
                print(f"stack success: {bool(self.env._check_success())}")

            qstate = np.concatenate([self.env.sim.data.qpos.ravel(), self.env.sim.data.qvel.ravel()])
            if not np.isfinite(qstate).all():
                raise RuntimeError("Simulation produced non-finite state values (NaN / inf).")

            interaction_phases = _interaction_phases(args.task)
            stats = _validate_traces(recorder.traces, sensor_meta, args, interaction_phases)
            print("touch_grid validation: PASS")
            for sensor_name, s in stats.items():
                print(
                    f"- {sensor_name}: baseline_max={s['baseline_max_abs']:.6f}, "
                    f"interaction_max={s['interaction_max_abs']:.6f}, std={s['std']:.6f}"
                )

            _plot_results(recorder.traces, sensor_meta, args, interaction_phases)
            if args.save_video:
                _render_tactile_shear_video(recorder.traces, sensor_meta, args)
        finally:
            if self.live_plot is not None:
                self.live_plot.close(hold=args.hold_live_plot)
            if self.env is not None:
                self.env.close()
            if hasattr(self, "_tmpdir_ctx"):
                self._tmpdir_ctx.__exit__(None, None, None)


def _render_tactile_shear_video(traces, sensor_meta, args):
    """Render a combined video: agentview (left) + tactile maps (right)."""
    import cv2

    out_path = Path(args.video_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ch = args.touch_nchannel
    H, W = args.touch_size_y, args.touch_size_x
    res = args.tactile_resolution

    if n_ch < 3:
        print(f"WARNING: Paper-style tactile video requires nchannel >= 3 (got {n_ch}). "
              "Shear arrows will be zero; only normal force will be shown.")

    # Precompute tactile grids
    all_grids = {}
    for name, _, _ in sensor_meta:
        v = np.array(traces["sensors"][name])
        all_grids[name] = v.reshape(len(v), n_ch, H, W)

    n_frames = len(traces["phase"])
    camera_frames = traces.get("agentview_frames", [])
    if len(camera_frames) != n_frames:
        raise RuntimeError(
            f"Expected {n_frames} agentview frames for video, got {len(camera_frames)}. "
            "Make sure rollout captured camera frames while save-video was enabled."
        )

    tactile_sensor_names = [name for name, _, _ in sensor_meta]
    if len(tactile_sensor_names) < 2:
        raise RuntimeError(
            f"Need at least 2 touch_grid sensors for 3-column output (got {len(tactile_sensor_names)})."
        )
    # User requested 3 fixed columns: camera + 2 tactile panels.
    tactile_sensor_names = tactile_sensor_names[:2]

    cam_w = args.video_camera_width
    cam_h = args.video_camera_height
    gap = 4
    tactile_panel_w = max(1, int((W * cam_h) / max(H, 1)))
    frame_w = cam_w + gap + tactile_panel_w + gap + tactile_panel_w

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.video_fps, (frame_w, cam_h + 30))

    print(f"Rendering {n_frames} tactile frames to {out_path} ...")
    for t in range(n_frames):
        canvas = np.zeros((cam_h + 30, frame_w, 3), dtype=np.uint8)

        # Left column: agentview camera
        cam_rgb = camera_frames[t]
        if cam_rgb.shape[:2] != (cam_h, cam_w):
            cam_rgb = cv2.resize(cam_rgb, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
        cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
        canvas[:cam_h, 0:cam_w] = cam_bgr

        # Right columns: two tactile panels
        x_offset = cam_w + gap
        for name in tactile_sensor_names:
            grid = all_grids[name][t]  # (n_ch, H, W)
            normal = grid[0]  # channel 0 = normal(z)
            if n_ch >= 3:
                shear = np.stack([grid[1], grid[2]], axis=-1)  # (H, W, 2) = tangent(x), tangent(y)
            else:
                shear = np.zeros((H, W, 2))

            img = _compute_tactile_shear_image(
                normal, shear,
                normal_force_threshold=args.tactile_nf_threshold,
                shear_force_threshold=args.tactile_sf_threshold,
                resolution=res,
            )
            img = cv2.resize(img, (tactile_panel_w, cam_h), interpolation=cv2.INTER_NEAREST)
            canvas[:cam_h, x_offset:x_offset + tactile_panel_w] = img
            x_offset += tactile_panel_w + gap

        # Add text overlay with phase/step info
        phase = traces["phase"][t]
        cv2.putText(
            canvas, f"step {t}  phase={phase}",
            (5, cam_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        writer.write(canvas)

        if (t + 1) % 50 == 0:
            print(f"  frame {t + 1}/{n_frames}")

    writer.release()
    print(f"Saved tactile shear video to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="Lift",
        choices=["Lift", "Stack"],
        help="Robosuite task to run.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pre-grasp-steps", type=int, default=20)
    parser.add_argument("--grasp-steps", type=int, default=50)
    parser.add_argument("--close-steps", type=int, default=50)
    parser.add_argument("--lift-steps", type=int, default=50)
    parser.add_argument("--stack-approach-steps", type=int, default=50, help="Stack-only steps to move above cubeB.")
    parser.add_argument("--stack-place-steps", type=int, default=50, help="Stack-only steps to descend and place on cubeB.")
    parser.add_argument("--stack-release-steps", type=int, default=10, help="Stack-only steps to open gripper on cubeB.")
    parser.add_argument("--stack-retreat-steps", type=int, default=10, help="Stack-only steps to retreat upward after release.")
    parser.add_argument("--pre-grasp-height", type=float, default=0.1)
    parser.add_argument("--grasp-height", type=float, default=0.01)
    parser.add_argument("--close-descend", type=float, default=0.015)
    parser.add_argument("--lift-height", type=float, default=0.20)
    parser.add_argument("--stack-approach-height", type=float, default=0.10, help="Stack-only height above cubeB before descend.")
    parser.add_argument("--stack-place-height", type=float, default=0.045, help="Stack-only placement target height offset from cubeB center.")
    parser.add_argument("--stack-retreat-height", type=float, default=0.12, help="Stack-only retreat target height offset from cubeB center.")
    parser.add_argument("--pos-gain", type=float, default=2.5)
    parser.add_argument("--pos-scale", type=float, default=0.05)
    parser.add_argument("--arm-clip", type=float, default=1.0)
    parser.add_argument("--baseline-eps", type=float, default=1e-6)
    parser.add_argument("--contact-threshold", type=float, default=1e-3)
    parser.add_argument("--touch-nchannel", type=int, default=None, help="Override XML nchannel.")
    parser.add_argument("--touch-size-x", type=int, default=None, help="Override XML touch-grid size x.")
    parser.add_argument("--touch-size-y", type=int, default=None, help="Override XML touch-grid size y.")
    parser.add_argument("--touch-fov-x", type=float, default=None, help="Override XML FOV x (deg).")
    parser.add_argument("--touch-fov-y", type=float, default=None, help="Override XML FOV y (deg).")
    parser.add_argument("--touch-gamma", type=float, default=None, help="Override XML gamma.")
    parser.add_argument("--plot-path", type=str, default="touch_grid_lift.png")
    parser.add_argument("--live-plot", action="store_true", help="Show live touch-grid heatmaps and timeseries.")
    parser.add_argument("--live-plot-every", type=int, default=5, help="Update live plot every N steps.")
    parser.add_argument("--live-channel", type=int, default=0, help="Touch-grid channel index to show live.")
    parser.add_argument("--hold-live-plot", action="store_true", help="Keep live plot window open after run.")
    parser.add_argument("--debug-contacts-every", type=int, default=0, help="Print contact pairs every N steps (0 disables).")
    parser.add_argument("--debug-contact-limit", type=int, default=6, help="Max contact rows printed per debug update.")
    parser.add_argument("--debug-sites-every", type=int, default=0, help="Print touch site poses and -z axes every N steps (0 disables).")
    parser.add_argument("--debug-progress-every", type=int, default=20, help="Print phase / step heartbeat every N steps (0 disables).")
    parser.add_argument(
        "--stack-contact-summary",
        action="store_true",
        help="When task is Stack, print cubeA/cubeB contact pair counts alongside contact debug.",
    )
    parser.add_argument("--save-video", action="store_true", help="Save paper-style tactile video (red=normal, green arrows=shear).")
    parser.add_argument("--video-path", type=str, default="touch_grid_tactile.mp4", help="Output path for tactile video.")
    parser.add_argument("--video-fps", type=int, default=20, help="FPS for tactile video.")
    parser.add_argument("--video-camera-width", type=int, default=320, help="Agentview column width in saved video.")
    parser.add_argument("--video-camera-height", type=int, default=240, help="Agentview column height in saved video.")
    parser.add_argument("--video-camera-name", type=str, default="agentview", help="Camera name used for the left video column.")
    parser.add_argument("--tactile-resolution", type=int, default=30, help="Pixels per taxel in tactile visualization.")
    parser.add_argument("--tactile-nf-threshold", type=float, default=1e-6, help="Normal force threshold for tactile display.")
    parser.add_argument("--tactile-sf-threshold", type=float, default=1e-6, help="Shear force threshold for arrow display.")
    cfg = TouchGridDemoConfig.from_parser(parser)
    runner = TouchGridDemoRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
