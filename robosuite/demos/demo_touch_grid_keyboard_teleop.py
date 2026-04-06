"""
Keyboard teleop demo for Panda + touch_grid logging, plots, and optional video.

This demo:
1) Builds a temporary Panda gripper XML with configurable touch_grid sensors.
2) Runs a selected robosuite environment with keyboard end-effector teleoperation.
3) Logs touch_grid observations on every teleop step.
4) Optionally validates baseline / interaction response and saves plot artifacts.
"""

import argparse
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure local checkout imports when script is run as:
# python robosuite/demos/demo_touch_grid_keyboard_teleop.py
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robosuite.controllers.composite.composite_controller import WholeBody  # noqa: E402
from robosuite.devices import Keyboard  # noqa: E402
from robosuite.models.grippers import register_gripper  # noqa: E402
from robosuite.models.grippers.gripper_model import GripperModel  # noqa: E402
from robosuite.models.grippers.panda_gripper import PandaGripper  # noqa: E402
from robosuite.utils.input_utils import choose_environment  # noqa: E402
from robosuite.utils.mjcf_utils import xml_path_completion  # noqa: E402
from robosuite.wrappers import VisualizationWrapper  # noqa: E402

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
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.baseline_steps < 1:
        raise ValueError("--baseline-steps must be >= 1")


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

    if args.pad_contact_only:
        # Ensure contact is resolved on the discretized pad geoms instead of the coarse finger mesh.
        # Otherwise, mesh contact can dominate and the touch-grid signal appears sparse / missing.
        for geom_name in ("finger1_collision", "finger2_collision"):
            geom = root.find(f".//geom[@name='{geom_name}']")
            if geom is not None:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    # Keep all mesh references valid when writing this XML to a temp directory.
    for node in root.findall("./asset/*[@file]"):
        rel = node.get("file")
        if rel is not None:
            node.set("file", str((base_dir / rel).resolve()))

    tree.write(dst_xml_path, encoding="unicode")


def _register_touch_grid_gripper(xml_path):
    class PandaTouchGridTeleopGripper(PandaGripper):
        def __init__(self, idn=0):
            GripperModel.__init__(self, xml_path, idn=idn)

    register_gripper(PandaTouchGridTeleopGripper)
    return PandaTouchGridTeleopGripper.__name__


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


def _print_contact_debug(env, step_idx, limit, name_filter=None):
    ncon = int(env.sim.data.ncon)
    if ncon == 0:
        print(f"[step {step_idx}] contacts: none")
        return
    print(f"[step {step_idx}] contacts: {ncon}")

    rows = []
    for i in range(ncon):
        con = env.sim.data.contact[i]
        g1 = env.sim.model.geom_id2name(int(con.geom1)) or f"geom{int(con.geom1)}"
        g2 = env.sim.model.geom_id2name(int(con.geom2)) or f"geom{int(con.geom2)}"
        rows.append((i, g1, g2, float(con.dist)))

    if name_filter:
        keys = [k.strip().lower() for k in name_filter.split(",") if k.strip()]
        if keys:
            matched = [
                row
                for row in rows
                if any(k in row[1].lower() or k in row[2].lower() for k in keys)
            ]
            print(f"  matching contact filter ({name_filter}): {len(matched)}")
            if matched:
                for i, g1, g2, dist in matched[:limit]:
                    print(f"  - {i}: {g1} <-> {g2}, dist={dist:.6f}")
                return
            print("  no filtered contacts found; printing first contacts instead.")

    for i, g1, g2, dist in rows[:limit]:
        print(f"  - {i}: {g1} <-> {g2}, dist={dist:.6f}")


def _print_sensor_debug(step_idx, latest_sensor_values):
    print(f"[step {step_idx}] touch-grid summary:")
    for name, val in latest_sensor_values.items():
        max_abs = float(np.max(np.abs(val)))
        mean_abs = float(np.mean(np.abs(val)))
        print(f"  - {name}: max|x|={max_abs:.6f}, mean|x|={mean_abs:.6f}")


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


class _LiveTouchGridPlot:
    """Live matplotlib plot with paper-style tactile visualization."""

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
        plt.ion()
        self.fig, all_axes = plt.subplots(2, max(n_sensors, 1), figsize=(6 * max(n_sensors, 1), 8), squeeze=False)
        self.ax_ts = all_axes[0, 0]
        for c in range(1, n_sensors):
            all_axes[0, c].set_visible(False)

        res = args.tactile_resolution
        H, W = args.touch_size_y, args.touch_size_x
        zero_img = np.zeros((H * res, W * res, 3), dtype=np.uint8)
        self.tac_im = []
        for s, name in enumerate(self.sensor_names):
            ax = all_axes[1, s]
            im = ax.imshow(zero_img)
            ax.set_title(f"{name}\n(red=normal, arrows=shear)", fontsize=8)
            ax.axis("off")
            self.tac_im.append(im)

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

        for s, name in enumerate(self.sensor_names):
            grid = sensor_values[name].reshape(n_ch, H, W)
            normal = grid[0]
            shear = np.stack([grid[1], grid[2]], axis=-1) if n_ch >= 3 else np.zeros((H, W, 2))
            tac_img = _compute_tactile_shear_image(
                normal,
                shear,
                normal_force_threshold=self.args.tactile_nf_threshold,
                shear_force_threshold=self.args.tactile_sf_threshold,
                resolution=self.args.tactile_resolution,
            )
            self.tac_im[s].set_data(tac_img[:, :, ::-1])

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)

    def close(self, hold=False):
        if hold:
            self.plt.ioff()
            self.plt.show()
        else:
            self.plt.close(self.fig)


def _compute_tactile_shear_image(
    tactile_normal_force,
    tactile_shear_force,
    normal_force_threshold=0.00008,
    shear_force_threshold=0.0005,
    resolution=30,
):
    import cv2

    H, W = tactile_normal_force.shape
    img = np.zeros((H * resolution, W * resolution, 3), dtype=np.uint8)

    nf_max = np.max(np.abs(tactile_normal_force))
    if nf_max < normal_force_threshold:
        nf_max = 1.0
    sf_max = np.max(np.abs(tactile_shear_force))
    if sf_max < shear_force_threshold:
        sf_max = 1.0

    for r in range(H):
        for c in range(W):
            nf = abs(tactile_normal_force[r, c])
            intensity = min(nf / nf_max, 1.0)
            red = int(255 * intensity)
            cy = (H - 1 - r) * resolution + resolution // 2
            cx = c * resolution + resolution // 2
            half = resolution // 2 - 1
            cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (0, 0, red), cv2.FILLED)

            sx, sy = tactile_shear_force[r, c]
            shear_mag = np.sqrt(sx ** 2 + sy ** 2)
            if shear_mag > shear_force_threshold:
                arrow_len = min(shear_mag / sf_max, 1.0) * (resolution * 0.4)
                dx = int(arrow_len * sx / shear_mag)
                dy = int(-arrow_len * sy / shear_mag)
                cv2.arrowedLine(
                    img,
                    (cx, cy),
                    (cx + dx, cy + dy),
                    (0, 255, 0),
                    thickness=max(1, resolution // 10),
                    tipLength=0.3,
                )
    return img


def _validate_traces_teleop(traces, sensor_meta, args):
    if not sensor_meta:
        raise RuntimeError("No touch_grid sensors found in simulation model.")

    expected_dim = args.touch_nchannel * args.touch_size_x * args.touch_size_y
    for name, _, dim in sensor_meta:
        if dim != expected_dim:
            raise RuntimeError(f"Sensor {name} dim={dim}, expected={expected_dim} from configured touch-grid spec.")

    n_steps = len(traces["phase"])
    if n_steps <= args.baseline_steps:
        raise RuntimeError(
            f"Need more than baseline steps for validation: n_steps={n_steps}, baseline_steps={args.baseline_steps}"
        )

    any_contact = False
    stats = {}
    for name, _, _ in sensor_meta:
        values = np.array(traces["sensors"][name])
        if not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite values found in sensor {name}.")

        baseline_vals = values[: args.baseline_steps]
        interaction_vals = values[args.baseline_steps :]
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
            f"No touch_grid sensor exceeded contact threshold {args.contact_threshold:.6f} "
            f"after baseline window ({args.baseline_steps} steps)."
        )
    return stats


def _plot_results(traces, sensor_meta, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(args.plot_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    suffix = out_path.suffix or ".png"
    parent = out_path.parent

    steps = np.arange(len(traces["phase"]))
    interaction_start = min(args.baseline_steps, len(steps))
    interaction_idx = np.arange(interaction_start, len(steps))

    ts_fig, ts_ax = plt.subplots(figsize=(12, 5))
    for name, _, _ in sensor_meta:
        vals = np.array(traces["sensors"][name])
        max_abs = np.max(np.abs(vals), axis=1)
        mean_abs = np.mean(np.abs(vals), axis=1)
        ts_ax.plot(steps, max_abs, label=f"{name} max|x|")
        ts_ax.plot(steps, mean_abs, linestyle="--", alpha=0.8, label=f"{name} mean|x|")
    if interaction_start < len(steps):
        ts_ax.axvline(interaction_start, linestyle=":", color="k", alpha=0.7, label="interaction_start")
    ts_ax.set_title("touch_grid timeseries")
    ts_ax.set_xlabel("Step")
    ts_ax.set_ylabel("Magnitude")
    ts_ax.legend(loc="upper right", fontsize=8)
    ts_ax.grid(True, alpha=0.3)
    ts_fig.tight_layout()
    ts_path = parent / f"{stem}_timeseries{suffix}"
    ts_fig.savefig(ts_path, dpi=160)
    plt.close(ts_fig)

    n_ch = args.touch_nchannel
    for name, _, _ in sensor_meta:
        vals = np.array(traces["sensors"][name])
        all_grids = vals.reshape(len(vals), n_ch, args.touch_size_y, args.touch_size_x)
        if len(interaction_idx) > 0:
            int_vals = vals[interaction_idx]
            peak_rel = int(np.argmax(np.max(np.abs(int_vals), axis=1)))
            peak_idx = int(interaction_idx[peak_rel])
        else:
            peak_idx = len(vals) - 1

        for label, idx in [("final", len(vals) - 1), (f"peak@{peak_idx}", peak_idx)]:
            grid = all_grids[idx]
            normal = grid[0]
            shear = np.stack([grid[1], grid[2]], axis=-1) if n_ch >= 3 else np.zeros((args.touch_size_y, args.touch_size_x, 2))
            tac_img = _compute_tactile_shear_image(
                normal,
                shear,
                normal_force_threshold=args.tactile_nf_threshold,
                shear_force_threshold=args.tactile_sf_threshold,
                resolution=args.tactile_resolution,
            )
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


def _render_tactile_shear_video(traces, sensor_meta, args):
    import cv2

    out_path = Path(args.video_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ch = args.touch_nchannel
    H, W = args.touch_size_y, args.touch_size_x
    res = args.tactile_resolution

    all_grids = {}
    for name, _, _ in sensor_meta:
        vals = np.array(traces["sensors"][name])
        all_grids[name] = vals.reshape(len(vals), n_ch, H, W)

    n_frames = len(traces["phase"])
    camera_frames = traces.get("agentview_frames", [])
    has_camera_frames = len(camera_frames) == n_frames and n_frames > 0

    tactile_sensor_names = [name for name, _, _ in sensor_meta]
    if len(tactile_sensor_names) < 2:
        raise RuntimeError(f"Need at least 2 touch_grid sensors for 3-column output (got {len(tactile_sensor_names)}).")
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
        if has_camera_frames:
            cam_rgb = camera_frames[t]
            if cam_rgb.shape[:2] != (cam_h, cam_w):
                cam_rgb = cv2.resize(cam_rgb, (cam_w, cam_h), interpolation=cv2.INTER_AREA)
            canvas[:cam_h, 0:cam_w] = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
        else:
            cv2.putText(
                canvas,
                "camera capture disabled",
                (10, cam_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (180, 180, 180),
                1,
            )

        x_offset = cam_w + gap
        for name in tactile_sensor_names:
            grid = all_grids[name][t]
            normal = grid[0]
            shear = np.stack([grid[1], grid[2]], axis=-1) if n_ch >= 3 else np.zeros((H, W, 2))
            img = _compute_tactile_shear_image(
                normal,
                shear,
                normal_force_threshold=args.tactile_nf_threshold,
                shear_force_threshold=args.tactile_sf_threshold,
                resolution=res,
            )
            img = cv2.resize(img, (tactile_panel_w, cam_h), interpolation=cv2.INTER_NEAREST)
            canvas[:cam_h, x_offset : x_offset + tactile_panel_w] = img
            x_offset += tactile_panel_w + gap

        cv2.putText(
            canvas,
            f"step {t}  phase={traces['phase'][t]}",
            (5, cam_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        writer.write(canvas)
    writer.release()
    print(f"Saved tactile shear video to {out_path}")


@dataclass
class TeleopDemoConfig:
    args: argparse.Namespace

    @classmethod
    def from_parser(cls, parser):
        args = parser.parse_args()
        # Backward compatibility: allow legacy --task flag.
        if args.environment is None and args.task is not None:
            args.environment = args.task
        # If still unspecified, match demo_control.py behavior and prompt interactively.
        if args.environment is None:
            args.environment = choose_environment()
        base_xml = xml_path_completion("grippers/panda_gripper_tactile.xml")
        _resolve_touch_config(args, base_xml)
        _validate_touch_config(args)
        np.random.seed(args.seed)
        return cls(args=args)


class TouchGridTeleopRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.args = cfg.args
        self.env = None
        self.live_plot = None
        self._tmpdir_ctx = None

    def setup_env(self):
        args = self.args
        base_xml = xml_path_completion("grippers/panda_gripper_tactile.xml")
        self._tmpdir_ctx = tempfile.TemporaryDirectory(prefix="touch_grid_keyboard_teleop_")
        tmpdir = self._tmpdir_ctx.__enter__()
        patched_xml = str(Path(tmpdir) / "panda_gripper_touch_grid.xml")
        _ensure_touch_grid_xml(base_xml, patched_xml, args)
        gripper_name = _register_touch_grid_gripper(patched_xml)

        # On many Linux/X11 setups, enabling both onscreen viewer and offscreen camera rendering
        # can trigger driver-level segfaults. Keep offscreen disabled by default for keyboard teleop.
        capture_camera = bool(args.save_video and args.video_include_camera and not args.render)
        if args.save_video and args.video_include_camera and args.render:
            print("WARNING: disabling camera column capture to avoid OpenGL segfault in render mode.")
        self._capture_camera_frames = capture_camera

        self.env = suite.make(
            args.environment,
            robots="Panda",
            gripper_types=gripper_name,
            controller_configs=None,
            has_renderer=args.render,
            has_offscreen_renderer=capture_camera,
            use_camera_obs=False,
            control_freq=args.control_freq,
            horizon=max(args.max_steps + 5, 50),
            ignore_done=True,
        )
        self.env = VisualizationWrapper(self.env, indicator_configs=None)

    def run(self):
        args = self.args
        if not args.render:
            raise ValueError("Keyboard teleop requires --render.")

        print(
            "touch_grid config: "
            f"nchannel={args.touch_nchannel}, size=({args.touch_size_x},{args.touch_size_y}), "
            f"fov=({args.touch_fov_x},{args.touch_fov_y}), gamma={args.touch_gamma}"
        )
        print(f"selected environment: {args.environment}")
        print(
            "teleop controls: arrows/.;/e/r/y/h/o/p for motion, space for gripper toggle, "
            "Ctrl+q to reset/exit episode."
        )

        self.setup_env()
        try:
            obs = self.env.reset()
            self.env.render()

            sensor_meta = _get_touch_sensor_meta(self.env.sim)
            site_meta = _get_touch_site_debug_meta(self.env.sim)
            self.live_plot = _LiveTouchGridPlot(sensor_meta, args) if args.live_plot else None

            traces = {
                "phase": [],
                "eef_pos": [],
                "sensors": {name: [] for name, _, _ in sensor_meta},
                "agentview_frames": [],
            }

            eef_key = _pick_obs_key(obs, "eef_pos")
            device = Keyboard(env=self.env, pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
            self.env.viewer.add_keypress_callback(device.on_press)
            device.start_control()

            all_prev_gripper_actions = [
                {
                    f"{robot_arm}_gripper": np.repeat([0], robot.gripper[robot_arm].dof)
                    for robot_arm in robot.arms
                    if robot.gripper[robot_arm].dof > 0
                }
                for robot in self.env.robots
            ]

            for step_idx in range(args.max_steps):
                start = time.time()
                active_robot = self.env.robots[device.active_robot]
                input_ac_dict = device.input2action()
                if input_ac_dict is None:
                    print("Received keyboard reset/exit command. Ending teleop rollout.")
                    break

                action_dict = deepcopy(input_ac_dict)
                for arm in active_robot.arms:
                    if isinstance(active_robot.composite_controller, WholeBody):
                        controller_input_type = active_robot.composite_controller.joint_action_policy.input_type
                    else:
                        controller_input_type = active_robot.part_controllers[arm].input_type

                    if controller_input_type == "delta":
                        action_dict[arm] = input_ac_dict[f"{arm}_delta"]
                    elif controller_input_type == "absolute":
                        action_dict[arm] = input_ac_dict[f"{arm}_abs"]
                    else:
                        raise ValueError(f"Unsupported controller input type: {controller_input_type}")

                env_action = [robot.create_action_vector(all_prev_gripper_actions[i]) for i, robot in enumerate(self.env.robots)]
                env_action[device.active_robot] = active_robot.create_action_vector(action_dict)
                env_action = np.concatenate(env_action)
                low, high = self.env.action_spec
                env_action = np.clip(env_action, low, high)
                for gripper_ac in all_prev_gripper_actions[device.active_robot]:
                    all_prev_gripper_actions[device.active_robot][gripper_ac] = action_dict[gripper_ac]

                obs, _, _, _ = self.env.step(env_action)
                self.env.render()

                traces["phase"].append("teleop")
                traces["eef_pos"].append(obs[eef_key].copy())
                latest = {}
                for name, adr, dim in sensor_meta:
                    val = self.env.sim.data.sensordata[adr : adr + dim].copy()
                    traces["sensors"][name].append(val)
                    latest[name] = val

                if args.save_video and self._capture_camera_frames:
                    frame = self.env.sim.render(
                        width=args.video_camera_width,
                        height=args.video_camera_height,
                        camera_name=args.video_camera_name,
                    )
                    traces["agentview_frames"].append(np.flipud(frame).copy())

                if args.debug_progress_every > 0 and step_idx % args.debug_progress_every == 0:
                    print(f"[step {step_idx}] phase=teleop")
                if args.debug_contacts_every > 0 and step_idx % args.debug_contacts_every == 0:
                    _print_contact_debug(
                        self.env,
                        step_idx,
                        args.debug_contact_limit,
                        name_filter=args.debug_contact_filter,
                    )
                if args.debug_sites_every > 0 and step_idx % args.debug_sites_every == 0:
                    _print_site_debug(self.env, step_idx, site_meta)
                if args.debug_sensors_every > 0 and step_idx % args.debug_sensors_every == 0:
                    _print_sensor_debug(step_idx, latest)
                if self.live_plot is not None and step_idx % max(args.live_plot_every, 1) == 0:
                    self.live_plot.update(step_idx, "teleop", latest)

                if args.max_fr is not None:
                    elapsed = time.time() - start
                    sleep_time = 1 / args.max_fr - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            qstate = np.concatenate([self.env.sim.data.qpos.ravel(), self.env.sim.data.qvel.ravel()])
            if not np.isfinite(qstate).all():
                raise RuntimeError("Simulation produced non-finite state values (NaN / inf).")

            if args.validate:
                stats = _validate_traces_teleop(traces, sensor_meta, args)
                print("touch_grid validation: PASS")
                for sensor_name, s in stats.items():
                    print(
                        f"- {sensor_name}: baseline_max={s['baseline_max_abs']:.6f}, "
                        f"interaction_max={s['interaction_max_abs']:.6f}, std={s['std']:.6f}"
                    )
            else:
                print("touch_grid validation: SKIPPED (--no-validate)")

            if len(traces["phase"]) > 0:
                _plot_results(traces, sensor_meta, args)
                if args.save_video:
                    _render_tactile_shear_video(traces, sensor_meta, args)
            else:
                print("No teleop steps recorded; skipping plot/video output.")
        finally:
            if self.live_plot is not None:
                self.live_plot.close(hold=args.hold_live_plot)
            if self.env is not None:
                self.env.close()
            if self._tmpdir_ctx is not None:
                self._tmpdir_ctx.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--environment",
        type=str,
        default=None,
        help="Robosuite environment to run during teleop. If omitted, an interactive chooser is shown.",
    )
    parser.add_argument("--task", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable onscreen rendering. Required for keyboard teleop.",
    )
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum teleop steps to run before auto-stop.")
    parser.add_argument("--max_fr", type=int, default=20, help="Cap simulation loop at this framerate (None to disable).")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="Scale factor for keyboard translation input.")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="Scale factor for keyboard rotation input.")
    parser.add_argument("--baseline-steps", type=int, default=120, help="Initial teleop samples used as baseline window.")
    parser.add_argument("--baseline-eps", type=float, default=1e-6)
    parser.add_argument("--contact-threshold", type=float, default=1e-3)
    parser.add_argument("--validate", action=argparse.BooleanOptionalAction, default=True, help="Enable touch-grid validation.")
    parser.add_argument("--touch-nchannel", type=int, default=None, help="Override XML nchannel.")
    parser.add_argument("--touch-size-x", type=int, default=None, help="Override XML touch-grid size x.")
    parser.add_argument("--touch-size-y", type=int, default=None, help="Override XML touch-grid size y.")
    parser.add_argument("--touch-fov-x", type=float, default=None, help="Override XML FOV x (deg).")
    parser.add_argument("--touch-fov-y", type=float, default=None, help="Override XML FOV y (deg).")
    parser.add_argument("--touch-gamma", type=float, default=None, help="Override XML gamma.")
    parser.add_argument(
        "--pad-contact-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable coarse finger mesh collision so contacts resolve on tactile pad geoms.",
    )
    parser.add_argument("--plot-path", type=str, default="touch_grid_teleop.png")
    parser.add_argument("--live-plot", action="store_true", help="Show live touch-grid heatmaps and timeseries.")
    parser.add_argument("--live-plot-every", type=int, default=5, help="Update live plot every N steps.")
    parser.add_argument("--hold-live-plot", action="store_true", help="Keep live plot window open after run.")
    parser.add_argument("--debug-contacts-every", type=int, default=0, help="Print contact pairs every N steps (0 disables).")
    parser.add_argument("--debug-contact-limit", type=int, default=6, help="Max contact rows printed per debug update.")
    parser.add_argument(
        "--debug-contact-filter",
        type=str,
        default="",
        help="Optional comma-separated geom-name filter for contact debug (e.g. finger,pad,gripper,nut).",
    )
    parser.add_argument("--debug-sites-every", type=int, default=0, help="Print touch site poses every N steps (0 disables).")
    parser.add_argument("--debug-sensors-every", type=int, default=0, help="Print per-sensor max/mean every N steps (0 disables).")
    parser.add_argument("--debug-progress-every", type=int, default=50, help="Print teleop progress every N steps (0 disables).")
    parser.add_argument("--save-video", action="store_true", help="Save paper-style tactile video.")
    parser.add_argument("--video-path", type=str, default="touch_grid_teleop_tactile.mp4")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-camera-width", type=int, default=320)
    parser.add_argument("--video-camera-height", type=int, default=240)
    parser.add_argument("--video-camera-name", type=str, default="agentview")
    parser.add_argument(
        "--video-include-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include agentview camera column in saved video (disabled by default in render mode for stability).",
    )
    parser.add_argument("--tactile-resolution", type=int, default=30)
    parser.add_argument("--tactile-nf-threshold", type=float, default=1e-6)
    parser.add_argument("--tactile-sf-threshold", type=float, default=1e-6)

    cfg = TeleopDemoConfig.from_parser(parser)
    runner = TouchGridTeleopRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
