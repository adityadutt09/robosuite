# Compatibility shim for mimicgen
# robosuite 1.5+ renamed SingleArmEnv to ManipulationEnv
# This file provides backward compatibility

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv

# Alias for backward compatibility with mimicgen
SingleArmEnv = ManipulationEnv
