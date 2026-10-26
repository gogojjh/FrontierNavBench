# Upstream: src/SRU-Odin/Deployment/src/sru_nav_go2/constants.py; original SHA256: b700485a84630daedd7443167dcb6ec446b6e48c11f10c0469f60c1c6ac58e54.
"""Constants and configuration values for the Go2 + Odin1 SRU navigation port.

Defaults are tuned conservatively for Unitree Go2 (no wheels, weaker lateral
ability) versus the original B2W (wheeled). Override at runtime through the
companion YAML / launch parameters.
"""

# ---------------------------------------------------------------------------
# Control parameters
# ---------------------------------------------------------------------------
DEFAULT_CONTROL_FREQUENCY = 5.0       # Hz (matches training)
DEFAULT_MIN_DEPTH = 0.25               # meters
DEFAULT_MAX_DEPTH = 10.0               # meters
ARRIVE_GOAL_THRESHOLD = 0.75           # meters
# Distance threshold (meters) for resetting the LSTM hidden state and the
# cmd_vel low-pass filter on a large goal jump. See reset_hidden_on_goal_jump
# in Deployment/config/sru_nav.yaml for the full rationale and current default.
GOAL_JUMP_RESET_DISTANCE = 1.0         # meters

# ---------------------------------------------------------------------------
# Policy output scaling. These are fallback defaults used only when the
# matching ROS param (policy_scale / lateral_velocity_scale /
# low_pass_filter_coef in sru_nav.yaml) is absent; the yaml values are what
# actually run in deployment. Keep them in sync with sru_nav.yaml so any
# launch path that skips the yaml doesn't silently regress the vx:omega
# ratio (turning radius) back to the old, oscillation-prone values.
# Original B2W training scale was [1.5, 1.0, 1.0].
# ---------------------------------------------------------------------------
POLICY_SCALE = [0.60, 0.27, 0.60]      # [linear_x, linear_y, angular_z]
LATERAL_VELOCITY_SCALE = 1.0           # extra damping on linear_y

# Low-pass filter coefficients (alpha): output = a*new + (1-a)*prev
LOW_PASS_FILTER_COEF = [0.5, 0.5, 0.5]

# ---------------------------------------------------------------------------
# Timer intervals
# ---------------------------------------------------------------------------
TARGET_VECTOR_PUBLISH_INTERVAL = 0.2   # 5 Hz

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
TWIST_MARKER_SCALE = 5.0
TWIST_MARKER_ID = 0
TARGET_VECTOR_MARKER_ID = 1

# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------
GRAVITY_MAGNITUDE = 9.81               # m/s^2
