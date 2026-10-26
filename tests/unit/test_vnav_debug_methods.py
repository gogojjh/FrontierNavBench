"""The distributed stop-now probe declares arrival without moving the simulator."""

from vnav_benchmark import fairness, metrics
from vnav_benchmark.interfaces import Pose2D
from vnav_benchmark.methods import make_method
from vnav_benchmark.methods.debug import DebugTruth
from vnav_benchmark.sim.backend import KinematicBackend
from vnav_benchmark.sim.ctrl_client import InProcessClient
from vnav_benchmark.sim.ctrl_server import CtrlServer
from vnav_benchmark.sim_env import SimEnv


def test_stop_now_declares_at_first_step() -> None:
    client = InProcessClient(CtrlServer(KinematicBackend(), "inproc://stop-now"))
    client.hello()
    env = SimEnv(client, 90.0)
    obs = env.start(Pose2D(0.0, 0.0, 0.0))
    goal = Pose2D(4.0, 4.0, 0.0)
    method = make_method("debug-stop-now")
    profile = fairness.parse_method_id(method.method_id)
    assert profile.uses_truth and method.uses_truth
    method.attach_truth(DebugTruth(None, env.truth_pose, goal))
    method.reset(None, obs, None)
    action, _ = fairness.clamp_action(method.act(obs))
    assert action.done and env.sim_time == 0.0
    assert (action.vx, action.vy, action.wz) == (0.0, 0.0, 0.0)
    assert not metrics.within_pose(env.truth_pose(), goal)
    method.close()
    client.close()
