"""Declare arrival immediately; T-I0 expects zero success."""

from frontier_nav_bench.interfaces import Action, EpisodeSpec, MethodContext, Observation
from frontier_nav_bench.methods.debug import DebugMethodBase

HOLD_S = 0.1  # Zero velocity when declaring done, using the minimum hold duration


class StopNowMethod(DebugMethodBase):
    def reset(self, spec: EpisodeSpec, obs: Observation, ctx: MethodContext) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        return Action(0.0, 0.0, 0.0, HOLD_S, done=True, note="stop_now")


def make(method_id: str) -> StopNowMethod:
    return StopNowMethod(method_id)
