from typing import Dict, Any, Tuple, List
import numpy as np
from rlgym.api import ActionParser, AgentID
from rlgym_compat.game_state import GameState


class LookupTableAction(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, Tuple[str, int]]):
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    def get_action_space(self, agent: AgentID) -> Tuple[str, int]:
        return 'discrete', len(self._lookup_table)

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def parse_actions(self, actions: Dict[AgentID, np.ndarray], state, shared_info) -> Dict[AgentID, np.ndarray]:
        parsed_actions = {}
        for agent, action in actions.items():
            if len(action.shape) == 2:
                action = action.squeeze(1)
            parsed_actions[agent] = self._lookup_table[action]
        return parsed_actions

    @staticmethod
    def make_lookup_table():
        actions = []
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:
                                continue
                            if pitch == roll == jump == 0:
                                continue
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        return np.array(actions)