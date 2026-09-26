from typing import List, Dict, Any, Tuple
import numpy as np
from rlgym.api import ObsBuilder, AgentID
from rlgym_compat.game_state import GameState
from rlgym_compat.car import Car
from rlgym_compat.common_values import ORANGE_TEAM


class DefaultObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    def __init__(self, zero_padding=None, pos_coef=1/2300, ang_coef=1/np.pi, lin_vel_coef=1/2300, ang_vel_coef=1/np.pi):
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.zero_padding = zero_padding

    def get_obs_space(self, agent):
        return 'real', -1

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info) -> Dict[AgentID, np.ndarray]:
        return {agent: self._build_obs(agent, state) for agent in agents}

    def _build_obs(self, agent: AgentID, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        if car.team_num == ORANGE_TEAM:
            ball = state.inverted_ball
            pads = state.inverted_boost_pad_timers
        else:
            ball = state.ball
            pads = state.boost_pad_timers

        obs = [
            ball.position * self.POS_COEF,
            ball.linear_velocity * self.LIN_VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF,
            pads,
            [car.is_holding_jump, car.handbrake, car.has_jumped, car.is_jumping,
             car.has_flipped, car.is_flipping, car.has_double_jumped, car.can_flip,
             car.air_time_since_jump],
        ]
        car_obs = self._car_obs(car, car.team_num == ORANGE_TEAM)
        obs.append(car_obs)

        allies, enemies = [], []
        for other, other_car in state.cars.items():
            if other == agent:
                continue
            (allies if other_car.team_num == car.team_num else enemies).append(
                self._car_obs(other_car, car.team_num == ORANGE_TEAM)
            )
        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _car_obs(self, car: Car, inverted: bool) -> np.ndarray:
        physics = car.inverted_physics if inverted else car.physics
        return np.concatenate([
            physics.position * self.POS_COEF,
            physics.forward,
            physics.up,
            physics.linear_velocity * self.LIN_VEL_COEF,
            physics.angular_velocity * self.ANG_VEL_COEF,
            [car.boost_amount, car.demo_respawn_timer, int(car.on_ground),
             int(car.is_boosting), int(car.is_supersonic)],
        ])