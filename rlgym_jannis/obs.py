from typing import List, Dict, Any, Tuple
import numpy as np
from rlgym.api import ObsBuilder, AgentID
from rlgym_compat.game_state import GameState
from rlgym_compat.car import Car
from rlgym_compat.common_values import ORANGE_TEAM

# Reihenfolge wie rlgym_compat.common_values.BOOST_LOCATIONS (= Reihenfolge in RocketSim, geprueft).
BIG_PAD_INDICES = (3, 4, 15, 18, 29, 30)
BIG_PAD_COOLDOWN = 10.0
SMALL_PAD_COOLDOWN = 4.0


def rlbot_pad_timers_to_cooldowns(timers: np.ndarray) -> np.ndarray:
    """RLBot liefert Sekunden SEIT dem Einsammeln (0 = aktiv), RocketSim im Training die
    verbleibende Abklingzeit BIS das Pad wieder aktiv ist (0 = aktiv). Hier: RLBot -> RocketSim."""
    respawn = np.full(len(timers), SMALL_PAD_COOLDOWN, dtype=np.float32)
    respawn[list(BIG_PAD_INDICES)] = BIG_PAD_COOLDOWN
    return np.where(timers > 0, np.maximum(respawn - timers, 0), 0).astype(np.float32)


class DefaultObs(ObsBuilder[AgentID, np.ndarray, GameState, Tuple[str, int]]):
    """Nachbau von rlgym.rocket_league.obs_builders.DefaultObs (rlgym 2.0.1) fuer rlgym_compat.

    Muss exakt dieselben Werte liefern wie im Training - inklusive pad_timer_coef und boost_coef.
    """

    def __init__(self, zero_padding=None, pos_coef=1/2300, ang_coef=1/np.pi, lin_vel_coef=1/2300, ang_vel_coef=1/np.pi,
                 pad_timer_coef=1/10, boost_coef=1/100):
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        self.PAD_TIMER_COEF = pad_timer_coef
        self.BOOST_COEF = boost_coef
        self.zero_padding = zero_padding

    def get_obs_space(self, agent):
        return 'real', -1

    def reset(self, agents, initial_state, shared_info) -> None:
        pass

    def build_obs(self, agents: List[AgentID], state: GameState, shared_info) -> Dict[AgentID, np.ndarray]:
        return {agent: self._build_obs(agent, state) for agent in agents}

    def _build_obs(self, agent: AgentID, state: GameState) -> np.ndarray:
        car = state.cars[agent]
        pads = rlbot_pad_timers_to_cooldowns(state.boost_pad_timers)
        if car.team_num == ORANGE_TEAM:
            ball = state.inverted_ball
            pads = pads[::-1]
        else:
            ball = state.ball

        obs = [
            ball.position * self.POS_COEF,
            ball.linear_velocity * self.LIN_VEL_COEF,
            ball.angular_velocity * self.ANG_VEL_COEF,
            pads * self.PAD_TIMER_COEF,
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
            [car.boost_amount * self.BOOST_COEF, car.demo_respawn_timer, int(car.on_ground),
             int(car.is_boosting), int(car.is_supersonic)],
        ])