from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BALL_MAX_SPEED,
    BLUE_GOAL_BACK,
    ORANGE_GOAL_BACK,
    TICKS_PER_SECOND,
)

_BLUE_GOAL_BACK = np.array(BLUE_GOAL_BACK, dtype=np.float32)
_ORANGE_GOAL_BACK = np.array(ORANGE_GOAL_BACK, dtype=np.float32)


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-6 else vec * 0.0


class _PerSecondReward(RewardFunction[AgentID, GameState, float]):
    """Basisklasse: `_rate()` liefert einen Wert pro Sekunde, der hier mit dt skaliert wird."""

    def __init__(self, include_negative_values: bool = True):
        self.include_negative_values = include_negative_values
        self.prev_tick = 0

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_tick = initial_state.tick_count

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        dt = (state.tick_count - self.prev_tick) / TICKS_PER_SECOND
        self.prev_tick = state.tick_count
        rewards = {}
        for agent in agents:
            rate = self._rate(agent, state)
            if not self.include_negative_values:
                rate = max(0.0, rate)
            rewards[agent] = float(rate * dt)
        return rewards

    def _rate(self, agent: AgentID, state: GameState) -> float:
        raise NotImplementedError


class FaceBallReward(_PerSecondReward):
    """Kosinus zwischen Fahrtrichtung (Nase) des Autos und Richtung zum Ball, in [-1, 1] pro Sekunde.

    Hilft ganz am Anfang, ueberhaupt auf den Ball auszurichten. Negativ = Ball im Ruecken.
    """

    def _rate(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent].physics
        return float(np.dot(car.forward, _unit(state.ball.position - car.position)))


class VelocityBallToGoalReward(_PerSecondReward):
    """Ballgeschwindigkeit in Richtung gegnerisches Tor, normiert auf BALL_MAX_SPEED, pro Sekunde.

    Wirkt unabhaengig davon, wer den Ball zuletzt beruehrt hat (bewertet die Spielsituation).
    Negativ, wenn der Ball Richtung eigenes Tor fliegt - das ist die symmetrische Bestrafung.
    """

    def _rate(self, agent: AgentID, state: GameState) -> float:
        target = _ORANGE_GOAL_BACK if state.cars[agent].is_blue else _BLUE_GOAL_BACK
        direction = _unit(target - state.ball.position)
        return float(np.dot(state.ball.linear_velocity, direction) / BALL_MAX_SPEED)


class InAirReward(_PerSecondReward):
    """1.0 pro Sekunde, solange das Auto nicht auf dem Boden ist.

    Nur mit sehr kleinem Gewicht verwenden: Soll verhindern, dass der Bot das Springen komplett
    "verlernt", weil Springen kurzfristig Tempo zum Ball kostet. Zu gross -> Dauer-Hopser.
    """

    def _rate(self, agent: AgentID, state: GameState) -> float:
        return 0.0 if state.cars[agent].on_ground else 1.0


class CooldownTouchReward(RewardFunction[AgentID, GameState, float]):
    """Ballberuehrung mit Abklingzeit, plus optionaler Bonus fuer die Ballbeschleunigung.

    Problem von `TouchReward`: Bei tick_skip=8 gibt es ~15 Schritte pro Sekunde. Schiebt der Bot
    den Ball nur vor sich her, kassiert er bis zu 15x pro Sekunde den vollen Touch-Reward
    ("Ball-Kuscheln"). Hier gibt es den festen `touch_reward` erst wieder, wenn seit der letzten
    belohnten Beruehrung `cooldown_seconds` vergangen sind.

    `acceleration_reward` belohnt zusaetzlich die Geschwindigkeitsaenderung des Balls
    (|dv| / BALL_MAX_SPEED) bei jeder Beruehrung - harte Schuesse zaehlen mehr als Anstupsen,
    und Schieben mit konstantem Tempo bringt fast nichts. Diesen Teil gibt es ohne Abklingzeit.
    """

    def __init__(self, touch_reward: float = 1.0, acceleration_reward: float = 0.0, cooldown_seconds: float = 0.5):
        self.touch_reward = touch_reward
        self.acceleration_reward = acceleration_reward
        self.cooldown_ticks = cooldown_seconds * TICKS_PER_SECOND
        self.prev_ball_vel = None
        self.last_rewarded_tick = {}

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.prev_ball_vel = initial_state.ball.linear_velocity.copy()
        self.last_rewarded_tick = {agent: -np.inf for agent in agents}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        ball_vel = state.ball.linear_velocity
        acceleration = float(np.linalg.norm(ball_vel - self.prev_ball_vel) / BALL_MAX_SPEED)
        self.prev_ball_vel = ball_vel.copy()

        rewards = {}
        for agent in agents:
            reward = 0.0
            if state.cars[agent].ball_touches > 0:
                reward += self.acceleration_reward * acceleration
                if state.tick_count - self.last_rewarded_tick.get(agent, -np.inf) >= self.cooldown_ticks:
                    reward += self.touch_reward
                    self.last_rewarded_tick[agent] = state.tick_count
            rewards[agent] = reward
        return rewards