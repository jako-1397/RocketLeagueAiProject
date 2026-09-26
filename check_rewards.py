"""Schneller Plausibilitaetscheck der Reward-Gewichte eines Projekts - ohne Training.

Laesst zwei feste (nicht lernende) Test-Policies in RocketSim spielen und zeigt, wie viel
jede Reward-Komponente (bereits gewichtet) pro Sekunde Spielzeit beitraegt:
  - random: zufaellige Aktionen (so startet ein frisches Netz)
  - chase:  faehrt stumpf mit Boost auf den Ball zu (grobes "gewuenschtes" Anfaengerverhalten)
Faustregel: chase sollte deutlich mehr Reward bekommen als random, und keine einzelne
Hilfskomponente sollte alles andere ueberdecken.
Hinweis: Symmetrische Rewards (GoalReward, GoalViewReward, VelocityBallToGoalReward) heben
sich hier ueber beide Autos nahezu auf, weil beide dieselbe Test-Policy spielen.

Aufruf:  python check_rewards.py  (fragt nach dem Projekt, wie train.py)
"""
import sys
from collections import defaultdict

import numpy as np

from train import TICK_SKIP, TIMEOUT_SECONDS, choose_project, find_projects, load_module

SECONDS_PER_POLICY = 300

# Zeilen aus LookupTableAction: Vollgas + Boost, Lenkung links / geradeaus / rechts
CHASE_ACTIONS = {-1: 14, 0: 18, 1: 22}


class RecordingReward:
    """Wickelt die RewardFunction des Projekts ein und merkt sich jede gewichtete Komponente."""

    def __init__(self, reward_fn):
        from rlgym.rocket_league.reward_functions import CombinedReward

        if isinstance(reward_fn, CombinedReward):
            self.parts = list(zip(reward_fn.reward_fns, reward_fn.weights))
        else:
            self.parts = [(reward_fn, 1.0)]
        self.totals = defaultdict(float)

    @staticmethod
    def name(fn):
        return type(fn).__name__

    def reset(self, agents, initial_state, shared_info):
        for fn, _ in self.parts:
            fn.reset(agents, initial_state, shared_info)

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        combined = {agent: 0.0 for agent in agents}
        for i, (fn, weight) in enumerate(self.parts):
            for agent, value in fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info).items():
                combined[agent] += value * weight
                self.totals[(i, self.name(fn))] += value * weight
        return combined


def chase_action(car, ball):
    to_ball = ball.position - car.physics.position
    side = float(np.dot(to_ball, car.physics.right)) / (np.linalg.norm(to_ball) + 1e-6)
    steer = 0 if abs(side) < 0.1 else int(np.sign(side))
    return CHASE_ACTIONS[steer]


def run(project_dir, policy):
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.sim import RocketSimEngine

    rewards_module = load_module(f"{project_dir.name}_rewards", project_dir / "training" / "rewards.py")
    state_module = load_module(f"{project_dir.name}_state_setters", project_dir / "training" / "state_setters.py")
    recorder = RecordingReward(rewards_module.build_reward_fn())

    env = RLGym(
        state_mutator=state_module.build_state_mutator(),
        obs_builder=DefaultObs(zero_padding=None),
        action_parser=RepeatAction(LookupTableAction(), repeats=TICK_SKIP),
        reward_fn=recorder,
        termination_cond=GoalCondition(),
        truncation_cond=NoTouchTimeoutCondition(timeout_seconds=TIMEOUT_SECONDS),
        transition_engine=RocketSimEngine(),
    )

    rng = np.random.default_rng(0)
    steps = int(SECONDS_PER_POLICY * 120 / TICK_SKIP)
    n_agents = 0
    episodes, goals, touches = 1, 0, 0
    env.reset()
    for _ in range(steps):
        state = env.state
        n_agents = len(env.agents)
        if policy == "random":
            actions = {a: np.array([rng.integers(90)]) for a in env.agents}
        else:
            actions = {a: np.array([chase_action(state.cars[a], state.ball)]) for a in env.agents}
        _, _, terminated, truncated = env.step(actions)
        touches += sum(env.state.cars[a].ball_touches > 0 for a in env.agents)
        if any(terminated.values()) or any(truncated.values()):
            goals += env.state.goal_scored
            episodes += 1
            env.reset()
    env.close()

    # Werte pro Agent und Sekunde Spielzeit
    norm = 1.0 / (SECONDS_PER_POLICY * max(n_agents, 1))
    print(f"\n=== Policy '{policy}': {SECONDS_PER_POLICY}s Spielzeit, {episodes} Episoden, "
          f"{goals} Tore, {touches} Schritte mit Ballkontakt")
    total = 0.0
    for (i, name), value in sorted(recorder.totals.items()):
        print(f"  [{i}] {name:<30} {value * norm:+8.4f} / s")
        total += value * norm
    print(f"      {'SUMME':<30} {total:+8.4f} / s")


if __name__ == "__main__":
    projects = find_projects()
    project = choose_project(projects) if len(sys.argv) < 2 else next(p for p in projects if p.name == sys.argv[1])
    for policy in ("random", "chase"):
        run(project, policy)