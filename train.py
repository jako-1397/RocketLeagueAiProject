import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent
RLGYM_PREFIX = "rlgym_"

TICK_SKIP = 8
TIMEOUT_SECONDS = 10


def find_projects() -> list[Path]:
    return sorted(d for d in ROOT_DIR.glob(f"{RLGYM_PREFIX}*") if d.is_dir())


def choose_project(projects: list[Path]) -> Path:
    print("\nFuer welches Projekt soll trainiert werden?")
    for i, p in enumerate(projects):
        print(f"  [{i}] {p.name}")
    while True:
        raw = input("Auswahl: ").strip()
        if raw.isdigit() and int(raw) < len(projects):
            return projects[int(raw)]
        print("Ungueltige Eingabe, bitte erneut.")


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_rlgym_env(project_dir: Path):
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym_ppo.util import RLGymV2GymWrapper

    rewards_module = load_module(f"{project_dir.name}_rewards", project_dir / "training" / "rewards.py")
    state_setters_module = load_module(f"{project_dir.name}_state_setters", project_dir / "training" / "state_setters.py")

    reward_fn = rewards_module.build_reward_fn()
    state_mutator = state_setters_module.build_state_mutator()

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z,
        ]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
    )

    rlgym_env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=RepeatAction(LookupTableAction(), repeats=TICK_SKIP),
        reward_fn=reward_fn,
        termination_cond=GoalCondition(),
        truncation_cond=NoTouchTimeoutCondition(timeout=TIMEOUT_SECONDS),
        transition_engine=RocketSimEngine(),
    )
    return RLGymV2GymWrapper(rlgym_env)


if __name__ == "__main__":
    from rlgym_ppo import Learner

    projects = find_projects()
    if not projects:
        raise SystemExit("Keine rlgym_*-Ordner gefunden!")
    project_dir = choose_project(projects)

    models_dir = project_dir / "training" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    learner = Learner(
        env_create_function=functools.partial(build_rlgym_env, project_dir),
        n_proc=8,  # bei schwaecherer CPU reduzieren, z.B. auf 4
        checkpoints_save_folder=str(models_dir),
        checkpoint_load_folder=None,  # None = immer frisch starten, nicht automatisch fortsetzen
        save_every_ts=100_000,
        timestep_limit=1_000_000_000,
    )
    print(f"\nStarte Training fuer '{project_dir.name}' ...")
    learner.learn()