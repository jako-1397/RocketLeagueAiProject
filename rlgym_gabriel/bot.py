from pathlib import Path

import numpy as np
import torch
from rlbot.flat import ControllerState, GamePacket, MatchPhase
from rlbot.managers import Bot
from rlgym_compat import GameState

from act import LookupTableAction
from discrete_policy import DiscreteFF
from obs import DefaultObs

TICK_SKIP = 8
LAYER_SIZES = (256, 256, 256)


def find_latest_checkpoint(project_dir: Path) -> Path:
    models_dir = project_dir / "training" / "models"
    checkpoints = [d for d in models_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not checkpoints:
        raise SystemExit(f"Kein Checkpoint in {models_dir} gefunden - erst trainieren!")
    latest = max(checkpoints, key=lambda d: int(d.name))
    return latest / "PPO_POLICY.pt"


class MLBot(Bot):
    def initialize(self):
        project_dir = Path(__file__).parent
        checkpoint_path = find_latest_checkpoint(project_dir)
        print(f"Lade Checkpoint: {checkpoint_path}")

        self.device = torch.device("cpu")
        state_dict = torch.load(checkpoint_path, map_location=self.device)

        obs_size = state_dict["model.0.weight"].shape[1]
        n_actions = state_dict[list(state_dict.keys())[-2]].shape[0]

        self.policy = DiscreteFF(obs_size, n_actions, LAYER_SIZES, self.device)
        self.policy.load_state_dict(state_dict)
        torch.set_num_threads(1)

        self.action_parser = LookupTableAction()
        self.obs_builder = DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([1 / 4096, 1 / 5120, 1 / 2044]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / 2300,
            ang_vel_coef=1 / np.pi,
        )
        self.game_state = GameState.create_compat_game_state(self.field_info)
        self.ticks = TICK_SKIP
        self.prev_control = ControllerState()

    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0 or packet.match_info.match_phase == MatchPhase.Ended:
            return ControllerState()

        self.game_state.update(packet)
        self.ticks += 1

        if self.ticks < TICK_SKIP:
            return self.prev_control
        self.ticks = 0

        obs = self.obs_builder.build_obs([self.index], self.game_state, {})[self.index]
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action_idx, _ = self.policy.get_action(obs_tensor, deterministic=True)

        parsed = self.action_parser.parse_actions(
            {self.index: np.array([action_idx])}, self.game_state, {}
        )[self.index]
        if parsed.ndim == 2:
            parsed = parsed[0]

        controls = ControllerState(
            throttle=float(parsed[0]),
            steer=float(parsed[1]),
            pitch=float(parsed[2]),
            yaw=float(parsed[3]),
            roll=float(parsed[4]),
            jump=parsed[5] > 0,
            boost=parsed[6] > 0,
            handbrake=parsed[7] > 0,
        )
        self.prev_control = controls
        return controls


if __name__ == "__main__":
    MLBot("rocketleagueaiproject/rlgym_gabriel").run()