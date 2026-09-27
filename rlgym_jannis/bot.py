import json
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
# Im Training zieht rlgym-ppo die Aktion immer zufaellig aus der Policy-Verteilung. Bei hoher
# Entropy ist argmax (deterministic=True) oft eine Aktion, die so allein nie gespielt wurde.
DETERMINISTIC = True
# Schreibt alle ~2 s eine Diagnosezeile in die Konsole (eigenes Auto, Obs-Wertebereich).
DEBUG = False
DEBUG_INTERVAL_TICKS = 240


def find_latest_checkpoint(project_dir: Path) -> Path:
    models_dir = project_dir / "training" / "models"
    checkpoints = [d for d in models_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not checkpoints:
        raise SystemExit(f"Kein Checkpoint in {models_dir} gefunden - erst trainieren!")
    return max(checkpoints, key=lambda d: int(d.name))


def load_obs_standardization(checkpoint_dir: Path) -> tuple[float, float]:
    """Liest die Obs-Normierung von rlgym-ppo (standardize_obs=True) aus dem Checkpoint.

    rlgym-ppo rechnet im Training obs = clip((obs - mean[0]) / std[0], -5, 5) - mit nur dem
    ERSTEN Eintrag von Mittelwert und Standardabweichung fuer alle Werte (batched_agent_manager.py).
    "var" im JSON ist die Summe der quadrierten Abweichungen (Welford), nicht die Varianz.
    """
    with open(checkpoint_dir / "BOOK_KEEPING_VARS.json") as f:
        stats = json.load(f).get("obs_running_stats")
    if stats is None or stats["count"] < 2:
        return 0.0, 1.0
    mean = stats["mean"][0]
    var = stats["var"][0] / (stats["count"] - 1)
    std = float(np.sqrt(var)) if var > 0 else 1.0
    return mean, std


class MLBot(Bot):
    def initialize(self):
        project_dir = Path(__file__).parent
        checkpoint_dir = find_latest_checkpoint(project_dir)
        print(f"Lade Checkpoint: {checkpoint_dir}")

        self.device = torch.device("cpu")
        state_dict = torch.load(checkpoint_dir / "PPO_POLICY.pt", map_location=self.device)

        obs_size = state_dict["model.0.weight"].shape[1]
        n_actions = state_dict[list(state_dict.keys())[-2]].shape[0]

        self.policy = DiscreteFF(obs_size, n_actions, LAYER_SIZES, self.device)
        self.policy.load_state_dict(state_dict)
        torch.set_num_threads(1)

        self.obs_mean, self.obs_std = load_obs_standardization(checkpoint_dir)
        print(f"Obs-Normierung: mean={self.obs_mean:.3e}, std={self.obs_std:.4f}")

        self.action_parser = LookupTableAction()
        self.obs_builder = DefaultObs(
            zero_padding=None,
            pos_coef=np.asarray([1 / 4096, 1 / 6000, 1 / 2044]),
            ang_coef=1 / np.pi,
            lin_vel_coef=1 / 2300,
            ang_vel_coef=1 / 5.5,
            pad_timer_coef=1 / 10,
            boost_coef=1 / 100,
        )
        self.game_state = GameState.create_compat_game_state(self.field_info)
        self.prev_control = ControllerState()
        self.last_action_frame = None
        self.last_debug_frame = -DEBUG_INTERVAL_TICKS

    def get_output(self, packet: GamePacket) -> ControllerState:
        if len(packet.balls) == 0 or packet.match_info.match_phase == MatchPhase.Ended:
            return ControllerState()

        # Jedes Packet einlesen (rlgym_compat integriert u.a. Boost- und Handbremsen-Zeiten).
        self.game_state.update(packet)

        # Wie im Training (RepeatAction, 8 Ticks): nach Spiel-Ticks entscheiden, nicht nach
        # Aufrufen - RLBot verarbeitet nur das neueste Packet, dazwischen koennen Ticks fehlen.
        frame = packet.match_info.frame_num
        if self.last_action_frame is not None and frame - self.last_action_frame < TICK_SKIP:
            return self.prev_control
        self.last_action_frame = frame

        # rlgym_compat speichert die Autos unter player_id, nicht unter dem Index.
        agent_id = packet.players[self.index].player_id
        raw_obs = self.obs_builder.build_obs([agent_id], self.game_state, {})[agent_id]
        obs = np.clip((raw_obs - self.obs_mean) / self.obs_std, -5, 5)

        if DEBUG and frame - self.last_debug_frame >= DEBUG_INTERVAL_TICKS:
            self.last_debug_frame = frame
            self._print_debug(packet, agent_id, obs)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_idx, _ = self.policy.get_action(obs_tensor, deterministic=DETERMINISTIC)
        action_idx = int(action_idx)

        parsed = self.action_parser.parse_actions(
            {agent_id: np.array([action_idx])}, self.game_state, {}
        )[agent_id]
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

    def _print_debug(self, packet: GamePacket, agent_id, obs: np.ndarray):
        own = self.game_state.cars[agent_id]
        loc = packet.players[self.index].physics.location
        print(
            f"[debug] index={self.index} agent_id={agent_id} cars={list(self.game_state.cars.keys())} "
            f"team={own.team_num} compat_pos={np.round(own.physics.position).tolist()} "
            f"packet_pos=[{loc.x:.0f}, {loc.y:.0f}, {loc.z:.0f}] boost={own.boost_amount:.0f} "
            f"obs_size={obs.shape[0]} |obs|max={np.abs(obs).max():.2f} clipped={np.mean(np.abs(obs) >= 5):.0%}"
        )


if __name__ == "__main__":
    MLBot("rocketleagueaiproject/rlgym_jannis").run()