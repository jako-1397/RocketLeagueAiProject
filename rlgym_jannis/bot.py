from typing import override

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot


class MLBot(Bot):
    @override
    def initialize(self):
        # TODO: hier später das trainierte RLGym-Modell laden
        pass

    @override
    def get_output(self, packet: GamePacket) -> ControllerState:
        # TODO: GamePacket -> RLGym-Observation -> Modell-Vorhersage -> ControllerState
        return ControllerState()


if __name__ == "__main__":
    MLBot("rocketleagueaiproject/rlgym_jannis").run()