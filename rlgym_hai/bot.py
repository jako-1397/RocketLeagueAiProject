from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.GameTickPacket import GameTickPacket

# Hier würdest du später dein Modell importieren (z.B. PPO von Stable-Baselines3)
# from stable_baselines3 import PPO

# Du brauchst später auch eine Klasse, die die Spieldaten für das Modell übersetzt
# from rlgym_compat import GameState

class MyMachineLearningBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        # 1. Lade hier dein trainiertes Modell aus dem models/ Ordner
        # self.model = PPO.load("training/models/mein_fertiges_modell.zip")
        
        # 2. Definiere, wie das Modell die Welt sieht (Observation)
        # self.obs_builder = CustomObservationBuilder()

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        # Dies wird jeden Frame ausgeführt!
        
        # 1. Wandle das GameTickPacket in ein Format um, das RLGym/Dein Modell versteht
        # game_state = GameState(packet)
        # observation = self.obs_builder.build_obs(self.player_index, game_state, previous_action)
        
        # 2. Frag das Modell nach der besten Aktion
        # action, _states = self.model.predict(observation)
        
        # 3. Wandle die Aktion des Modells in RLBot-Befehle um
        controls = SimpleControllerState()
        
        # Beispiel (dies hängt von deinem Action-Parser in RLGym ab):
        # controls.throttle = action[0]
        # controls.steer = action[1]
        # controls.jump = action[5] > 0.5 
        
        return controls