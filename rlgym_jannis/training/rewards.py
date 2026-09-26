"""Reward-Strategie fuer rlgym_jannis - Curriculum ueber Phasen.

PHASE von Hand umstellen, wenn der Bot die Ziele der aktuellen Phase erreicht.
train.py setzt am letzten Checkpoint fort, d.h. das Netz bleibt erhalten, nur die
"Bewertung" aendert sich. Beim Phasenwechsel sinkt die "Policy Reward" im Iteration
Report meist sprunghaft - das ist normal (andere Skala), kein Rueckschritt.

Skalen (siehe reward_lib.py): dichte Rewards sind "pro Sekunde" normiert
(Gewicht 1.0 = max. 1.0 pro Sekunde), Ereignis-Rewards geben einen Wert pro Ereignis.
Da rlgym-ppo die Returns standardisiert, zaehlen die VERHAELTNISSE der Gewichte.
"""
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym_tools.rocket_league.reward_functions.goal_prob_reward import GoalViewReward
from rlgym_tools.rocket_league.reward_functions.velocity_player_to_ball_reward import VelocityPlayerToBallReward

from RocketLeagueAiProject.reward_lib import CooldownTouchReward, FaceBallReward, VelocityBallToGoalReward

PHASE = 1


def _phase_1():
    """Grundlagen: zum Ball fahren und ihn treffen.

    Ziel/Wechselkriterium: Bot faehrt zielstrebig zum Ball und trifft ihn regelmaessig,
    auch ausserhalb des Anstosses (im Spiel pruefen). Erfahrungsgemaess einige 10 Mio. Steps.
    """
    return CombinedReward(
        # Dichter Wegweiser zum Ball. Nur positive Werte: Wegfahren wird hier (noch) nicht
        # bestraft, sonst lernt ein Anfaenger-Bot schnell "stehen bleiben = sicher".
        (VelocityPlayerToBallReward(include_negative_values=False), 1.0),
        # Kleine Hilfe beim Ausrichten. Bewusst schwach, damit es nicht wichtiger ist als Fahren.
        (FaceBallReward(), 0.25),
        # Der eigentliche Lernanreiz dieser Phase. Mit Abklingzeit gegen Ball-Kuscheln;
        # der Beschleunigungsanteil macht einen echten Schlag wertvoller als Anstupsen.
        (CooldownTouchReward(touch_reward=1.0, acceleration_reward=2.0, cooldown_seconds=0.5), 1.0),
        # Schwacher Richtungshinweis schon ab Start: sonst lernt der Bot, den Ball egal wohin
        # zu hauen, und muss das spaeter muehsam wieder verlernen. Negativ Richtung eigenes Tor.
        (VelocityBallToGoalReward(), 0.5),
        # Das eigentliche Spielziel ist von Anfang an da (+1 Tor / -1 Gegentor), auch wenn
        # Tore am Anfang selten und eher zufaellig sind.
        (GoalReward(), 10.0),
    )


def _phase_2():
    """ENTWURF - Richtung Tor spielen. Erst anpassen, wenn Phase 1 sitzt."""
    return CombinedReward(
        (VelocityPlayerToBallReward(include_negative_values=True), 0.5),
        (CooldownTouchReward(touch_reward=0.3, acceleration_reward=2.0, cooldown_seconds=0.5), 1.0),
        (VelocityBallToGoalReward(), 2.0),
        # Potentialbasiert (Ng et al. 1999): belohnt jede Verbesserung der Torchance,
        # bestraft jede Verschlechterung. Summe ueber eine Episode ist beschraenkt.
        (GoalViewReward(), 3.0),
        (GoalReward(), 20.0),
    )


def _phase_3():
    """ENTWURF - Tore entscheiden, Hilfs-Rewards fast weg."""
    return CombinedReward(
        (VelocityPlayerToBallReward(include_negative_values=True), 0.1),
        (CooldownTouchReward(touch_reward=0.05, acceleration_reward=1.0, cooldown_seconds=0.5), 1.0),
        (VelocityBallToGoalReward(), 1.0),
        (GoalViewReward(), 5.0),
        (GoalReward(), 30.0),
    )


_PHASES = {1: _phase_1, 2: _phase_2, 3: _phase_3}


def build_reward_fn():
    """Wird von train.py aufgerufen. Muss die RewardFunction fuer dieses Projekt liefern."""
    return _PHASES[PHASE]()