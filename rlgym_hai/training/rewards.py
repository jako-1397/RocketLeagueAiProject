from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward


def build_reward_fn():
    """Wird von train.py aufgerufen. Muss die RewardFunction fuer dieses Projekt liefern."""
    return CombinedReward(
        (GoalReward(), 10.0),
        (TouchReward(), 0.1),
    )