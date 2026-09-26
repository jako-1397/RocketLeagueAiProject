from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

TEAM_SIZE = 1
SPAWN_OPPONENTS = True


def build_state_mutator():
    """Wird von train.py aufgerufen. Muss den StateMutator fuer dieses Projekt liefern."""
    return MutatorSequence(
        FixedTeamSizeMutator(
            blue_size=TEAM_SIZE,
            orange_size=TEAM_SIZE if SPAWN_OPPONENTS else 0,
        ),
        KickoffMutator(),
    )