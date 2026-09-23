from pathlib import Path
from time import sleep

from rlbot import flat
from rlbot.managers import MatchManager

MATCH_CONFIG_FILE = "rlbot.toml"
GENERATED_CONFIG_FILE = "_generated_match.toml"
RLGYM_PREFIX = "rlgym_"


def find_bots(root_dir: Path) -> list[tuple[str, Path]]:
    bots = []
    example_toml = root_dir / "example_bot" / "bot.toml"
    if example_toml.exists() and example_toml.stat().st_size > 0:
        bots.append(("example_bot", example_toml))
    for d in sorted(root_dir.glob(f"{RLGYM_PREFIX}*")):
        toml_path = d / "bot" / "bot.toml"
        if toml_path.exists() and toml_path.stat().st_size > 0:
            bots.append((d.name, toml_path))
    return bots


def choose_bot(bots: list[tuple[str, Path]]) -> Path:
    print("\nGegen welchen Bot moechtest du spielen?")
    for i, (name, _) in enumerate(bots):
        print(f"  [{i}] {name}")
    while True:
        raw = input("Auswahl: ").strip()
        if raw.isdigit() and int(raw) < len(bots):
            return bots[int(raw)][1]
        print("Ungueltige Eingabe, bitte erneut.")


if __name__ == "__main__":
    root_dir = Path(__file__).parent

    bots = find_bots(root_dir)
    if not bots:
        raise SystemExit("Keine bot.toml-Dateien gefunden!")
    chosen_bot = choose_bot(bots)
    chosen_rel = chosen_bot.relative_to(root_dir).as_posix()

    # rlbot.toml unveraendert uebernehmen, nur den Pfad zum Gegner-Bot
    # gegen die getroffene Auswahl austauschen
    base_text = (root_dir / MATCH_CONFIG_FILE).read_text(encoding="utf-8")
    new_text = base_text.replace(
        'config_file = "example_bot/bot.toml"',
        f'config_file = "{chosen_rel}"',
    )
    generated_path = root_dir / GENERATED_CONFIG_FILE
    generated_path.write_text(new_text, encoding="utf-8")

    # Start RLBotServer and the match
    match_manager = MatchManager()
    match_manager.start_match(generated_path)

    sleep(5)

    # wait for the match to end
    # or press ctrl+c to kill the match
    while (
        match_manager.packet is None
        or match_manager.packet.match_info.match_phase != flat.MatchPhase.Ended
    ):
        sleep(0.1)

    # ensure RLBotServer shuts down
    match_manager.shut_down()
    generated_path.unlink(missing_ok=True)