import zipfile
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent
RLGYM_PREFIX = "rlgym_"
HISTORY_DIRNAME = "history"


def find_projects() -> list[Path]:
    return sorted(d for d in ROOT_DIR.glob(f"{RLGYM_PREFIX}*") if d.is_dir())


def choose_project(projects: list[Path]) -> Path:
    print("\nIn welchem Projekt arbeitest du gerade?")
    for i, p in enumerate(projects):
        print(f"  [{i}] {p.name}")
    while True:
        raw = input("Auswahl: ").strip()
        if raw.isdigit() and int(raw) < len(projects):
            return projects[int(raw)]
        print("Ungueltige Eingabe, bitte erneut.")


def choose_action() -> str:
    print("\nWas moechtest du tun?")
    print("  [1] Aktuellen Zwischenstand speichern")
    print("  [2] Letzten Zwischenstand wiederherstellen (ueberschreibt aktuelle Dateien!)")
    while True:
        raw = input("Auswahl: ").strip()
        if raw in ("1", "2"):
            return raw
        print("Ungueltige Eingabe, bitte erneut (1 oder 2).")


def save_snapshot(project_dir: Path) -> None:
    history_dir = project_dir / HISTORY_DIRNAME
    history_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_path = history_dir / f"{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in project_dir.rglob("*"):
            if file_path.is_dir():
                continue
            if history_dir in file_path.parents:
                continue  # history/ selbst nie mit einzippen
            if "__pycache__" in file_path.parts:
                continue
            zf.write(file_path, file_path.relative_to(project_dir))

    print(f"\nZwischenstand gespeichert: {zip_path.relative_to(ROOT_DIR)}")


def restore_snapshot(project_dir: Path) -> None:
    history_dir = project_dir / HISTORY_DIRNAME
    zips = sorted(history_dir.glob("*.zip"))
    if not zips:
        print(f"\nKeine gespeicherten Zwischenstaende in {history_dir} gefunden.")
        return

    latest = zips[-1]
    print(f"\nLetzter Zwischenstand: {latest.name}")
    confirm = input(
        "Das ueberschreibt alle aktuellen Dateien in "
        f"'{project_dir.name}' (ausser history/). Fortfahren? [j/N]: "
    ).strip().lower()
    if confirm != "j":
        print("Abgebrochen.")
        return

    with zipfile.ZipFile(latest, "r") as zf:
        zf.extractall(project_dir)

    print(f"\nZwischenstand wiederhergestellt aus: {latest.name}")


if __name__ == "__main__":
    projects = find_projects()
    if not projects:
        raise SystemExit("Keine rlgym_*-Ordner gefunden!")

    project_dir = choose_project(projects)
    action = choose_action()

    if action == "1":
        save_snapshot(project_dir)
    else:
        restore_snapshot(project_dir)