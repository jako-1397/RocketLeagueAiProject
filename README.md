# RocketLeagueAiProject

Uni-Projekt: Vergleich eines regelbasierten RLBot-Bots mit einem via RLGym trainierten Reinforcement-Learning-Agenten.

Wiki: [RLBot/python-interface/wiki](https://github.com/RLBot/python-interface/wiki)

## Projektstruktur

- `example_bot/` – statischer, regelbasierter Baseline-Bot (RLBot-Beispielbot), dient als Vergleichsmaßstab
- `rlgym_general/` – gemeinsame, offizielle RLGym-Grundlage des Teams
- `rlgym_hai/`, `rlgym_jannis/`, `rlgym_gabriel/` – individuelle Experimentierstände, gleiche Struktur wie `rlgym_general/`:
    - `bot.py` / `bot.toml` / `loadout.toml` – der spielbare RLBot-Wrapper um das jeweils trainierte Modell
    - `training/` – Trainings-Setup (`train.py`, `rewards.py`, `state_setters.py`) sowie `training/models/` mit den trainierten Checkpoints
    - `history/` – archivierte Zwischenstände (wird automatisch von `history.py` angelegt)
- `run.py` – interaktives Startskript für Testmatches
- `history.py` – interaktives Skript zum Sichern/Wiederherstellen von Zwischenständen

## Quick Start

1. Install [Python 3.12 or later](https://www.python.org/)
2. Create a Python virtual environment
    - `python -m venv venv`
3. Activate the virtual environment
    - Windows: `.\venv\Scripts\activate`
    - Linux: `source venv/bin/activate`
4. Install the required packages
    - `pip install -r requirements.txt`
5. Modify `rlbot.toml` to your liking
    - Note: `dev.toml` also exists with a few changed settings that might be useful for development
6. Download "RLBotServer.exe" from `https://github.com/RLBot/core/releases/tag/v5.0.0-rc17` into this project
7. Start a match with:
    ```
    python run.py
    ```
    Du wirst gefragt, gegen welchen Bot du spielen möchtest (`example_bot` oder einer der `rlgym_*`-Bots). Du spielst dabei selbst als Mensch (Team Blau) gegen den gewählten Bot (Team Orange).

## Zwischenstände sichern/wiederherstellen

Für den Trainingsnachweis kann jeder den aktuellen Stand seines Projektordners als ZIP archivieren:

```
python history.py
```

- Zuerst das Projekt wählen (`rlgym_general`, `rlgym_hai`, `rlgym_jannis` oder `rlgym_gabriel`)
- Dann entweder **speichern** (landet als `history/<Zeitstempel>.zip` im jeweiligen Ordner) oder den **letzten Zwischenstand wiederherstellen** (überschreibt vorhandene gleichnamige Dateien, löscht aber nichts zusätzlich Vorhandenes)

## Trainierte Modelle teilen

`training/models/` wird bewusst **nicht** von Git ignoriert. Ein neu trainierter Checkpoint wird also ganz normal Teil deiner nächsten Änderungen: Sobald du wie gewohnt committest und pusht (z. B. über die Quellcodeverwaltung in VS Code), wird er automatisch mit hochgeladen. Alle anderen im Team sehen den aktuellen Modellstand dann nach ihrem nächsten Pull direkt im jeweiligen Ordner.

## Changing the bots

- `example_bot/bot.py` – statische, regelbasierte Logik
- `rlgym_<name>/bot.py` – lädt (sobald trainiert) das jeweilige RLGym-Modell
- Aussehen jeweils über die zugehörige `loadout.toml` im selben Ordner

## Configuring for the v5 botpack

1. `pip install pyinstaller`
2. `pyinstaller --onefile example_bot/bot.py --paths example_bot` -
   This will create a file called `bot.spec`.
3. Create `bob.toml` in the same directory as the spec file with the following content:

    ```toml
    [[config]]
    project_name = "PythonExample"
    bot_configs = ["example_bot/bot.toml"]

    [config.builder_config]
    builder_type = "pyinstaller"
    entry_file = "bot.spec"
    ```

    - `project_name` will be the name of your bot's folder in the botpack
    - `bot_configs` is a list of bot configs that will be included in the botpack
    - `builder_type` should always be `pyinstaller`
    - `entry_file` is the name of the spec file

4. Commit both `bot.spec` and `bob.toml` to your bot's repository.
   Note that `bob.toml` CANNOT be renamed, but `bot.spec` can be anything as long as `entry_file` is also renamed to reflect the change.

## Link die wir für den Download benutzt hatten:

- https://github.com/RLBot/core/releases/tag/v5.0.0-rc17
- https://github.com/RLBot/python-example
