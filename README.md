# RocketLeagueAiProject

Uni-Projekt: Vergleich eines regelbasierten RLBot-Bots mit einem via RLGym trainierten Reinforcement-Learning-Agenten.

Wiki: [RLBot/python-interface/wiki](https://github.com/RLBot/python-interface/wiki)

## Projektstruktur

- `example_bot/` – statischer, regelbasierter Baseline-Bot (RLBot-Beispielbot), dient als Vergleichsmaßstab
- `rlgym_general/` – gemeinsame, offizielle RLGym-Grundlage des Teams
- `rlgym_hai/`, `rlgym_jannis/`, `rlgym_gabriel/` – individuelle Experimentierstände, gleiche Struktur wie `rlgym_general/`:
    - `bot.py` / `bot.toml` / `loadout.toml` – der spielbare RLBot-Wrapper um das jeweils trainierte Modell
    - `act.py` / `obs.py` / `discrete_policy.py` – Hilfsmodule, die `bot.py` nutzt, um das trainierte Modell live im Spiel auszuführen
    - `training/rewards.py` – legt fest, wofür der Bot beim Training belohnt wird (individuell anpassbar)
    - `training/state_setters.py` – legt Start-/Kickoff-Positionen fest (individuell anpassbar)
    - `training/models/` – alle Trainings-Checkpoints, dauerhaft (siehe unten)
- `run.py` – interaktives Startskript für Testmatches gegen einen Bot
- `train.py` – interaktives Startskript zum Trainieren eines Bots

## Quick Start

### 1. Python 3.12 installieren

Wichtig: Es muss **genau Python 3.12** sein, nicht die neueste Version. Grund (kurz erklärt): Unsere KI-Trainingsbibliothek (RLGym) unterstützt aktuell offiziell nur bis Python 3.12 – das ist bei spezialisierter KI-Software normal, sie zieht mit neuen Python-Versionen oft erst später nach. Falls du bereits eine neuere Python-Version installiert hast, ist das kein Problem – wir installieren 3.12 einfach zusätzlich, ohne etwas zu verändern.

- Lade Python 3.12 von [python.org/downloads](https://www.python.org/downloads/) herunter (z. B. 3.12.10)
- Beim Installer reicht **"Install Now"**, keine weiteren Einstellungen nötig

### 2. Virtuelle Umgebung anlegen

Im Projektordner, in PowerShell:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\activate
```

Prüfen, ob es geklappt hat:

```powershell
python --version
```

Sollte `Python 3.12.x` anzeigen, und links im Terminal sollte `(venv)` stehen.

Falls du das Terminal später neu öffnest, muss die Umgebung jedes Mal erneut aktiviert werden:

```powershell
.\venv\Scripts\activate
```

### 3. Pakete installieren

```powershell
pip install -r requirements.txt
```

### 4. RLBotServer bereitstellen

Lade "RLBotServer.exe" von `https://github.com/RLBot/core/releases/tag/v5.0.0-rc17` herunter und lege sie ins Projektverzeichnis.

### 5. Match starten

```powershell
python run.py
```

Du wirst gefragt, gegen welchen Bot du spielen möchtest (`example_bot` oder einer der `rlgym_*`-Bots). Du spielst dabei selbst als Mensch (Team Blau) gegen den gewählten Bot (Team Orange), der live mit seinem aktuellsten trainierten Checkpoint spielt.

## Einen Bot trainieren

```powershell
python train.py
```

Du wirst gefragt, für welches Projekt trainiert werden soll (`rlgym_general`, `rlgym_hai`, `rlgym_jannis` oder `rlgym_gabriel`). Das Training läuft dann im Hintergrund (kein Rocket-League-Fenster nötig) und speichert regelmäßig Checkpoints in `training/models/` des gewählten Projekts. Startest du `train.py` erneut für dasselbe Projekt, wird automatisch am letzten Checkpoint weitertrainiert statt von vorne zu beginnen. Je nach Rechner kann ein sinnvoller Trainingslauf mehrere Stunden dauern – das Fenster kann in der Zeit einfach offen bleiben.

Wer eigene Ideen für Belohnungen oder Startpositionen ausprobieren will, passt `training/rewards.py` bzw. `training/state_setters.py` im eigenen Projektordner an (siehe Kommentare in den Dateien).

## Trainierte Modelle und deren Historie

`training/models/` wird bewusst **nicht** von Git ignoriert, und Checkpoints werden **nie automatisch gelöscht** – jeder je gespeicherte Trainingsstand bleibt dauerhaft als eigener Unterordner erhalten (benannt nach der Anzahl trainierter Zeitschritte, z. B. `1000000`, `2000000`, ...). Das dient gleichzeitig als vollständiger Trainingsnachweis über den gesamten Projektzeitraum – ein separates Backup-Skript ist dafür nicht nötig.

Committen/pushen (z. B. über die Quellcodeverwaltung in VS Code) läuft dabei ganz normal wie bei jeder anderen Datei:

```powershell
git add rlgym_<name>/training/models
git commit -m "Neue Trainings-Checkpoints"
git push
```

Alle anderen im Team sehen den aktuellen wie auch den gesamten historischen Modellstand nach ihrem nächsten Pull direkt im jeweiligen Ordner.

Die Entwicklung der Belohnungslogik selbst (`rewards.py`, `state_setters.py`) lässt sich jederzeit über die normale Git-Commit-Historie dieser Dateien nachvollziehen (`git log -p rlgym_<name>/training/rewards.py`).

## Changing the bots

- `example_bot/bot.py` – statische, regelbasierte Logik
- `rlgym_<name>/bot.py` – lädt automatisch den neuesten Checkpoint aus `training/models/` und lässt das trainierte Modell live spielen
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
