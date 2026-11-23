"""Package LUMEN as a desktop app:  python build.py"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sep = ";" if sys.platform.startswith("win") else ":"

cmd = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
    "--name", "LUMEN", "--windowed",
    "--collect-all", "mediapipe",
    "--collect-submodules", "lumen_app",
]
model = os.path.join(HERE, "models", "hand_landmarker.task")
if os.path.isfile(model):
    cmd += ["--add-data", f"{model}{sep}models"]
else:
    print("Note: models/hand_landmarker.task not found; the app will download it on first run if needed.")
cmd.append(os.path.join(HERE, "main.py"))

print(" ".join(cmd))
sys.exit(subprocess.call(cmd, cwd=HERE))
