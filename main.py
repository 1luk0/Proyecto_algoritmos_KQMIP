"""
Punto de entrada principal. Lanza la GUI del analizador IIT (KQNodes / KGeoMIP).

Uso:
    python main.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gui"))

from app import App

if __name__ == "__main__":
    App().mainloop()
