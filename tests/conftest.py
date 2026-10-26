"""Configuracion comun de pytest: agrega lab3_multicamara/ al path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
