"""Regenerate the current v2 slides and figures from saved evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Paper.build_v2_artifacts import main
if __name__=="__main__":main()
