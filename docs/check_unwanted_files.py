from pathlib import Path

CURRENT_DIR = Path(__file__).parents[1]
RESULTS_DIR = CURRENT_DIR / "results"

# make sure that there are no files `results/**/reference_outputs.json`
# nor `results/**/leaderboard.csv`
matched_files = []
for filename in ["reference_outputs.json", "leaderboard.csv"]:
    matched_files += list(RESULTS_DIR.glob(f"**/{filename}"))

assert len(matched_files) == 0, f"The following files should not be present: {matched_files}"
