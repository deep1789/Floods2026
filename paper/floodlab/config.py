"""Experiment configuration. Defaults reproduce the original 2023-2026 study; environment variables select the extended panel."""
import os, pandas as pd
def _ints(name, default): return tuple(int(x) for x in os.environ.get(name, default).split(','))
CSV_OVERRIDE = os.environ.get('FLOOD_CSV')                                  # path to an alternative panel with the same columns
T0 = pd.Timestamp(os.environ.get('FLOOD_T0', '2025-09-01'))                 # chronological test start
LOMO_YEARS = _ints('FLOOD_LOMO', '2023,2024,2025,2026')                     # held-out monsoons for leave-one-monsoon-out
FC_YEARS = _ints('FLOOD_FC', '2025,2026')                                   # forward-chaining test monsoons
INNER_YEARS = _ints('FLOOD_INNER', '2023,2024,2025')                        # inner folds for boosting tuning (all before T0)
TUNE_YEAR = int(os.environ.get('FLOOD_TUNE_YEAR', '2025'))                  # inner forward-chaining year for LSTM tuning (before T0)
ABL_YEARS = _ints('FLOOD_ABL', '2023,2024,2025,2026')                       # folds used for the feature-group ablation
N_CAND = int(os.environ.get('FLOOD_NCAND', '30'))                           # random boosting configurations
EXTRA_FC = _ints('FLOOD_EXTRA_FC', '2025,2026')                             # forward-chaining folds for TFT-lite / graph arms
