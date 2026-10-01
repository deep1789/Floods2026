#!/usr/bin/env bash
# Re-run the whole study on the corrected + extended panel (2010-01-01..2026-08-31). Stages are ordered by importance; run alone (CPU-bound).
set -e
cd "$(dirname "$0")"
export FLOOD_CSV="$PWD/panel_2010_2026.csv" FLOOD_T0=2023-01-01
export FLOOD_LOMO=$(python3 -c "print(','.join(str(y) for y in range(2010,2027)))")
export FLOOD_FC=2024,2025,2026 FLOOD_EXTRA_FC=2026 FLOOD_INNER=2020,2021,2022 FLOOD_TUNE_YEAR=2022 FLOOD_ABL=2018,2019,2020,2021,2022,2023,2024,2025,2026 FLOOD_NCAND=12
P=..
mkdir -p results tables2 figures
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a results/progress.log; }
log leakage;   python3 $P/tests/test_leakage.py            > results/leakage.log 2>&1
log benchmark; python3 $P/run_benchmark.py                  > results/bench_tree.log 2>&1
log evaluate-core; python3 $P/evaluate.py                   > results/evaluate.log 2>&1 || true
log analyses;  python3 $P/analyses.py                       > results/analyses.log 2>&1 || true
log ablation;  python3 $P/ablation.py                       > results/ablation.log 2>&1 || true
log tune;      python3 $P/run_tune.py                       > results/tune.log 2>&1
log lstm;      python3 $P/run_lstm.py 5 3 60                > results/bench_lstm.log 2>&1
log extra;     python3 $P/run_extra.py 3                    > results/bench_extra.log 2>&1
log extras2;   python3 $P/run_extras2.py                    > results/extras2.log 2>&1
log evaluate;  python3 $P/evaluate.py > results/evaluate.log 2>&1 && python3 $P/evaluate2.py > results/evaluate2.log 2>&1
log done
