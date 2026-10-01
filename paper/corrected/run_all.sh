#!/usr/bin/env bash
# Full study re-run on the corrected-discharge panel (same 2023-2026 weather, same splits as the original study). Run alone (CPU-bound).
set -e
cd "$(dirname "$0")"
export FLOOD_CSV="$PWD/panel_corrected_2023_2026.csv"
P=..
mkdir -p results tables2 figures
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a results/progress.log; }
log leakage;       python3 $P/tests/test_leakage.py   > results/leakage.log 2>&1
log benchmark;     python3 $P/run_benchmark.py        > results/bench_tree.log 2>&1
log evaluate-core; python3 $P/evaluate.py             > results/evaluate.log 2>&1 || true
log analyses;      python3 $P/analyses.py             > results/analyses.log 2>&1 || true
log ablation;      python3 $P/ablation.py             > results/ablation.log 2>&1 || true
log tune;          python3 $P/run_tune.py             > results/tune.log 2>&1
log lstm;          python3 $P/run_lstm.py 10 5 60     > results/bench_lstm.log 2>&1
log extra;         python3 $P/run_extra.py 5          > results/bench_extra.log 2>&1
log extras2;       python3 $P/run_extras2.py          > results/extras2.log 2>&1
log evaluate;      python3 $P/evaluate.py > results/evaluate.log 2>&1 && python3 $P/evaluate2.py > results/evaluate2.log 2>&1
log done
