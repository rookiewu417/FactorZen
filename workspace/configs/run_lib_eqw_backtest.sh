#!/usr/bin/env bash
# 因子库等权组合分数面板全窗口回测：2 分数列 × 2 票池，串行跑（防 OOM）。
set -u
cd /home/ivenwu/projects/FactorZen

CFG=workspace/configs/lib_eqw_backtest.yaml
START=20160104
END=20260720
LOGDIR=workspace/_ops/logs/lib_eqw_backtest
mkdir -p "$LOGDIR"

for FACTOR in lib_eqw_all lib_eqw_posonly; do
  for UNIV in csi500 all_a; do
    TAG="${FACTOR}_${UNIV}"
    echo "=== START $TAG $(date -Is) ==="
    timeout 5400 pixi run -- python -m factorzen.cli.main factor backtest "$FACTOR" \
      --config "$CFG" --universe "$UNIV" --start "$START" --end "$END" \
      >"$LOGDIR/${TAG}.log" 2>&1
    echo "=== END $TAG rc=$? $(date -Is) ==="
  done
done

touch "$LOGDIR/all.done"
echo "ALL DONE $(date -Is)"
