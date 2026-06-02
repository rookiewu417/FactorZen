# 杩愯鎵嬪唽

甯哥敤鍛戒护锛?
```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20250101 --end 20260513 --universe csi500
pixi run fz report path <run_id>
```

YAML 閰嶇疆杩愯锛?
```bash
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
```

鎶ュ憡鐢熸垚锛?
```bash
pixi run fz report build momentum_20d --start 20250101 --end 20260513
pixi run fz report build momentum_20d --start 20250101 --end 20260513 --reuse
```

鏁版嵁鎷夊彇锛?
```bash
pixi run fz data fetch daily --start 20250101 --end 20260513
pixi run fz data fetch daily-basic --start 20250101 --end 20260513
```

鍏抽敭杈撳嚭鍦?`workspace/factor_evaluations/{run_id}/`銆傛柊澧炴祦绋嬬粺涓€浼樺厛浣跨敤 `fz`锛沗daily`銆乣report`銆乣factor test` 鍜?`report open` 浠呬綔涓哄吋瀹瑰埆鍚嶄繚鐣欍€?
