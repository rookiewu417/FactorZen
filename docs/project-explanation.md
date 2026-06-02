# FactorZen 椤圭洰璇存槑

鏈€鍚庢洿鏂帮細2026-05-30

FactorZen 鏄潰鍚?A 鑲″洜瀛愮爺绌剁殑宸ョ▼鍖栨鏋躲€傚綋鍓嶉」鐩垎涓轰袱灞傦細

- `src/factorzen/`锛氭鏋跺唴鏍革紝鏀剧ǔ瀹氫唬鐮併€?- `workspace/`锛氭棩甯哥爺绌跺伐浣滃尯锛屾斁鐢ㄦ埛鍥犲瓙銆佸疄楠岄厤缃拰杩愯浜х墿銆?
## 1. 椤圭洰杈圭晫

椤圭洰閲嶇偣鏄妸鍗曞洜瀛愮爺绌跺仛鎴愬彲澶嶇幇閾捐矾锛?
```text
鏁版嵁缂撳瓨 -> 鍥犲瓙璁＄畻 -> 棰勫鐞?-> IC/鍥炴祴 -> walk-forward -> manifest -> HTML 鎶ュ憡
```

褰撳墠涓嶈鐩栵細

- 瀹炵洏 OMS / EMS銆?- Tick 鏁版嵁鎺ュ叆銆佽鍗曠翱閲嶅缓鍜岀洏鍙ｆ垚浜ゆā鎷熴€?- 鐢熶骇绾х粍鍚堟墽琛岄棴鐜€?- CI 涓緷璧栫湡瀹?Tushare 缃戠粶鐘舵€佺殑 smoke test銆?
`src/factorzen/research/combination/` 鏄疄楠屾€у鍥犲瓙鍚堟垚宸ュ叿銆俙ic_weighted` 鍜?`max_ir` 鐩墠浣跨敤鏍锋湰鍐?IC 浼版潈閲嶏紝涓嶈兘瑙ｉ噴涓烘棤鍋?OOS 缁勫悎琛ㄧ幇銆?
## 2. 褰撳墠鐩綍缁撴瀯

```text
FactorZen/
鈹溾攢鈹€ src/factorzen/        # 妗嗘灦浠ｇ爜
鈹?  鈹溾攢鈹€ automation/       # 璋冨害銆佷綔涓氱姸鎬併€佹棩缁堟祦姘寸嚎
鈹?  鈹溾攢鈹€ cli/              # fz 缁熶竴鍛戒护鍏ュ彛
鈹?  鈹溾攢鈹€ config/           # 璺緞銆佸父閲忋€乀ushare 閰嶇疆
鈹?  鈹溾攢鈹€ core/             # loader/storage/calendar/universe/registry 绛夊簳搴?鈹?  鈹溾攢鈹€ daily/            # 鏃?鍛?鏈堥鍥犲瓙銆侀澶勭悊銆佽瘎浼般€佷紭鍖?鈹?  鈹溾攢鈹€ intraday/         # 鍒嗛挓绾у洜瀛愩€侀澶勭悊銆佽瘎浼?鈹?  鈹溾攢鈹€ pipelines/        # daily/report 绛夊彲鎵ц娴佹按绾?鈹?  鈹溾攢鈹€ reports/          # HTML Tear Sheet
鈹?  鈹斺攢鈹€ research/         # 瀹為獙鎬х爺绌跺伐鍏?鈹溾攢鈹€ workspace/
鈹?  鈹溾攢鈹€ factors/          # 鏂板洜瀛愬叆鍙?鈹?  鈹溾攢鈹€ configs/          # 瀹為獙 YAML 閰嶇疆
鈹?  鈹溾攢鈹€ notebooks/        # 涓存椂鐮旂┒绗旇
鈹?  鈹斺攢鈹€ runs/             # 姣忔杩愯鐨勮嚜鍖呭惈杈撳嚭
鈹溾攢鈹€ data/                 # 鏈湴琛屾儏鍜岃储鍔℃暟鎹紦瀛?鈹溾攢鈹€ tests/                # pytest 娴嬭瘯
鈹?  鈹斺攢鈹€ benchmarks/       # 鎬ц兘鍩哄噯鑴氭湰
鈹斺攢鈹€ docs/                 # 褰撳墠鏂囨。鍜屽彂甯冭鏄?```

鏃ュ父鐮旂┒涓嶈鍦ㄦ繁灞?`src/factorzen/daily/...` 閲屾柊寤轰釜浜哄洜瀛愶紱鏂板洜瀛愭斁鍦?`workspace/factors/{daily,weekly,monthly,intraday}/`銆?
## 3. 甯哥敤鍏ュ彛

```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20250101 --end 20260513 --universe csi500
pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml
pixi run fz report build momentum_20d --start 20250101 --end 20260513
pixi run fz report build momentum_20d --start 20250101 --end 20260513 --reuse
pixi run fz report path <run_id>
pixi run fz data fetch daily --start 20250101 --end 20260513
```

## 4. 杈撳嚭绾﹀畾

杩愯浜х墿缁熶竴姹囨€诲埌锛?
```text
workspace/factor_evaluations/{run_id}/
鈹溾攢鈹€ manifest.json
鈹溾攢鈹€ report.html
鈹溾攢鈹€ factor.parquet
鈹溾攢鈹€ ic.parquet
鈹溾攢鈹€ quality.json
鈹溾攢鈹€ walk_forward.json
鈹斺攢鈹€ universe.parquet
```

`manifest.json` 璁板綍閰嶇疆銆佸懡浠ゃ€乬it SHA銆乨irty 鐘舵€併€乣pixi.lock` hash銆佽繍琛岀姸鎬併€侀敊璇俊鎭拰杈撳嚭璺緞銆?
## 5. 鏁版嵁鐩綍

```text
data/
鈹溾攢鈹€ raw/
鈹?  鈹溾攢鈹€ daily/year=YYYY/month=MM/data.parquet
鈹?  鈹溾攢鈹€ daily_basic/year=YYYY/month=MM/data.parquet
鈹?  鈹溾攢鈹€ finance/year=YYYY/quarter=Q/data.parquet
鈹?  鈹斺攢鈹€ minute/year=YYYY/month=MM/data.parquet
鈹斺攢鈹€ cache/
```

鏍稿績璇诲彇鍜屽啓鍏ラ€昏緫鍦?`src/factorzen/core/storage.py` 涓?`src/factorzen/core/loader.py`銆備笟鍔′唬鐮佷笉瑕佹墜鍐欓」鐩牴鐩綍璺緞銆?
## 6. 閰嶇疆浣撶郴

杩愯閰嶇疆鏀惧湪 `workspace/configs/`锛?
```text
workspace/configs/daily/daily_factor_template.yaml
workspace/configs/daily/daily_factor_template.yaml
```

閰嶇疆鍔犺浇鍜屾牎楠屽湪 `src/factorzen/core/config_loader.py`銆俌AML 涓殑 `preprocessing`銆乣backtest`銆乣cost_model`銆乣walk_forward` 瀛楁浼氬奖鍝嶇湡瀹炶繍琛屻€?
妗嗘灦甯搁噺鍜岃矾寰勫湪锛?
- `src/factorzen/config/settings.py`
- `src/factorzen/config/constants.py`
- `src/factorzen/config/tushare_config.py`

## 7. 鍐欐柊鍥犲瓙

```bash
pixi run fz factor new my_alpha --frequency daily
```

鐢熸垚鏂囦欢锛?
```text
workspace/factors/daily/my_alpha.py
```

瀹炵幇 `DailyFactor.compute(context)` 鍚庤繍琛岋細

```bash
pixi run fz factor run my_alpha --start 20250101 --end 20260513
```

鍥犲瓙瀹炵幇缁熶竴鏀惧湪 `workspace/factors/`銆俙src/factorzen/daily/factors/` 鍜?`src/factorzen/intraday/factors/` 鍙繚鐣欐鏋跺熀绫诲拰娉ㄥ唽涓績銆?
## 8. 璐ㄩ噺闂?
```bash
pixi run test
pixi run lint
pixi run typecheck
pixi run coverage
```

褰撳墠绫诲瀷妫€鏌ョ洰鏍囨槸 `src/factorzen`銆傛柊澧炴鏋朵唬鐮佸簲淇濇寔 ruff銆乵ypy銆乸ytest 閫氳繃锛涙柊澧炰釜浜哄洜瀛愯嚦灏戝簲閫氳繃涓€娆″崟鍥犲瓙杩愯楠岃瘉銆?
## 9. 鏂囨。绱㈠紩

- `docs/architecture.md`锛氱粨鏋勬€昏銆?- `docs/factor-authoring.md`锛氭柊鍥犲瓙缂栧啓娴佺▼銆?- `docs/runbook.md`锛氳繍琛屽拰鏌ユ姤鍛婃墜鍐屻€?- `docs/evolution-plan-2026.md`锛氬悗缁紨杩涜鍒掋€?- `docs/release-notes/`锛氬彂甯冭鏄庛€?
