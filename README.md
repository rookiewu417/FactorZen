# FactorZen

> **FactorZen** 鏄竴涓潰鍚?A 鑲″崟鍥犲瓙鐨勫彲淇＄爺绌舵鏋讹紝寮鸿皟涓ヨ皑銆佸厠鍒跺拰鍙鐜般€傚綋鍓嶆牳蹇冧富绾胯鐩栧洜瀛愯绠椼€侀澶勭悊銆両C/鍥炴祴璇勪及銆亀alk-forward OOS銆佹暟鎹川閲忔姤鍛娿€佸疄楠?manifest 涓?Tear Sheet 鎶ュ憡鐢熸垚銆俙research/combination/` 鎻愪緵瀹為獙鎬у鍥犲瓙鍚堟垚宸ュ叿锛岀敤浜庣爺绌跺姣旓紝涓嶄綔涓哄綋鍓嶇敓浜х粍鍚堜紭鍖栨ā鍧椼€?
## 褰撳墠椤圭洰缁撴瀯

FactorZen 鐜板湪鍒嗕负妗嗘灦鍐呮牳鍜岀爺绌跺伐浣滃尯锛?
```text
src/factorzen/      # 妗嗘灦浠ｇ爜锛氭暟鎹€佸洜瀛愭敞鍐屻€侀澶勭悊銆佽瘎浼般€佹姤鍛娿€丆LI
workspace/factors/  # 鏃ュ父鏂板洜瀛愬叆鍙?workspace/configs/  # 瀹為獙閰嶇疆
workspace/factor_evaluations/     # 姣忔杩愯鐨?report.html銆乵anifest.json銆乸arquet/json 浜х墿
data/               # 鏈湴鏁版嵁缂撳瓨
tests/              # pytest 娴嬭瘯
docs/               # 鏋舵瀯銆佸洜瀛愮紪鍐欍€佽繍琛屾墜鍐?```

鏃ュ父浼樺厛浣跨敤缁熶竴 CLI锛?
```bash
pixi run fz factor list
pixi run fz factor new my_alpha --frequency daily
pixi run fz factor run my_alpha --start 20250101 --end 20260513 --universe csi500
pixi run fz report path <run_id>
```

鏇村璇存槑瑙?`docs/project-explanation.md`銆乣docs/architecture.md`銆乣docs/factor-authoring.md` 鍜?`docs/runbook.md`銆?
## 鐩綍缁撴瀯涓庨鐜囪瘝姹囪〃

| 鐩綍 | 棰戠巼/鑱岃矗 | 鏁版嵁鏉ユ簮 | 鎴愮啛搴?|
|------|------|---------|--------|
| `src/factorzen/daily/` | 鏃?鍛?鏈堬紙鏃ョ嚎涓嬮噰鏍凤級 | Tushare 鏃ョ嚎琛屾儏 + 浼板€?+ 璐㈡姤 | 鉁?瀹屾暣 |
| `src/factorzen/intraday/` | 鍒嗛挓锛?min/5min锛?| Tushare 鍒嗛挓绾?| 鉁?璇勪及绠＄嚎瀹屾暣锛堝緟瀹炴暟鎹級|
| `src/factorzen/core/` | 閫氱敤搴曞骇 | 鈥?| 鉁?瀹屾暣 |
| `src/factorzen/research/` | 瀹為獙鎬х爺绌跺伐鍏?| 澶嶇敤 daily/intraday 鏁版嵁 | 鈿狅笍 闈炵敓浜?|
| `src/factorzen/reports/` | HTML Tear Sheet | 鈥?| 鉁?瀹屾暣 |
| `workspace/factors/` | 鐢ㄦ埛鑷畾涔夊洜瀛?| 澶嶇敤妗嗘灦鏁版嵁涓婁笅鏂?| 鉁?鏃ュ父鍏ュ彛 |
| `workspace/factor_evaluations/` | 瀹為獙杈撳嚭 | 鈥?| 鉁?鏂拌緭鍑哄叆鍙?|

> **鍛藉悕璇存槑**锛歚daily` 鈮?浣庨锛堜笟鐣屽父璇寸殑鏃ラ/鏈堥鍥犲瓙锛夛紱`intraday` 鈮?鏃ュ唴锛堝垎閽熺骇锛夛紱`research` 鈮?瀹為獙鎬х爺绌跺伐鍏枫€?
## 蹇€熷紑濮?

### 1. 鐜瀹夎

```bash
# 瀹夎 pixi锛圵indows锛?
winget install prefix-dev.pixi

# 鍏嬮殕椤圭洰鍚庡畨瑁呬緷璧栵紙鍚?editable install锛?
cd FactorZen
pixi install
```

### 2. 閰嶇疆 Tushare Token

```bash
cp .env.example .env
# 缂栬緫 .env锛屽～鍏?TUSHARE_TOKEN=your_token
```

鍦?[tushare.pro/user/token](https://tushare.pro/user/token) 鑾峰彇 token銆?

### 3. 鎷夊彇琛屾儏鏁版嵁

```bash
# 鎷夊彇鏃ョ嚎琛屾儏锛堢害 5 鍒嗛挓锛?pixi run fz data fetch daily --start 20250101 --end 20260513

# 鎷夊彇姣忔棩浼板€硷紙PE/PB/甯傚€硷紝鏈堥鍥犲瓙渚濊禆锛?pixi run fz data fetch daily-basic --start 20250101 --end 20260513
```

### 4. 杩愯鍗曞洜瀛愯瘎浼?

```bash
# 鍗曞洜瀛愬畬鏁磋瘎浼?鈫?workspace/factor_evaluations/{run_id}/
pixi run fz factor run momentum_20d --start 20250101 --end 20260513

# 浣跨敤 YAML 閰嶇疆杩愯锛沺reprocessing/backtest/cost_model/walk_forward 瀛楁浼氱湡瀹炵敓鏁?pixi run fz factor run --config workspace/configs/daily/daily_factor_template.yaml

# 鐢熸垚 HTML Tear Sheet 鈫?workspace/factor_evaluations/{run_id}/report.html
pixi run fz report build momentum_20d --start 20250101 --end 20260513

# 澶嶇敤宸叉湁 parquet 绉掑嚭鎶ュ憡锛堥渶鍏堣窇杩囦笂闈换鎰忎竴鏉★級
pixi run fz report build momentum_20d --start 20250101 --end 20260513 --reuse

# 澶氬洜瀛愬悎鎴愮爺绌朵娇鐢?factorzen.research.combination 鍖呭唴 API
```

`pixi run daily`銆乣pixi run report`銆乣pixi run fz factor test` 鍜?`pixi run fz report open` 淇濈暀涓哄吋瀹瑰埆鍚嶏紱鏂版枃妗ｅ拰鏂版祦绋嬩紭鍏堜娇鐢?`fz factor run`銆乣fz report build`銆乣fz report path`銆?
涓昏杈撳嚭锛?
- `workspace/factor_evaluations/{run_id}/report.html`锛氭湰娆″疄楠屾姤鍛娿€?- `workspace/factor_evaluations/{run_id}/manifest.json`锛氬疄楠?manifest锛岃褰曞畬鏁撮厤缃€佸懡浠ゃ€乬it SHA銆乨irty 鐘舵€併€乣pixi.lock` hash銆佹垚鍔?澶辫触鐘舵€併€侀敊璇俊鎭拰宸茬敓鎴愯緭鍑鸿矾寰勩€?- `workspace/factor_evaluations/{run_id}/factor.parquet`锛氶澶勭悊鍚庣殑鍥犲瓙鐭╅樀鍓湰銆?- `workspace/factor_evaluations/{run_id}/ic.parquet`锛欼C 搴忓垪鍓湰銆?- `workspace/factor_evaluations/{run_id}/quality.json`锛氭暟鎹川閲忔姤鍛婂壇鏈€?- `workspace/factor_evaluations/{run_id}/walk_forward.json`锛歸alk-forward/OOS 鎽樿鍓湰銆?
## 鍙敤鍥犲瓙鍒楄〃

### daily 鈥?鏃ラ锛?0 涓級

| 鍥犲瓙鍚?| 绫诲埆 | 鎻忚堪 |
|--------|------|------|
| `momentum_20d` | 鍔ㄩ噺 | 20 鏃ヤ环鏍煎姩閲忥紱淇濈暀鍏煎锛岀爺绌朵笂寤鸿浼樺厛浣跨敤 `momentum_12_1` |
| `momentum_12_1` | 鍔ㄩ噺 | Jegadeesh-Titman 12-1 鍔ㄩ噺锛屽墧闄ゆ渶杩?1 涓湀鍙嶈浆鏁堝簲 |
| `reversal_5d` | 鍙嶈浆 | 5 鏃ョ煭鏈熷弽杞?|
| `volatility_20d` | 娉㈠姩 | 20 鏃ュ凡瀹炵幇娉㈠姩鐜?|
| `turnover_5d` | 鎹㈡墜 | 5 鏃ュ钩鍧囨崲鎵嬬巼 |
| `amihud_illiquidity` | 娴佸姩鎬?| Amihud (2002) 闈炴祦鍔ㄦ€ф寚鏍?|
| `beta_60d` | 椋庨櫓 | 60 鏃?CAPM Beta |
| `idiosyncratic_vol_20d` | 娉㈠姩 | 20 鏃ョ壒璐ㄦ尝鍔ㄧ巼锛堝幓闄ゅ競鍦?Beta 鍚庢畫宸?std锛?|
| `max_return_5d` | 褰╃エ鏁堝簲 | 5 鏃ユ渶澶у崟鏃ユ定骞咃紙Bali et al. 2011 MAX 鍥犲瓙锛?|
| `skewness_20d` | 鍋忓害 | 20 鏃ユ敹鐩婂亸搴︼紙姝ｅ亸鑲＄エ鏈潵鏀剁泭鍋忎綆锛?|

### daily 鈥?鍛ㄩ锛? 涓級

| 鍥犲瓙鍚?| 绫诲埆 | 鎻忚堪 |
|--------|------|------|
| `momentum_weekly` | 鍔ㄩ噺 | 鍛ㄩ蹇収鍔ㄩ噺 |
| `turnover_weekly` | 鎹㈡墜 | 鍛ㄩ蹇収鎹㈡墜鐜?|
| `volatility_weekly` | 娉㈠姩 | 鍛ㄩ蹇収娉㈠姩鐜?|

### daily 鈥?鏈堥锛? 涓級

| 鍥犲瓙鍚?| 绫诲埆 | 鎻忚堪 |
|--------|------|------|
| `pe_ttm` | 浼板€?| 鏈堥婊氬姩甯傜泩鐜囷紙渚濊禆 daily_basic锛?|
| `pb` | 浼板€?| 鏈堥甯傚噣鐜囷紙渚濊禆 daily_basic锛?|
| `ep_ratio` | 浼板€?| 鏈堥 E/P锛? 1/PE_TTM锛?|
| `bm_ratio` | 浼板€?| 鏈堥 B/M锛? 1/PB锛?|
| `roe_ttm` | 璐ㄩ噺 | 鏈堥 ROE TTM锛孭IT 瀵归綈锛堜緷璧?finance锛?|
| `asset_growth` | 璐ㄩ噺 | 骞村害鎬昏祫浜у閫燂紙渚濊禆 finance锛?|

### intraday 鈥?鍒嗛挓棰戯紙2 涓級

| 鍥犲瓙鍚?| 绫诲埆 | 鎻忚堪 |
|--------|------|------|
| `momentum_1min` | 鍔ㄩ噺 | 1 鍒嗛挓 5-bar 鏀剁泭鍔ㄩ噺 |
| `vwap_deviation` | 浠锋牸鍋忕 | 褰撳墠浠风浉瀵规棩鍐?VWAP 鍋忕搴?|

## 寮€鍙戝懡浠?

```bash
pixi run test      # 杩愯娴嬭瘯
pixi run lint      # ruff check
pixi run typecheck # mypy锛歴rc/factorzen
pixi run coverage  # pytest coverage锛屽綋鍓嶉棬妲?70%
pixi run format    # ruff format
pixi run lab       # 鍚姩 JupyterLab
```

## 椤圭洰鏋舵瀯

```
FactorZen/
鈹溾攢鈹€ src/factorzen/   # 妗嗘灦鍐呮牳
鈹?  鈹溾攢鈹€ core/        # 鏁版嵁搴曞骇锛坙oader/storage/calendar/universe锛?鈹?  鈹溾攢鈹€ config/      # 璺緞甯搁噺銆乀ushare 閰嶇疆
鈹?  鈹溾攢鈹€ daily/       # 鏃?鍛?鏈堥鍥犲瓙妗嗘灦
鈹?  鈹溾攢鈹€ intraday/    # 鍒嗛挓棰戝洜瀛愭鏋?鈹?  鈹溾攢鈹€ reports/     # HTML Tear Sheet 鐢熸垚
鈹?  鈹溾攢鈹€ research/    # 瀹為獙鎬х爺绌跺伐鍏凤紙闈炵敓浜т紭鍖栵級
鈹?  鈹溾攢鈹€ pipelines/   # daily/report 绛夎繍琛岀绾?鈹?  鈹斺攢鈹€ cli/         # fz 缁熶竴 CLI
鈹溾攢鈹€ workspace/       # 鐢ㄦ埛鐮旂┒宸ヤ綔鍖猴紙鍥犲瓙銆侀厤缃€乺un 杈撳嚭锛?鈹斺攢鈹€ tests/           # pytest 娴嬭瘯濂椾欢锛堝惈 benchmarks/ 鎬ц兘鍩哄噯鑴氭湰锛?```

## 鏁版嵁鐩綍绾﹀畾

```
data/
鈹溾攢鈹€ raw/
鈹?  鈹溾攢鈹€ daily/year=YYYY/month=MM/data.parquet       # 鏃ョ嚎琛屾儏
鈹?  鈹溾攢鈹€ daily_basic/year=YYYY/month=MM/data.parquet # 姣忔棩浼板€?
鈹?  鈹溾攢鈹€ finance/year=YYYY/quarter=Q/data.parquet    # 璐㈠姟鏁版嵁
鈹?  鈹斺攢鈹€ minute/year=YYYY/month=MM/data.parquet      # 鍒嗛挓绾?
鈹斺攢鈹€ cache/           # 鑲＄エ姹犮€佷氦鏄撴棩鍘嗙瓑灏忓瀷缂撳瓨

workspace/
鈹溾攢鈹€ factors/         # 鐢ㄦ埛鑷畾涔夊洜瀛?鈹溾攢鈹€ configs/         # 瀹為獙閰嶇疆
鈹斺攢鈹€ runs/            # 姣忔瀹為獙鐨勮嚜鍖呭惈杈撳嚭
```

## 宸茬煡杈圭晫

- **Tick 绾х爺绌?*锛氬綋鍓嶄笉淇濈暀姝ｅ紡浠ｇ爜鍖呫€俆ushare 涓嶆彁渚?Tick 鏁版嵁锛屾湭鏉ュ瀵规帴 CTP 鎴?Wind锛屽簲鍗曠嫭璁捐鏁版嵁 adapter銆佽鍗曠翱/閫愮瑪瀛樺偍涓庤瘎浼板彛寰勩€?- **`src/factorzen/research/combination/`**锛氬疄楠屾€у鍥犲瓙鍚堟垚锛堢瓑鏉冦€両C 鍔犳潈銆丮ax-IR锛夛紝鐢ㄤ簬鐮旂┒闃舵瀵规瘮銆傚綋鍓嶆潈閲嶄及璁′粛鏄?in-sample 鍙ｅ緞锛屼笉鑳芥妸缁勫悎缁撴灉绉颁负 OOS锛屼篃涓嶈兘鐩存帴瑙ｉ噴涓虹敓浜у彲浜ゆ槗缁勫悎浼樺寲锛岃 [`src/factorzen/research/combination/README.md`](src/factorzen/research/combination/README.md)銆?- **鐢熶骇浜ゆ槗杈圭晫**锛氬綋鍓嶆鏋惰仛鐒︾爺绌跺彲淇″害锛屼笉鍖呭惈 tick 鏁版嵁銆佸疄鐩?OMS銆佺洏鍙ｆ垚浜ゃ€佺湡瀹?Tushare 缃戠粶 smoke 鎴栫敓浜х骇缁勫悎鎵ц闂幆銆?
