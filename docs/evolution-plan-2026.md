# FactorZen 2026 婕旇繘璁″垝

鏈€鍚庢洿鏂帮細2026-05-30

## 鐩爣

鐭湡鐩爣涓嶆槸鎵╁睍鏇村棰戠巼鎴栧疄鐩樺姛鑳斤紝鑰屾槸鎶婁綆棰戝洜瀛愮爺绌剁殑鍙俊搴︾户缁仛瀹烇細

1. 鍘熷鏁版嵁瀹屾暣鎬у彲瀹¤銆?2. 姣忔瀹為獙鐨?universe銆侀厤缃拰缁撴灉鍙鐜般€?3. 鎶ュ憡鑳芥竻妤氬尯鍒嗘牱鏈唴銆佹牱鏈鍜屽疄楠屾€х粨璁恒€?4. 璐ㄩ噺闂ㄨ鐩栨鏋跺叧閿矾寰勩€?
## 浼樺厛绾у師鍒?
```text
鏁版嵁姝ｇ‘鎬?-> 鍙鐜版€?-> 鐮旂┒缁撹鍙俊搴?-> 宸ョ▼璐ㄩ噺闂?```

`intraday` 鏆傛椂缁存寔鐜扮姸锛屼笉绾冲叆涓嬩竴杞川閲忛棬鎵╁紶銆俆ick 绾х爺绌朵粛涓嶈繘鍏ユ寮忎唬鐮佸寘銆?
## Phase 1锛氬師濮嬫暟鎹畬鏁存€?
鐩爣锛氱敤涓€鏉″懡浠ゅ璁℃湰鍦?`data/raw/` 鏄惁瓒充互鏀拺鏌愪釜鐮旂┒鍖洪棿銆?
寤鸿鏂板锛?
- `src/factorzen/core/data_audit.py`
- `fz data audit` 瀛愬懡浠?- `tests/test_data_audit.py`
- `tests/test_loader.py`

瀹炵幇绾︽潫锛?
- 瀹¤鏈湴 parquet锛屼笉鐩磋繛 Tushare銆?- 杈撳嚭 JSON 鍜屼汉绫诲彲璇绘憳瑕併€?- JSON 缁撴瀯灏介噺瀵归綈鐜版湁 `build_daily_quality_report` 鐨?`status/checks/warnings/errors` 椋庢牸銆?
楠屾敹锛?
```bash
pixi run fz data audit --data-type daily_basic --universe csi300 --start 20230101 --end 20231231
pixi run test
pixi run typecheck
pixi run lint
```

## Phase 2锛氬疄楠屽彲澶嶇幇鎬?
鐩爣锛歚workspace/factor_evaluations/{run_id}` 鑳界嫭绔嬪洖绛斺€滆繖娆″疄楠屽埌搴曠敤浜嗕粈涔堚€濄€?
寤鸿澧炲己锛?
- universe 蹇収锛歳un 寮€濮嬫椂鎶婃渶缁堣偂绁ㄦ睜鏄庣粏銆佹潵婧愩€佹槸鍚﹂檷绾у啓鍏?`workspace/factor_evaluations/{run_id}/universe.parquet`銆?- run 绱㈠紩锛氱淮鎶よ交閲忕储寮曪紝渚嬪 `workspace/factor_evaluations/index.jsonl` 鎴?SQLite銆?- manifest 鎵╁睍锛氱户缁褰曞懡浠ゃ€侀厤缃€乬it SHA銆乨irty 鐘舵€併€乣pixi.lock` hash銆佽緭鍑烘枃浠惰矾寰勫拰澶辫触鍘熷洜銆?
楠屾敹锛?
```bash
pixi run fz factor run momentum_20d --start 20250101 --end 20260513
pixi run fz report path <run_id>
```

## Phase 3锛氱爺绌剁粨璁哄彲淇″害

鐩爣锛氶伩鍏嶆妸瀹為獙鎬х粨鏋滆璇绘垚鐢熶骇绾ф垨鏃犲亸鏍锋湰澶栫粨璁恒€?
寤鸿澧炲己锛?
- 鍦ㄧ粍鍚堢爺绌舵姤鍛婂拰 manifest 涓樉寮忔爣娉?`research/combination` 鐨勬牱鏈唴鏉冮噸浼拌杈圭晫銆?- 瀵?`src/factorzen/daily/evaluation/cost_models.py` 澧炲姞鏇村畬鏁寸殑鏋佺鍦烘櫙娴嬭瘯銆?- 鏀跺彛鍥炴祴涓拰 `adv_20d`銆佸啿鍑绘垚鏈浉鍏崇殑 TODO锛屼繚璇侀厤缃拰瀹為檯浣跨敤璺緞涓€鑷淬€?- OOS 鍖哄潡鍦ㄦ牱鏈笉瓒虫椂缁欏嚭鏄庣‘鍘熷洜銆?
楠屾敹锛?
```bash
pixi run fz report build momentum_20d --start 20250101 --end 20260513
pixi run test
```

## Phase 4锛氳川閲忛棬鏀跺彛

鐩爣锛氭鏋跺叧閿矾寰勮繘鍏ョǔ瀹氳川閲忛棬锛屼釜浜哄疄楠屼粛淇濇寔杞婚噺銆?
寤鸿锛?
- mypy 缁х画瑕嗙洊 `src/factorzen`銆?- coverage 淇濇寔褰撳墠闂ㄦ锛屾柊澧炴牳蹇冩ā鍧楀繀椤绘湁娴嬭瘯銆?- `workspace/factors/` 涓嶅己鍒惰繘鍏ュ畬鏁?coverage锛屼絾搴旇兘閫氳繃鍗曞洜瀛?CLI 杩愯楠岃瘉銆?- 鐪熷疄 Tushare smoke 淇濇寔鎵嬪姩瑙﹀彂锛屼笉杩涘叆榛樿 CI銆?
鎺ㄨ崘鍥炲綊鍛戒护锛?
```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
```

## 鏆備笉鎶曞叆

- Tick 鏁版嵁鎺ュ叆銆?- 瀹炵洏 OMS / EMS銆?- 鐢熶骇缁勫悎鎵ц闂幆銆?- 鏂板鏃ュ唴鍥犲瓙涓荤嚎銆?
