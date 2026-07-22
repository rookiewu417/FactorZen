# Changelog

本文件记录值得注意的变更，遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本遵循 [SemVer](https://semver.org/lang/zh-CN/)。

条目中的 `#NNN` 为本仓库的 Pull Request 编号，可在 GitHub 上查证。

## [Unreleased]

自 v0.3.0 起，项目从「端到端可复现的量化研究平台」重心转向**以因子库准入为核心的多市场研究平台**：候选因子必须相对既有因子库跑出统计显著的增量（lift）才能进库。同期完成了多市场扩展、分钟级研究一等公民化、一轮全链路性能优化、一轮内存治理，以及回测双轨分离与一轮结构级清障。测试套件经归并重构后为 952 个用例 / 97 个测试文件。

### Added

#### 因子库与增量准入

- **准入架构倒置（#97）：** lift 增量检验升级为**入库的最终裁决**，单因子门降级为排序信号，硬门只剩数据质量；灰区候选进入无上界的 lift 队列，组门短路；三个消费方统一走同一套准入逻辑。
- **试用池与第二条入库路（#93）：** 单因子指标偏弱但对组合有增量的因子经 probation 通道入库。
- **向前确认机制（#101）：** 新增 `fz factor-library forward-track` 记录与 `forward-review` 裁决，构成 probation → forward → promote 的完整生命周期；`--apply` 之前默认 dry-run。
- **准入证据可重放（#99, #100, #101）：** `FactorRecord` 持久化 lift 准入 provenance（评估窗口、CV 参数、阈值、基线 hash），事后可重放准入统计量；新增 `evidence_tier` 与 probation 封顶。
- **跨 session 多重检验记账（#101）：** campaign trial family 让 Deflated Sharpe 的 N 跨 session 累计，并提供 `fz mine team --no-campaign-prior` 逃生口。
- **唯一登记簿（#124）：** `factor_library` 成为因子的单一登记簿，`FactorRecord` 四态状态机（active / correlated / probation / no_lift）；Python 因子经 `py::` 哨兵复用 expression 主键实现零改键；物化分派收敛为单点；废除导出 `.py` 桥。
- **库消费闭环（#126）：** 新增 `fz combine from-library`，直接用在库因子做多因子组合；Python 因子面板落磁盘缓存（缓存键含源码 hash）。
- **评估证据链接（#127）：** 因子记录关联最近一次评估的 run，且不覆盖准入裁决指标。
- **组合增量目标（#85）：** 挖掘目标残差化——用「对现有库的增量 IC」而非孤立 IC 评估候选，`--objective residual` 成为默认。
- **库级正交过滤（#81）：** 搜索期即按与在库因子的相关性过滤候选。
- **因子资产库（#172, #174）：** 每个因子物化为 `meta.json` + `factor.py` + `factor.parquet` 三件套，落 `workspace/factor_store/<market>/<name>/`；`factor_library` 的 jsonl 仍是**裁决唯一真相**，store 只是资产载体，两者口径正式分离。新增 `fz factor-library store sync` / `store verify`；expression 因子自动生成可执行 `factor.py`，防漂移测试锁定与生产求值逐位一致。物化口径统一为全 A + 2016 起。
- **残差增量口径（#131）：** lift 引擎从 LightGBM walk-forward 改为对在库因子做正交化后的残差 IC 口径 `residual_ic_v1`。
- **lift 轨去相关门（#132）：** lift 通道补上此前缺失的相关性门；同批新增 `ts_decay_linear` 真实现。
- **裸 IC 同号门（#135）：** 裸 IC 与库内方向相反的候选不再准入——等权组合无法表达负贡献。
- **定向重估与组门 sub-floor（#138）：** `rebuild` 支持只重估指定来源，解锁积压的重估账；组门补 sub-floor 通道。
- **稀疏事件因子 sleeve（#168–#171）：** 稀疏因子改用事件掩码评分并以 overlay 方式叠加，避免 fill-0 在组合层稀释；overlay 专用 lift 阈值经 500 次掩码内置换的 null 校准确定。
- **lift 统计层 null 校准（#138 后续）：** 新增 `fz factor-library lift-null`，在「无真实 lift」的 H0 下扫描参数网格的误准入率。

#### 多市场

- **crypto USDT-M 永续（#24, #25）：** Ports & Adapters 架构下的完整 crypto 适配，含 Binance Vision 分钟级数据湖。
- **多市场挖掘（#78）：** A 股 / crypto / 期货三市场 LLM 挖掘全通；期货主力连续后复权；新增美股市场支持（自建 Yahoo provider + S&P 500 快照 universe）。

#### 分钟级与日内研究

- **日内特征引擎（#106）：** A 股分钟 session 单一真源 + 17 个微观结构特征电池，分钟 bar 聚合为日频特征面板并作为挖掘叶子接入全链（语义零回归）。**#143 补入涨跌停邻域三叶后为 20 个。**
- **端到端透传（#107）：** research 链路与无人值守日链路接入日内面板的增量构建。
- **日内表达式 Scout（#108）：** bar 级表达式求值器 + `ix_*` 内容寻址动态叶子注册表；LLM 每轮提案 bar 表达式并注入 session，动态扩展搜索空间。

#### 新数据源与挖掘叶子

- **两融叶子（#86）：** `margin_detail`，T+1 滞后结构性内置。
- **股东户数与龙虎榜叶子（#88）。**
- **叶子反馈（#80）：** 挖掘过程回灌叶子健康度，开局自动摘除死叶。
- **涨跌停邻域三叶（#143）：** `i_limit_up_seal_share` / `i_limit_up_open_count` / `i_limit_up_first_touch`——离散状态机与连续路径统计正交，是库饱和后首批过门的新叶。
- **业绩预告 / 快报事件叶（#167）：** `fc_type_score` / `fc_surprise` / `fc_flag` / `express_yoy`，按公告日 PIT 对齐至 t+1。
- **阈值与游程算子（#166）：** `ts_count_gt` / `ts_streak_gt` / `ts_count_cross_up`，让表达式能表达「连续 N 日超阈值」这类此前只能做成叶子的结构。

#### LLM 挖掘

- **研究范式对齐（#60）：** 多目标、护栏对齐、自愈循环纳入求值期诊断、任务分解、结构化假设、自适应终止。
- **双 profile 适配（#89）：** 可在两套 OpenAI 兼容网关间切换。
- **轮内并行（#95）：** 独立 LLM 调用并发，`--llm-workers 1` 为串行零回归。
- **反馈闭环（#117）：** lift 拒绝原因回写实验登记簿并注入下一轮提案；exhausted 表达式族硬过滤；按族聚类识别拥挤叶子。
- **提案质量（#118）：** rank 指纹去重、未知算子不进自愈循环、窗口字面量按预算钳制、空轮跳过 Critic、Critic 注入残差与库相关信息。

#### 组合研究

- **四方法样本外对比（#28）：** 等权 / IC 加权 / max_ir / LightGBM，含因子重要性解释（shap 可选、gain 兜底）与 `fz combine run`。
- **换手与带成本净收益（#142, #144）：** 组合对比表补上换手、`net_spread_10bp` 与 `net_sharpe_10bp`——实测 IC 最高的方法换手也最高，只看 IC 会选错。
- **组合直通回测（#173）：** 各方法的 OOS 分数拼接落盘 `oos_scores/<method>.parquet`（折间日期零重叠，重叠即 fail-loudly），新增 `fz combine backtest` 把任意分数面板送进日环回测引擎；`--rebalance-days` 以调仓日降采样 + 按股票 ffill 实现。组合的可实现净值自此一条命令可得。

#### 运营与展示

- **无人值守日链路（#27）：** 8 阶段幂等编排（守卫 → 取数 → 审计 → 日内特征 → 信号 → 执行 → 报告 → 发布），失败告警、非交易日短路、失败处续跑；`fz ops daily` / `fz ops status`；Docker / compose、systemd timer 与 Windows 任务计划兜底。
- **只读服务层（#30）：** FastAPI 只读 REST API（health / runs / detail / nav + OpenAPI）+ 单页 Web Dashboard；`pixi run serve` 启动。

#### 编排

- **research 编排器（#52）：** `fz research run` 串起单因子研究链路。
- **挖掘引擎扩容（#26）：** 算子库扩充，`ts_skew` / `ts_rank` 改用 polars 原生 rolling 实现。
- **库池子进程预构建（#123）：** 新增顶层命令 `fz pool-prebuild` 与 `fz mine team --pool-subproc`，把因子库池的内存尖峰隔离到子进程，退出即全额归还；池缓存可跨 session 复用。

### Changed

#### 命令面与口径 ⚠️ 含破坏性变更

- **⚠️ 回测双轨分离（#177）：** `fz factor run` **已删除，不留别名**，拆成两条语义分明的轨：
  - `fz factor eval` —— 因子研究评估（信号层，纯向量化：IC / 分层 / 多空 / 单调性 / 换手），**毛口径**；
  - `fz factor backtest` —— 模拟交易回测（日环撮合 + 交易约束 + 成本），**净口径**，含 walk-forward。

  两轨参数面一致、产物不互撞（eval 轨 HTML 用 `_eval.html`）。同一因子在两轨下的差异是真实存在的：
  实测 `momentum_20d` / csi300 / 2024H1 信号轨毛多空 +10.82%，交易轨净 −2.14%。
- **⚠️ 信号轨移除全部成本参数（#180）：** `--cost-bps`、`ls_ret_net`、`nav_net` 及所有 `*_net` 指标从信号轨移除。
  粗略的 bps 折算既非真实撮合，又让人误以为研究轨能算净收益。`ls_turnover` 保留，但语义收窄为「信号换手强度」。
  交易轨的 `fz combine backtest --cost-bps` 不受影响。
- **⚠️ 挖掘与评估默认改为可实现口径（#145, #146, #175）：** 前向收益默认 `exec_lag=1` / `exec_price_col=open_adj`
  （即 t 日算、t+1 开盘成交），`--exec-lag 0` 为逃生口。此前的 close→close 默认隐含「T 日收盘成交」，
  与「t 日算 → t+1 执行」的 PIT 铁律矛盾。**库裁决链（`rebuild` / `lift-test` / `forward-track`）保持历史口径不动**，
  以维持已入库记录的可比性。跨版本对照数字时须注意口径差。
- **⚠️ CLI 结构级收敛（#175, #176）：** 顶层命令 16 → 14；`fz mine team` 参数面 32 → 15，
  高级参数改由 **`--set KEY=VALUE`** 通配传入（未知键 fail-loudly），`fz mine search` / `mine agent` /
  `factor-library lift-test` 同此。删除 `tag-legacy` / `render` / `runs show` 等低频命令，
  `config validate` 移入 `ops`、`pool-prebuild` 归入 `mine` 组。
- **⚠️ workspace 目录收敛（#178）：** 一级目录 21+ → 13。`workspace/factors/` **整树退役**——
  用户 Python 因子的唯一路径改为 `workspace/factor_store/`，三处 registry 不再扫描旧目录；
  运维杂项归并到 `workspace/_ops/`。
- **报告年化改几何口径（#179）：** 算术年化 `mean × 252` 在高波动下与几何年化差约 `σ²/2 × 252`，
  日波动 1.4~1.7% 即拖累 3~4pp，足以翻号——实测同一份报告里柱状图（算术）显示各组年化全正，
  而累计净值实际是亏的。展示一律改用与净值曲线同源的几何年化，算术值仅保留在 `summary_stats` 供跨轨比较。

#### 性能

一轮全链路性能优化（#110–#115），前后数值经等价性验证（挖掘候选表逐字节一致、回测净值 `max|Δ|` 在 1e-15 量级）：

- **单因子评估** csi500 两年窗口 17.25s → 7.08s；全 A 42.13s → 13.84s。
- **表达式挖掘** 50 trials（默认开库正交）426.37s → 31.39s。
- **Barra 风险模型构建** csi500 两年 170.98s → 4.14s；research 链路的风险段 40.2s → 1.07s。
- **日内特征电池**已覆盖月份增量跳过，6s/月 → 0.007s。
- 具体手段：财务 PIT 对齐向量化、行业/市值中性化改矩阵求解、指数成分 membership 改月度缓存 as-of 展开、库正交检查矩阵化、回测消除逐行迭代、挖掘评分只算 1 日 horizon。
- **组合管道提速（#82）：** 4.3×，IC / z-score / 面板全样本一次逐折切片。
- **贪心去相关提速（#84）：** 290×，共享网格紧凑矩阵。
- **CI 单遍并行（#87）：** 合并原本各跑一遍全量的 Test 与 Coverage 两步，总时长约减半。

#### 内存

一轮内存治理（#119–#123, #125, #128），目标是让全 A 长窗口挖掘在 24 GB 内存的机器上跑完：

- 因子库池改单骨架宽面板，超阈值自动切换。
- 数据帧瘦身：前向收益列收窄、字段白名单、单次预处理复用。
- 单副本纪律：`ts_code` 大规模时转 Categorical，键窄投影，逐层交接释放。
- 表达式求值有界化：时序子树按标的整批求值、中间列消费即弃、窄帧直算。
- 护栏相关面板：`present` 掩码改推导、超阈值降精度、日期分块流式归约，进而改为免物化的惰性宽网格。
- lift 并发按可用内存自适应（`max(2, min(4, 可用GB//5))`，上限 4，读取失败回退 2），修复 6 并发下的 OOM。

#### 其他

- **单因子评估精简（#109）：** 指标收敛为核心集（RankIC / 衰减 / 单调性 / 分位回测 / 换手 / walk-forward）+ 单页报告；移除 LLM 因子解读链路；默认策略改为 `quantile_ls_5` 单策略 + `csi500` 基准。
- **回测双路径收敛（#112, #114）：** 交易约束统一为单一约束核 `apply_trade_constraints_batch`（慢/快路径与纸面撮合三方共用）；慢/快路径合并为单一日环引擎，调仓日程语义统一取自权重表键。
- **数据根统一：** 全部数据（行情、数据湖、缓存、工具）收敛到 `data/`，`workspace/` 只放研究产出。
- **实盘定位：** 文档中实盘对接由「不覆盖」改写为分阶段路线目标，当前处于纸面向前执行阶段。
- **Walk-forward：** 策略 walk-forward 样本外评估改为**默认关闭**（`WalkForwardConfig.enabled` 默认 `false`），按需通过 YAML 或 `--set walk_forward.enabled=true` 开启。
- **报告模块解耦：** `tear_sheet.py` 2986 → 1054 行（-65%），按职责拆为 `_formatting` / `_scoring` / `_charts` / `_strategy` / `_summaries` 五个模块；经 re-export 保持对外导入接口不变。
- **报告按轨重建（#177，覆盖上一条）：** 上述 `tear_sheet.py` 随双轨分离**整体删除**——两轨共用一份报告会让 eval 轨的交易区块永远留空。
  现拆为 `signal_report.py`（分层净值 / 分层收益 / IC 时序·累计·衰减·分布 / 分层年度热力 / 换手，顶部带毛口径横幅）
  与 `trading_report.py`（净值 vs 基准 / 回撤 / **成本侵蚀瀑布** / **敞口时序** / **拒单原因分布** / 月度热力 / 滚动 Sharpe），两者区块零重叠。
  后三张图的数据源（`nav` 的 `gross_return` / `cost` / `cash_weight`、`trades.block_reason`）一直存在，只是旧报告从未消费。
- **测试库归并重构（#147–#164）：** 测试文件 326 → 97、用例 2,725 → 952，覆盖率基本持平。
  目录按模块组织（`tests/<模块>/`），根下只留架构守卫；大量小文件归并为多断言 suite。
  同批全仓清除 `monkeypatch.undo()`——它会撤销同实例上 fixture 打的离线 mock，导致后续断言静默走真实 API（本地绿、CI 红）。
- **工程化：** `.pre-commit-config.yaml` 改为通过 `pixi run` 的 local hooks，保证 pre-commit / CI / 本地三者版本一致。
- **CI：** 增加 `permissions: contents: read` 最小权限与 `concurrency` 取消重复运行；覆盖率门槛固定为 74%。
- **可复现性：** `run_experiment` 在工作树 dirty 时告警；manifest 增记 `duration_seconds`。

### Fixed

#### 核心正确性

- **嵌套 `over` 全 null（#61）：** 截面算子（`rank .over trade_date`）套时序算子（`ts_std .over ts_code`）时，编译出的单个嵌套表达式在 polars 下返回全 null，导致 IC 恒为 0 且静默失明；5 个物化点全部命中。
- **Agent 护栏系统性偏松（#62）：** Agent 侧 Deflated Sharpe 漏传 `sharpe_variance`，实测放松约 1.60×，真实运行中 2/2 候选裁决翻转。
- **风险模型静默丢日（#115）：** 因子集锁定在首个有效截面，导致 484 个交易日中 451 个被静默丢弃、协方差实际只用 33 天估出；修复后 R² 由 0.287 变为 0.322（有意变化）。
- **holdout 覆盖守卫（#79）：** 空截面的 `0.0` 哨兵造成误杀与假过关。
- **预热扩窗一致性（#75）：** holdout 段扩窗预热在全链透传，此前真实数据上 IC 偏差达 40%。
- **指数成分 as-of 漂移（#104 相关修复）：** 同一交易日的成分随查询窗口漂移，改为真逐日 as-of；月缓存缺月不再静默用更早快照顶替。
- **PIT 幸存者偏差（#101 后续修复）：** research / report 路径改逐日 PIT membership 过滤；membership 构造失败或空池改为 fail-closed。
- **执行前视（#101 后续修复）：** sizing 改用 `pre_close`，逐日 ST 与涨跌停判定收窄。
- **时序算子负/零窗口（#78）：** 根治由此引入的前视，并堵住历史回灌路径。
- **全库评审六波修复（#53–#59）：** 覆盖研究正确性、数据缓存与审计、执行链路加固、crypto 数据链、挖掘与 Agent 护栏、CLI 接线、挖掘模块收尾，共 66 条缺陷（3 个 P0 / 28 个 P1 / 35 个 P2）。

#### 护栏与统计

- **Deflated Sharpe 单双侧口径（#71）** 与 **最终基准 deflation 配方共享（#65, #72）。**
- **实验登记簿契约（#66）** 与 **索引窗口作用域（#67）。**
- **holdout 预热（#68）** 与 **异常记账卫生（#69）。**
- **遗传搜索种群卫生（#73）** 与 **薄截面告警（#74）。**
- **测试判别力（#70）：** 消灭一批恒真断言。
- **lift 队列与组门单点化（#116）：** 灰区地板按 2 SE 抬升；非 top-K 旁路统一走同一道候选门；覆盖过滤与组门收敛为单一实现，供 session 钩子与 CLI 两个消费方共用。
- **lift 批量对齐（#104）：** 多 session 候选按各自准入窗分组评分，修复选择后的样本外污染；`top_m` 静默截断改为默认全测。

#### 数据与链路

- **LLM 客户端韧性（#63, #90）：** 故障不再全损，改为重试 + 轮层容错 + 增量落盘；OpenAI 兼容网关流式响应异常统一包装。
- **挖掘循环韧性与 manifest 可复现（#64）。**
- **抓取健壮性（#91, #92）：** 数据抓取重试与市场模式判定。
- **因子库重建覆盖（#83）** 与 **共享面板全零行（#104 相关）：** 静默剔除与安全名撞列双根因，无信息不得伪装成强结论。
- **crypto / A 股 provider 频率守卫（#48）**、**行业 IC 退化截面（#49）**、**中性化回归失败返回 NaN（#50）**、**截面 rank 空值比例口径（#51）。**
- **借券成本按频率重复计费（#35）**、**风险模型 lookback 退化（#36）**、**执行归因正确性（#37）**、**执行回放续跑（#38）**、**组合优化约束（#39）**、**复权收盘价 IC（#40）**、**`factor sweep` 异常退出（#41）**、**内置财务因子必需数据声明（#42）**、**服务层输入校验（#43）**、**Tear Sheet 中 LLM 文本的 HTML 转义（#44）**、**Agent 护栏双向样本外判定（#45）**、**universe 停牌缺 bar（#46）**、**模拟交易权重上限（#47）。**
- **挖掘 alpha 选择缺陷（#33）** 与 **挖掘可观测性 / 护栏松紧（#77）。**
- **报告引擎：** 事件研究 `ci_95=None` 时模板对 `None` 下标取值导致整份报告崩溃；统一多空判定为单一 `_resolve_is_long_short`，修复概览与策略分页自相矛盾；图表辅助函数单列输入的 `StopIteration` 守卫；分位收益除零防护。
- **文档漂移：** 修正 walk-forward 默认行为、LLM 默认行为、Brinson 方法论描述、benchmark 遗漏与若干死链；修复合并中被双重编码损坏的中文文档。

### Security

- 新增 `SECURITY.md` 凭据管理、本地脱敏检查与凭据轮换流程；仓库当前文件与全部 git 历史经扫描确认无凭据泄露。

## [0.3.0]

详见 [docs/release-notes/v0.3.0.md](docs/release-notes/v0.3.0.md)。自 v0.2.0 起，项目从「A 股低频单因子研究框架」扩展为端到端、可复现的买方研究平台。测试套件扩展到 1109 个离线可重复用例。

### Added

- **因子挖掘引擎：** `discovery/` 算子库（30+ 时序/截面/算术算子）+ 表达式 AST 双向序列化 + 随机/遗传搜索（`fz mine search`，`--trials` 默认 200）+ IC 打分去相关；新增 `fz mine export-alpha` 把单个候选算成 `(ts_code, alpha)` 单截面 parquet，衔接组合优化。
- **防过拟合护栏：** `validation/` block bootstrap IC 置信区间、Deflated Sharpe Ratio、PBO/CSCV（候选池）、holdout 段永久隔离；`fz validate overfit` 打印单因子 IC/IR/DSR/bootstrap CI（不落盘，N=1 不计 PBO）。
- **Barra 风险模型：** `risk/` 8 个风格因子（size/value/momentum/volatility/liquidity/quality/growth/leverage）+ 中信一级行业因子 + Newey-West 协方差 + James-Stein 特质风险收缩 + 边际风险贡献；`fz risk build`（默认 `--cov-half-life 90 --nw-lags 2 --spec-shrinkage 0.3 --spec-half-life 90`）。
- **组合优化与归因：** `portfolio/` cvxpy mean-variance QP（CLARABEL solver，box/预算/换手/行业中性约束，单截面建仓）+ `attribution/` Brinson 多期归因与风险因子归因；`fz portfolio build`。
- **单/多 Agent 挖掘：** `agents/` 零外部依赖自建 LLM 闭环（假设→生成→护栏→IC 验证→反思）+ Negative RAG 失败注入（`fz mine agent`）；4 个角色 Agent（Hypothesis/Coder/Critic/Librarian）+ Evaluator 评估环节 + 跨 session 长期记忆（`fz mine team`）。
- **模拟交易 + 成果展示：** `sim/engine.py` 多周期权重回测（对齐行情、扣换手成本、净值与绩效指标）+ `reports/portfolio_report.py` 组合绩效 HTML Dashboard（指标卡 + 净值曲线 + 月度热图 + 归因 + 风险摘要）；`fz sim run` / `fz sim show` / `fz report portfolio`。
- **微观结构与交易约束：** `core/universe.py` universe 快照（停牌/涨跌停/ST/次新股/流通市值过滤）+ `core/benchmark.py` 基准管理（HS300/ZZ500/ZZ1000 + 行业等权替代基准）+ 回测引擎 GEM 双路径容差、T+1、`signal_date` 解耦、ADV 零值 fallback。
- **端到端教程：** 从拉数据到组合展示的完整链路逐步教程。

### Changed

- **定位升级：** 项目文档由单因子框架口径改写为端到端买方研究平台口径。
- **依赖：** 新增 cvxpy（CLARABEL solver）强依赖，组合优化功能依赖此包。

## [0.2.0]

见 [docs/release-notes/v0.2.0.md](docs/release-notes/v0.2.0.md)。

## [0.1.0]

见 [docs/release-notes/v0.1.0.md](docs/release-notes/v0.1.0.md)。
