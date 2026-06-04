# 贡献指南

感谢参与 FactorZen。本项目当前聚焦日频因子评估与报告，优先接受能提升研究可信度、可复现性、测试覆盖和文档清晰度的改动。

## 环境

```bash
pixi install
cp .env.example .env
pre-commit install
```

`.env` 只放本地凭据，不提交到仓库。

## 提交前质量门

```bash
pixi run lint
pixi run typecheck
pixi run test
pixi run coverage
```

CI 在 push / PR 到 `main` 或 `master` 时运行同一套检查。提交前请尽量在本地跑通。

## 工作流

- 从最新主分支创建功能分支，不直接向主分支提交。
- 提交信息建议遵循 Conventional Commits，例如 `fix:`、`feat:`、`docs:`、`test:`、`chore:`。
- bugfix 和核心行为变更应先有能复现问题的测试，再实现修复。
- 涉及收益、价格、成交约束或样本切分的改动，必须说明是否可能引入未来函数，并补充相应回归测试。
- 不提交本地行情数据、运行产物、日志、notebook checkpoint、`.env` 或任何 token。

## 编码风格

- Python 代码由 ruff 和 ruff-format 统一处理。
- 中文文档和注释使用 UTF-8 保存。
- 框架代码放在 `src/factorzen/`；用户可扩展因子放在 `workspace/factors/{daily,weekly,monthly,intraday}/`。
- 保持改动聚焦，避免顺手重排无关代码。

## 报告问题

普通问题请通过 GitHub Issues 提交。安全问题不要公开披露，处理方式见 [SECURITY.md](SECURITY.md)。
