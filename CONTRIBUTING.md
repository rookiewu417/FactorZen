# 贡献指南

感谢参与 FactorZen。本项目当前聚焦**日频因子评估与报告**。

## 环境

```bash
pixi install
pre-commit install        # 启用提交前 ruff + mypy 钩子
cp .env.example .env       # 配置 TUSHARE_TOKEN(不入库)
```

## 提交前必过的质量门

```bash
pixi run lint        # ruff
pixi run typecheck   # mypy(src/factorzen,必须 0 错)
pixi run test        # pytest
pixi run coverage    # 覆盖率
```

CI 在 push / PR 到 `master` 时运行同一套门。**typecheck 失败会跳过测试**,务必本地先跑通。

## 工作流约定

- **分支:** 不直接提交 `master`,从 `master` 切 `fix/`、`feat/`、`chore/`、`docs/` 分支,经 PR 合并。
- **提交信息:** 遵循 Conventional Commits(`fix:` / `feat:` / `chore:` / `docs:` / `test:`)。
- **作者身份:** `rookiewu417 <1007372080@qq.com>`。
- **TDD:** bugfix 与新功能先写失败测试,再实现;关键路径必须有回归测试。
- **无未来函数:** 评估代码严禁引入前瞻偏差;涉及收益/价格对齐的改动需有 lookahead 安全测试。

## 编码风格

- 行宽由 ruff-format 管理;遵循 `pyproject.toml` 的 ruff 规则集。
- 中文注释/字符串保持正确 UTF-8,**不要**用 GBK 或 latin1 保存源文件/文档。
- 新增因子放在 `workspace/factors/{daily,...}`,框架代码放在 `src/factorzen/`。

## 报告问题

通过 GitHub Issues 提交。安全问题请勿公开,见 [SECURITY.md](SECURITY.md)。
