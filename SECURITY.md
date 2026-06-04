# 安全策略

## 报告漏洞

请通过私有渠道报告安全问题，邮件：1007372080@qq.com。不要在公开 Issue、PR 或讨论区披露可利用细节。

## 凭据管理

- 不提交 `.env`、Tushare token、GitHub token、商业行情数据或私有研究产物。
- `.env` 已在 `.gitignore` 中，示例值只放在 `.env.example`。
- CI 密钥通过 GitHub Actions Secrets 注入，不能写进 workflow、源码、文档或测试快照。
- 不把 token 写进 `git remote` URL。优先使用 Git Credential Manager、系统凭据助手或 `gh auth`。

## 本地凭据检查

提交或开源前建议执行脱敏检查：

```bash
git status --short
git grep -n "TUSHARE_[T]OKEN\|g[h]p_\|github_[p]at_\|FACTORZEN_LLM_[A]PI_KEY" -- . ':!.env.example'
```

检查 remote URL 是否嵌入凭据时，不要把 URL 原文贴到 Issue 或日志里，只输出布尔结果：

```bash
url=$(git remote get-url origin)
case "$url" in
  *://*@*) echo 'origin-has-embedded-credential' ;;
  *)       echo 'origin-clean' ;;
esac
```

## 凭据轮换流程

如果 token 曾出现在本地配置、remote URL、日志、聊天记录、提交历史或第三方服务中，按泄露处理：

1. 在对应平台吊销旧 token。
2. 创建最小权限的新 token。
3. 清理本地 remote URL，例如：

   ```bash
   git remote set-url origin https://github.com/rookiewu417/FactorZen.git
   ```

4. 更新本机凭据助手或 GitHub Actions Secrets。
5. 用上面的脱敏检查确认仓库内容不含明文凭据。

## 分支保护

条件允许时，对 `main` / `master` 启用分支保护：必须通过 CI、禁止直接推送、敏感变更需要 PR 审查。
