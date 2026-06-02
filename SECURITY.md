# 安全策略

## 报告漏洞

请通过私有渠道(邮件 1007372080@qq.com)报告安全问题,**不要**在公开 Issue 中披露。

## 凭据管理

- `.env`、Tushare token、GitHub token **不得入库**;`.env` 已在 `.gitignore` 中。
- CI 中的密钥通过 GitHub Actions Secrets(如 `TUSHARE_TOKEN`)注入,不写在工作流文件里。

## ⚠️ 待处理:轮换暴露的凭据

以下凭据曾以明文出现在开发环境中(本地 `~/.claude/CLAUDE.md`、git remote URL),应尽快**轮换**:

- **GitHub Personal Access Token (classic, `ghp_...`)** —— 当前嵌在 `git remote get-url origin` 的 URL 里。
  建议:
  1. 在 GitHub → Settings → Developer settings → Tokens 中**吊销该 token 并新建**(按最小权限授予 `repo`)。
  2. 改用凭据助手而非把 token 写进 remote URL:
     ```bash
     git remote set-url origin https://github.com/rookiewu417/FactorZen.git
     git config --global credential.helper store   # 或使用 GCM / gh auth
     ```
- **Tushare token** —— 若曾出现在明文配置/提交中,同样建议在 Tushare 控制台重置。

> 私有仓库不会自动启用分支保护(需 GitHub Pro 或公开仓库)。在条件允许时,建议对 `master` 启用"必须通过 CI + PR 审查"保护。
