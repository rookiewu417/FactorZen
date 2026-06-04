# 安全策略

## 报告漏洞

请通过私有渠道(邮件 1007372080@qq.com)报告安全问题,**不要**在公开 Issue 中披露。

## 凭据管理

- `.env`、Tushare token、GitHub token **不得入库**;`.env` 已在 `.gitignore` 中。
- CI 中的密钥通过 GitHub Actions Secrets(如 `TUSHARE_TOKEN`)注入,不写在工作流文件里。
