#!/usr/bin/env bash
# 把 workspace/ops/site/ 发布到 gh-pages 分支的 live/ 路径。
# 用临时 git worktree 操作,绝不触碰主工作区的未跟踪文件。
# 用法: tools/publish_track_record.sh [site_dir]
#   默认 site_dir = <repo>/workspace/ops/site
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SITE_DIR="${1:-$REPO_ROOT/workspace/ops/site}"
TMP_WT="$(mktemp -d)"

cleanup() { git -C "$REPO_ROOT" worktree remove --force "$TMP_WT" 2>/dev/null || true; }
trap cleanup EXIT

if [ ! -f "$SITE_DIR/index.html" ]; then
  echo "错误: $SITE_DIR/index.html 不存在(先跑 fz ops daily 且 publish_enabled=true)" >&2
  exit 1
fi

cd "$REPO_ROOT"

# 确保 gh-pages 分支存在;首次创建孤儿分支
if git show-ref --verify --quiet refs/heads/gh-pages || \
   git ls-remote --exit-code --heads origin gh-pages >/dev/null 2>&1; then
  git worktree add "$TMP_WT" gh-pages
else
  git worktree add --detach "$TMP_WT"
  ( cd "$TMP_WT"
    git checkout --orphan gh-pages
    git rm -rf . >/dev/null 2>&1 || true
    echo "# FactorZen track record" > README.md
    git add README.md
    git commit -q -m "init gh-pages" )
fi

mkdir -p "$TMP_WT/live"
rm -rf "${TMP_WT:?}/live/"*
cp -r "$SITE_DIR"/. "$TMP_WT/live/"

cd "$TMP_WT"
git add live
if git diff --cached --quiet; then
  echo "无变更,跳过发布。"
else
  git commit -q -m "publish track record"
  git push origin gh-pages
  echo "已发布。GitHub Pages 生效后访问 https://<user>.github.io/<repo>/live/"
fi
