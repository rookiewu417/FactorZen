# FactorZen 无人值守运营镜像:pixi 环境 + fz CLI。
# 构建: docker build -t factorzen:latest -f deploy/docker/Dockerfile .
# 运行: docker run --rm --env-file .env -v $PWD/data:/app/data -v $PWD/workspace:/app/workspace \
#         factorzen:latest ops daily --config workspace/configs/ops.yaml
FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app

# 先拷依赖清单与源码(editable 安装需要 src 与 pyproject),利用镜像层缓存
COPY pixi.toml pixi.lock pyproject.toml README.md ./
COPY src/ ./src/

# 按锁文件精确安装(含 editable 的 factorzen 本体),保证与本地/CI 一致
RUN pixi install --locked

ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# 入口固定为 fz;compose/命令行再追加子命令如 `ops daily --config ...`
ENTRYPOINT ["pixi", "run", "fz"]
CMD ["--help"]
