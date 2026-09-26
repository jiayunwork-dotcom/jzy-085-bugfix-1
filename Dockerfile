# 平面刚架直接刚度法核算服务
# 运行时锁定 Python 3.12
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 再拷业务代码
COPY framesolver ./framesolver

# 非 root 用户运行
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 一条命令对外提供接口
CMD ["uvicorn", "framesolver.main:app", "--host", "0.0.0.0", "--port", "8000"]
