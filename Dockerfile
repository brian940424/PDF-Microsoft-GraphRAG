# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PROJECTS_ROOT=/app/projects \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Override at build time when another mirror is faster or required.
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# Install uv separately so dependency resolution and downloads use its parallel Rust implementation.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --index-url "${PYPI_INDEX_URL}" uv

# Keep the large GraphRAG dependency layer independent from application source.
# BuildKit reuses both this layer and uv's package cache on later builds.
COPY requirements.docker.txt ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --index-url "${PYPI_INDEX_URL}" -r requirements.docker.txt

# Source changes only invalidate the lightweight application install layer below.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-deps . \
    && mkdir -p /app/projects

VOLUME ["/app/projects"]
EXPOSE 7860

CMD ["automotive-graphrag"]
