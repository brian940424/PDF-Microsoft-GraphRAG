# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PREFER_BINARY=1 \
    PROJECTS_ROOT=/app/projects \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

WORKDIR /app

# Keep the large GraphRAG dependency layer independent from application source.
# BuildKit reuses both this layer and pip's download cache on later builds.
COPY requirements.docker.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements.docker.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-deps . \
    && mkdir -p /app/projects

VOLUME ["/app/projects"]
EXPOSE 7860

CMD ["automotive-graphrag"]
