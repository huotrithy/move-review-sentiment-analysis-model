FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"

WORKDIR /app
COPY app.py app.py
COPY deploy deploy
COPY pyproject-ui-deploy.toml pyproject.toml

RUN uv add \
    onnxruntime \
    torch>=2.9.0 \
    transformers[torch]>=4.57.1 \
    streamlit>=1.51.0

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]