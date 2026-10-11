FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The image runs the Claude backend: the local models need more memory than small cloud instances
# have. Install the local extra on a larger machine to run them here too.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[claude,postgres]"

# Run as an unprivileged user (uid 1000, which Hugging Face Spaces expects); it owns the trace store.
RUN useradd --create-home --uid 1000 vantage && mkdir -p /app/data && chown vantage /app/data
USER vantage

ENV LLM_PROVIDER=claude
EXPOSE 8000
HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/api/health')" || exit 1

# Render and Hugging Face Spaces set PORT; elsewhere it is 8000.
CMD ["sh", "-c", "exec uvicorn vantage.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
