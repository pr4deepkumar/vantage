FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package requirements and install dependencies
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY data/ ./data/
COPY dbt/ ./dbt/

# Install the package (non-editable; src/ must exist first)
RUN pip install --no-cache-dir .

# Expose FastAPI application port
EXPOSE 8000

# Healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8000/api/overview || exit 1

# Launch FastAPI application server
CMD ["uvicorn", "vantage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

