FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Copy only packaging metadata and source package first to maximize dependency-layer cache reuse.
COPY pyproject.toml LICENSE README.md /app/
COPY src /app/src

RUN pip install --upgrade pip && \
    pip install .

# Copy only runtime-required project folders/files (avoid copying the full repository context).
COPY configs /app/configs
COPY scripts /app/scripts
COPY Makefile /app/Makefile

CMD ["python", "-m", "src.main", "--help"]
