FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY . /app
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .[dev]

CMD ["python", "-m", "src.main", "--help"]
