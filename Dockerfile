FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

RUN mkdir -p models/production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WORKERS=1
ENV PORT=8000

EXPOSE ${PORT}

# Run with optimized parameters
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --limit-concurrency 1 --timeout-keep-alive 75 