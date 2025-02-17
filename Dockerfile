FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN pip install -e .

RUN mkdir -p models/production

EXPOSE 8000

CMD uvicorn src.api.main:app --host 0.0.0.0 --port $PORT 