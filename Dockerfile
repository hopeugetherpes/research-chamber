FROM python:3.12-slim
WORKDIR /app
COPY live/requirements.txt live/
RUN pip install --no-cache-dir -r live/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
COPY live/ live/
COPY runs/exp39/broad_pain_direction.json runs/exp39/
CMD ["sh", "-c", "uvicorn server:app --app-dir live --host 0.0.0.0 --port ${PORT:-8000}"]
