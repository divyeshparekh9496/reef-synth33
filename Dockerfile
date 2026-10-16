FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE ./
COPY reefsynth ./reefsynth
COPY models ./models
COPY jobs ./jobs
RUN pip install --no-cache-dir --no-deps . && chmod +x jobs/*.sh

EXPOSE 8000
# Default: serve the dashboard + API (Nebius Serverless Endpoint).
# For the batch generator (Nebius Serverless Job) override the command with jobs/generate_and_train.sh
CMD ["uvicorn", "reefsynth.api:app", "--host", "0.0.0.0", "--port", "8000"]
