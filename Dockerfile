FROM python:3.13.12-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app
COPY ./frontend_dist /code/frontend_dist

ARG VOICES_URL=https://github.com/queai-project/QueAI-TTS-CPU-LOCAL-MS/releases/download/v1.0.0/voices.tar.gz

RUN mkdir -p /code/voices \
    && curl -fL "$VOICES_URL" -o /tmp/voices.tar.gz \
    && tar -xzf /tmp/voices.tar.gz -C /code \
    && rm /tmp/voices.tar.gz

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
