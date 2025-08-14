
FROM python:3.13-slim

# installing system deps and io interface support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential python3-dev linux-libc-dev \
    libportaudio2 libasound2 libsndfile1 alsa-utils pulseaudio-utils \
 && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir pip-tools

COPY requirements.in ./

# generate requirements.txt here to avoid os dep packages
RUN pip-compile --quiet --upgrade -o requirements.txt requirements.in

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["python", "src/orchestrator.py"]
