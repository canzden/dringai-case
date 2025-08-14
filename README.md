# dringai-case

This repo contains simple real time voice assistant designed for hands-free, conversational interactions which are ideal for customer service and support.

## What could be done better?
- Latency is huge, the data should flow through the streams reduce latency
- More robust error handling, maybe custom exceptions
- Smarter assistant/llm logic
---


## Installation & Usage
I'm not gonna include installation & usage with Docker, because I was unable to test it since I'm not on Linux and do not really have time to create a VM. The image is building though.

```bash
git clone https://github.com/canzden/dringai-case.git
cd dringai-case

# create and activate vent
python -m venv .venv
source .venv/bin/activate

# generate deps from .in file to avoid os specific deps
pip install pip-tools
pip-compile -o requirements.txt requirements.in

# install deps
pip install requirements.txt

# copy .env file and fill api keys
cp .env.example .env

# run (assuming accessibility settings are granted)
python src/orchestrator.py

```
## Logs
Each new conversation is saved in to data/logs as jsonl files.
```json
{
    "ts": "2025-08-13T23:55:37.075+00:00",
    "turn_id": 1,
    "user_text": "Merhaba, bana randevu konusunda yardımcı edin misin lütfen?",
    "assistant_text": "Merhaba, ben DringAI müşteri hizmetleri asistanıyım. Randevu konusunda size nasıl yardımcı olabilirim? Hangi tarih ve saat için randevu almak istiyorsunuz?"
}
```
