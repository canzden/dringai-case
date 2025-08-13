from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

from config import load_config


class STTProvider(ABC):
    @abstractmethod
    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """Returns the transcript for the passed WAV bytes."""


class DeepgramSTT(STTProvider):
    """
    DeepgramSTT is an extension of the STTProvider for transcribing audio using the Deepgram API.

    Attributes:
        DEFAULT_TIMEOUT_SECONDS (int): Default timeout for API requests in seconds.
    """

    DEFAULT_TIMEOUT_SECONDS = 30

    def __init__(self):
        self.config = load_config()

    @property
    def _headers(self):
        return {
            "Authorization": f"Token {self.config.api_key}",
            "Content-Type": "audio/wav",
        }

    def transcribe_wav(
        self,
        wav_bytes: bytes,
        *,
        model: Optional[str] = None,
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribes speech from WAV audio bytes using a remote STT (Speech-to-Text) service.

        Args:
            wav_bytes (bytes): The WAV audio data to be transcribed.
            model (Optional[str], optional): The speech recognition model to use. Defaults to the configured model.
            language (Optional[str], optional): The language of the audio. Defaults to the configured language.

        Returns:
            str: The transcribed text from the audio, or an empty string if no audio is provided.

        Raises:
            requests.HTTPError: If the HTTP request to the STT service fails.
        """
        ENDPOINT = "/listen"

        if not wav_bytes:
            return ""
        url = self.config.base_url + ENDPOINT

        params: Dict[str, Any] = {
            "model": model or self.config.model,
            "language": language or self.config.language,
        }
        if self.config.smart_format:
            params["smart_format"] = "true"
        if self.config.punctuate:
            params["punctuate"] = "true"

        response = requests.post(
            url,
            headers=self._headers,
            params=params,
            data=wav_bytes,
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        response = response.json()

        try:
            return response["results"]["channels"][0]["alternatives"][0].get("transcript", "")
        except Exception:
            return str(response)
