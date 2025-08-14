from abc import ABC, abstractmethod

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

from config import ElevenLabsConfig


class TTSProvider(ABC):
    @abstractmethod
    def generate_speech(self, speech_text: str) -> bytes:
        """Returns the speech audio file using speech_text."""


class ElevenLabsTTS(TTSProvider):
    DEFAULT_OUTPUT_FORMAT = "pcm_16000"
    SAMPLE_RATE = 16_000

    def __init__(self, config: ElevenLabsConfig):
        self.config = config
        self.client = ElevenLabs(api_key=self.config.api_key)

    @property
    def voice_settings(self) -> VoiceSettings:
        return VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.1,
        )

    def generate_speech(self, speech_text: str) -> bytes:
        response = self.client.text_to_speech.convert(
            voice_id=self.config.voice_id,
            output_format=self.DEFAULT_OUTPUT_FORMAT,
            text=speech_text,
            model_id=self.config.model,
            voice_settings=self.voice_settings,
        )
        return b"".join(response)
