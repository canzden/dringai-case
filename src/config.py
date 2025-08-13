import os

from dotenv import load_dotenv

load_dotenv()


class BaseConfig:
    pass


class DeepgramConfig(BaseConfig):
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        language: str | None = None,
        smart_format: bool = True,
        punctuate: bool = True,
        base_url: str = "https://api.deepgram.com/v1",
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        self.model = model or os.getenv("DEEPGRAM_MODEL")
        self.language = language or os.getenv("DEEPGRAM_LANGUAGE")

        if not self.api_key:
            raise ValueError("Deepgram API key is not set")
        if not self.model:
            raise ValueError("Deepgram model is not set")
        if not self.language:
            raise ValueError("Deepgram language is not set")

        self.smart_format = smart_format
        self.punctuate = punctuate
        self.base_url = base_url
