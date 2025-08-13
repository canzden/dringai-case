from abc import ABC, abstractmethod

from openai import OpenAI

from config import OpenAIConfig


class LLMProvider(ABC):
    @abstractmethod
    def reply_short(
        self,
        user_text: str,
    ) -> str:
        """Returns short reply using user prompt."""


class OpenAIClient(LLMProvider):
    AGENT_INSTRUCTION = "Sen DringAI sirketinde calisan ve turkce konusan bir musteri hizmetleri asistanisin, konusma yeni baslatiliyorsa kendini kisaca tanit. Musteriden sana gelen isteklere kisa ve oz cevap ver"
    RESPONSE_MAX_TOKENS = 120
    TEMPERATURE = 0.4

    def __init__(self, config: OpenAIConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=self.config.api_key)
        self.latest_response_id = None

    def reply_short(self, user_text: str) -> str:
        """
        Generates a short reply to the given user text using the configured language model.
        Tries to keep chat context by tracking previous message id.

        Args:
            user_text (str): The input text from the user.

        Returns:
            str: The generated reply text. Returns an empty string if the input is empty or whitespace.
        """
        if not user_text.strip():
            return ""

        response = self.client.responses.create(
            model=self.config.model,
            max_output_tokens=self.RESPONSE_MAX_TOKENS,
            temperature=self.TEMPERATURE,
            previous_response_id=self.latest_response_id,
            instructions=self.AGENT_INSTRUCTION,
            input=[{"role": "user", "content": user_text}],
        )

        self.latest_response_id = response.id

        return response.output_text
