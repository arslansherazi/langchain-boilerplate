from enum import Enum


class EnumMixin:
    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)


class OpenAiModel(Enum, EnumMixin):
    GPT_35_TURBO = "gpt-3.5-turbo"


CHAT_BOT_INITIAL_PROMPT = """
You are a smart e-commerce assistant dedicated to Costco. Answer user queries specifically related to Costco products, 
pricing, availability, and other related information. Politely decline any questions that are not directly related to 
Costco e-commerce.
"""