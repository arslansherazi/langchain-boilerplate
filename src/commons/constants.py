from enum import Enum


class OpenAiModel(Enum):
    GPT_35_TURBO = "gpt-3.5-turbo"


CHAT_BOT_INITIAL_PROMPT = """
You are an intelligent e-commerce assistant. Answer the user's queries about products, pricing, availability, and more.
"""