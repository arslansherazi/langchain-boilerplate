from src.llms.openai import OpenAiLLM


def llm_factory(llm_identifier: str):
    """
    factory to get the llm using the identifier

    :param llm_identifier: LLM identifier example: openai, gemini etc.
    :return: LLM class
    """
    llms = {
        "openai": OpenAiLLM
    }
    return llms.get(llm_identifier) or OpenAiLLM
