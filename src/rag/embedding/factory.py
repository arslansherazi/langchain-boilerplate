from src.rag.embedding.openai import OpenAIEmbedding


def embedding_factory(llm_identifier: str):
    """
    factory to get the llm embedding using the identifier

    :param llm_identifier: LLM identifier example: openai, gemini etc.
    :return: LLM Embedding class
    """
    llm_embeddings = {
        "openai": OpenAIEmbedding
    }
    return llm_embeddings.get(llm_identifier)
