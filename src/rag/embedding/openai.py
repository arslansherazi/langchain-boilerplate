from typing import List

from langchain_openai import OpenAIEmbeddings

from src.rag.embedding.base import Embedding


class OpenAIEmbedding(Embedding):
    def __init__(self, document_splits: List):
        embedding = OpenAIEmbeddings()
        super().__init__(document_splits, embedding)
