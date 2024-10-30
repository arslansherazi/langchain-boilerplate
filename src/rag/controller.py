from singleton_decorator import singleton

from src.rag.retriever import DocumentRetriever


@singleton
class RagController:
    def __init__(self, query: str, llm_identifier: str):
        self.query = query
        self.llm_identifier = llm_identifier
        self._retriever = None

    def get_retriever(self) -> str:
        """
        Get the context text for the RAG model. Ensures the retriever is only instantiated once.

        :return: The context text.
        """
        if self._retriever is None:
            self._retriever = DocumentRetriever(self.query, self.llm_identifier).get_vector_db_retriever()
        return self._retriever
