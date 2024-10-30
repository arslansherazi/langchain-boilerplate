from typing import List

from src.rag.embedding.factory import embedding_factory
from src.rag.splitter import DocumentSplitter


class DocumentRetriever:
    def __init__(self, query: str, llm_identifier: str):
        self.query = query
        self.llm_identifier = llm_identifier

    @staticmethod
    def _get_document_splits():
        """
        Get the document splits.
        :return:  document splits
        """
        return DocumentSplitter().get_document_splits()


    def _get_vector_db(self, document_splits: List):
        """
        Get the vector database.

        :param document_splits: document splits
        :return: vector DB
        """
        document_embedding_klass = embedding_factory(self.llm_identifier)
        document_embeddings = document_embedding_klass(document_splits)
        return document_embeddings.vector_db

    def get_vector_db_retriever(self):
        """
        Get the vector database retriever.
        """
        document_splits = self._get_document_splits()
        vector_db = self._get_vector_db(document_splits)
        return vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.5})

