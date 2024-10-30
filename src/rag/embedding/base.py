from typing import List, Any

from langchain_community.vectorstores import Chroma

from src.commons.constants import VECTOR_DB_DIR


class Embedding:
    def __init__(self, document_splits: List, embedding: Any):
        self.document_splits = document_splits
        self.embedding = embedding
        self.vector_db = self._initialize_vector_db()

    def _initialize_vector_db(self):
        """
        Initialize the vector database.
        """
        return Chroma.from_documents(
            documents=self.document_splits,
            embedding=self.embedding,
            persist_directory=VECTOR_DB_DIR
        )
