from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.commons.constants import EMPLOYEE_HANDBOOK_PATH


class DocumentSplitter:
    def __init__(self):
        self.loader =  None

    def load_document(self):
        """
        Load the document to be split.
        """
        loader = PyPDFLoader(EMPLOYEE_HANDBOOK_PATH)
        self.loader = loader.load()

    def _split_document(self):
        """
        Split the document into sections.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=250
        )
        splits = text_splitter.split_documents(self.loader)
        return splits

    def get_document_splits(self):
        """
        Get the document splits.

        :return: The document splits.
        """
        self.load_document()
        return self._split_document()
