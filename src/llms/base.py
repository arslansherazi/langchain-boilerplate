"""
Base module for all LLMs
"""
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


class BaseLLM:
    """
    Base class for all LLMs
    """
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._llm = None
        self._verify_model_name()
        self._initialize_llm()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def get_response(self, query: str, retriever) -> str:
        """
        Get response from the LLM

        :param query: user query
        :param retriever: vector DB retriever
        :return: llm response
        """
        conversational_retrieval_chain = ConversationalRetrievalChain.from_llm(
            self._llm,
            retriever=retriever,
            memory=self.memory
        )
        result = conversational_retrieval_chain.invoke({"question": query})
        results_data = result.get("answer") or "unable to response this query"
        return results_data

    def get_llm(self):
        """
        Get the llm object

        :return: llm object
        """
        return self._llm

    def _verify_model_name(self):
        """
        Verify the model name. If the model name is not valid, set the default model name.
        """
        pass

    def _initialize_llm(self):
        """
        Initialize the OpenAi LLM
        """
        pass
