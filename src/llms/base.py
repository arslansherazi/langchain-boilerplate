from langchain_core.messages import HumanMessage


class BaseLLM:
    def __init__(self, model_name: str):
        self.__model_name = model_name
        self.__llm = None

    def get_response(self, query: str) -> str:
        messages = [
            HumanMessage(content=query)
        ]
        result = self.__llm(messages)
        results_data = result.content
        return results_data

    def get_llm(self):
        return self.__llm
