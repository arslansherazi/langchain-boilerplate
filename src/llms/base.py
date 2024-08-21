from langchain_core.messages import HumanMessage


class BaseLLM:
    def __init__(self):
        self.llm = None

    def get_response(self, query: str) -> str:
        messages = [
            HumanMessage(content=query)
        ]
        result = self.llm(messages)
        results_data = result.content
        return results_data
