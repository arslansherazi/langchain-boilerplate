from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


class OpenAiLLM:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")

    def get_response(self, query: str) -> str:
        messages = [
            HumanMessage(content=query)
        ]
        result = self.llm(messages)
        results_data = result.content
        return results_data
    