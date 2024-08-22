from typing import List

from langchain.agents import initialize_agent, AgentExecutor
from langchain_core.tools import Tool

from src.llms.openai import OpenAiLLM


class ProductsAgent:
    def __init__(self):
        self.__llm = OpenAiLLM().get_llm()
        self.__tools = self.__prepare_agent_tools()
        self.__agent = self.__initialize_agent()

    def __initialize_agent(self) -> AgentExecutor:
        agent = initialize_agent(self.__tools, self.__llm, agent="zero-shot-react-description", verbose=True)
        return agent

    def __prepare_agent_tools(self) -> List[Tool]:
        tools = []



        return tools
