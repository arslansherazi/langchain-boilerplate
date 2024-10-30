"""
Base Agent module for all agents
"""
from typing import List

from langchain.agents import initialize_agent, AgentExecutor
from langchain_core.tools import Tool

from src.llms.openai import OpenAiLLM


class BaseAgent:
    """
    Base agent class.
    """
    def __init__(self, llm, agent_name):
        self._llm = llm
        self._agent_name = agent_name
        self._tools = self._prepare_agent_tools()
        self._agent = self._initialize_agent()

    def _initialize_agent(self) -> AgentExecutor:
        """
        Initialize the agent.
        """
        agent = initialize_agent(self._tools, self._llm, agent=self._agent_name, verbose=True)
        return agent

    def _prepare_agent_tools(self) -> List[Tool]:
        """
        Prepare the agent tools.
        """
        pass

    def get_agent(self):
        """
        Get the agent object.
        """
        return self._agent
