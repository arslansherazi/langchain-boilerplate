"""
Chatbot Controller
"""
import sys

import gradio as gr
from dotenv import load_dotenv
import time

from commons.constants import CHATBOT_NAME, USER_QUERY_PLACEHOLDER, GRADIO_CSS, GRADIO_TITLE_ELEMENT_ID
from commons.factory.llms import llm_factory
from src.rag.controller import RagController

load_dotenv()


class ChatbotController:
    """
    A controller class to manage the interaction between the user and the chatbot.
    """
    def __init__(self, _llm_identifier: str, _model_name: str):
        self._llm_identifier = _llm_identifier
        llm_klass = llm_factory(_llm_identifier)
        if not llm_klass:
            raise ValueError(f"Invalid LLM identifier: {llm_identifier}.")
        self.llm = llm_klass(_model_name)

    def get_response_generator(self, query: str) -> str:
        """
        Generates a response from the chatbot, yielding the response character by character.

        :param query: The user's input message.
        :return: A generator yielding each character of the chatbot's response.
        """
        retriever = RagController(query, self._llm_identifier).get_retriever()
        response = self.llm.get_response(query, retriever)
        for char in response:
            yield char
            time.sleep(0.03)

    def run(self):
        """
        Initializes and launches the Gradio interface for the chatbot.
        The interface includes a chatbot window, a prompt box at the bottom, and a send button.
        """
        with gr.Blocks(css=GRADIO_CSS) as demo:
            gr.Markdown(CHATBOT_NAME, elem_id=GRADIO_TITLE_ELEMENT_ID)
            chatbot = gr.Chatbot(elem_id="chatbox", height=450)

            with gr.Row(visible=True):
                with gr.Column():
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder=USER_QUERY_PLACEHOLDER,
                        lines=1,
                        max_lines=1,
                        autoscroll=False,
                        elem_id="user_input"
                    )

            def submit_message(query, chat_history):
                """
                Handles the submission of the user's message, updates the chat history,
                and clears the input box.

                :param query: The user's input message.
                :param chat_history: The current chat history.
                :return: A generator yielding the updated chat history with the live-typed
                         chatbot response.
                """
                # Append user's message immediately to chat history
                chat_history.append((query, ""))
                yield chat_history, ""

                response_generator = self.get_response_generator(query)
                bot_response = ""

                for char in response_generator:
                    bot_response += char
                    chat_history[-1] = (query, bot_response)
                    yield chat_history, ""

            # Connect input submission to the submit_message function
            user_input.submit(
                fn=submit_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
                queue=True,
            )

        demo.launch()


if __name__ == '__main__':
    try:
        llm_identifier = sys.argv[1]
        model_name = sys.argv[2]
        ChatbotController(llm_identifier, model_name).run()
    except IndexError:
        print("Provide the LLM identifier and model name as command line arguments.")
