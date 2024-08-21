import gradio as gr
from dotenv import load_dotenv
import time

from commons.constants import (
    CHATBOT_NAME, USER_QUERY_PLACEHOLDER, SEND_BTN_TEXT, GRADIO_CSS, GRADIO_TITLE_ELEMENT_ID
)
from src.llms.openai import OpenAiLLM

load_dotenv()


class ChatbotController:
    """
    A controller class to manage the interaction between the user and the e-commerce chatbot.
    """
    @staticmethod
    def get_response_generator(query: str):
        """
        Generates a response from the e-commerce chatbot, yielding the response character by character.

        :param query: The user's input message.
        :return: A generator yielding each character of the chatbot's response.
        """
        response = OpenAiLLM().get_response(query)
        for char in response:
            yield char
            time.sleep(0.03)

    def run(self):
        """
        Initializes and launches the Gradio interface for the e-commerce chatbot.
        The interface includes a chatbot window, a prompt box at the bottom, and a send button.
        """
        with gr.Blocks(css=GRADIO_CSS) as demo:
            gr.Markdown(CHATBOT_NAME, elem_id=GRADIO_TITLE_ELEMENT_ID)
            chatbot = gr.Chatbot(elem_id="chatbox", height=800)

            with gr.Row(visible=True):
                with gr.Column(scale=3, elem_id="user_input_container"):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder=USER_QUERY_PLACEHOLDER,
                        lines=1,
                        max_lines=1,
                        autoscroll=False,
                        elem_id="user_input"
                    )
                with gr.Column(scale=1):
                    submit_button = gr.Button(SEND_BTN_TEXT, elem_id="send_button")

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

            # Connect the button click and input submission to the submit_message function
            submit_button.click(
                fn=submit_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
                queue=True,
            )

            user_input.submit(
                fn=submit_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
                queue=True,
            )

        demo.launch()


if __name__ == '__main__':
    ChatbotController().run()
