import gradio as gr
from dotenv import load_dotenv

from src.llms.openai import OpenAiLLM

load_dotenv()


class ChatbotController:
    """
    A controller class to manage the interaction between the user and the e-commerce chatbot.
    """
    @staticmethod
    def get_response(query: str, chat_history: list) -> list:
        """
        Generates a response from the e-commerce chatbot and updates the chat history.

        :param query: The user's input message.
        :param chat_history: A list of tuples containing the previous chat history,
                             where each tuple is (user_message, bot_response).
        :return: Updated chat history with the new user query and chatbot response.
        """
        response = OpenAiLLM().get_response(query)
        chat_history.append((query, response))
        return chat_history

    def run(self):
        """
        Initializes and launches the Gradio interface for the e-commerce chatbot.
        The interface includes a chatbot window, a prompt box at the bottom, and a send button.
        """
        with gr.Blocks(css=".container {max-width: 100%; margin: 10px;}") as demo:
            gr.Markdown("## E-commerce Chatbot", elem_id="title")
            chatbot = gr.Chatbot(height=750)

            with gr.Row(visible=True):
                with gr.Column(scale=10):
                    user_input = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message...",
                        lines=1,
                    )
                with gr.Column(scale=1, min_width=60):
                    submit_button = gr.Button("Send")

            def submit_message(query, chat_history):
                """
                Handles the submission of the user's message, updates the chat history,
                and clears the input box.

                :param query: The user's input message.
                :param chat_history: The current chat history.
                :return: Updated chat history and an empty string to clear the input box.
                """
                updated_history = self.get_response(query, chat_history)
                return updated_history, ""

            # Connect the button click and input submission to the submit_message function
            submit_button.click(
                fn=submit_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
                queue=False,
            )

            user_input.submit(
                fn=submit_message,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input],
                queue=False,
            )

        demo.launch()


if __name__ == '__main__':
    ChatbotController().run()
