"""
app.py

Launches the actual chatbot UI. Run with:

    python app.py

Upload an image, ask it something, keep asking follow-ups - it remembers
the conversation until you upload a new image or hit clear.

I'm lazy-loading the model (only load it once someone actually opens the
tab and the first question comes in) so the app starts up fast even
though the model itself takes a while to load.
"""

import gradio as gr

from src.engine import VQAEngine
from src.utils import load_and_prep_image

# loaded on first use, not at import time - keeps startup snappy
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = VQAEngine()
    return _engine


def respond(image, question, chat_display, raw_history):
    if image is None:
        chat_display = chat_display + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "Upload an image first and I'll take a look!"},
        ]
        return chat_display, raw_history, ""

    if not question or not question.strip():
        return chat_display, raw_history, ""

    engine = get_engine()
    prepped = load_and_prep_image(image)

    answer = engine.ask(prepped, question.strip(), history=raw_history)

    # keep our internal history in sync so follow-up questions have context
    updated_history = raw_history + [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": prepped},
                {"type": "text", "text": question.strip()},
            ],
        },
        {"role": "assistant", "content": answer},
    ]

    chat_display = chat_display + [
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer},
    ]
    return chat_display, updated_history, ""


def reset_conversation():
    return [], [], None


def reset_chat_only():
    # used when a new image comes in - wipe the conversation but don't
    # touch the image widget itself (that'd just erase what they uploaded)
    return [], []


with gr.Blocks(title="Visual Q&A Assistant") as demo:
    gr.Markdown(
        """
        # Visual Q&A Assistant
        Upload an image, ask questions about it, and follow up as much as you want.
        Powered by [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) running locally.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Image")
            clear_btn = gr.Button("Clear conversation")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversation", height=450)
            question_input = gr.Textbox(
                label="Your question",
                placeholder="What's going on in this image?",
                lines=1,
            )
            ask_btn = gr.Button("Ask", variant="primary")

    # internal state that keeps the raw message history for the model
    # (separate from chatbot's display-friendly tuples)
    history_state = gr.State([])

    ask_btn.click(
        respond,
        inputs=[image_input, question_input, chatbot, history_state],
        outputs=[chatbot, history_state, question_input],
    )
    question_input.submit(
        respond,
        inputs=[image_input, question_input, chatbot, history_state],
        outputs=[chatbot, history_state, question_input],
    )

    clear_btn.click(reset_conversation, outputs=[chatbot, history_state, image_input])
    image_input.change(reset_chat_only, outputs=[chatbot, history_state])


if __name__ == "__main__":
    demo.launch()