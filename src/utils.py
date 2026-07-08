"""
Small helpers that don't really belong anywhere else.
"""

from PIL import Image

MAX_DIM = 1536  # downscale big images so we don't choke the model / VRAM


def load_and_prep_image(path_or_image) -> Image.Image:
    """Accepts either a filepath/PIL Image and returns a clean RGB PIL image,
    resized down if it's huge. Gradio usually hands us a PIL image directly,
    but I kept the path option around for the CLI script and tests.
    """
    if isinstance(path_or_image, str):
        img = Image.open(path_or_image)
    else:
        img = path_or_image

    img = img.convert("RGB")

    if max(img.size) > MAX_DIM:
        img.thumbnail((MAX_DIM, MAX_DIM))

    return img


def format_history_for_display(chat_history):
    """Turns our internal Qwen-style message list into the simple
    (user, bot) tuple pairs Gradio's Chatbot component wants.
    """
    pairs = []
    pending_user = None

    for msg in chat_history:
        if msg["role"] == "user":
            # content is a list of blocks, grab the text one
            text_block = next(
                (b["text"] for b in msg["content"] if b.get("type") == "text"), ""
            )
            pending_user = text_block
        elif msg["role"] == "assistant" and pending_user is not None:
            pairs.append((pending_user, msg["content"]))
            pending_user = None

    return pairs
