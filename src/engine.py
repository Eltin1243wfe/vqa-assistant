"""
engine.py

This is the core piece that actually talks to the vision-language model.
Everything else (Gradio UI, CLI, whatever) is just a wrapper around this.

I went with Qwen2.5-VL-3B-Instruct as the default because it's small enough
to run on a single consumer GPU (or even CPU if you're patient) but still
gives genuinely good answers on real-world images. Swapping to LLaVA or a
bigger Qwen checkpoint is basically a one-line change - see the MODEL_NAME
constant below.

Note: first call to VQAEngine() will download the model weights from
Hugging Face (a few GB), so give it a minute the first time.
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Default model - small enough to be practical, still solid quality.
# Bigger options if you've got the VRAM: Qwen/Qwen2.5-VL-7B-Instruct
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

MAX_NEW_TOKENS = 256


class VQAEngine:
    """Thin wrapper around a Qwen2.5-VL checkpoint for visual Q&A.

    Keeps a conversation history internally so you can ask follow-up
    questions about the same image without re-uploading it.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        self.model_name = model_name

        # auto-pick device if caller didn't specify one
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[engine] loading {model_name} onto {device}... this can take a bit")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)

        self.processor = AutoProcessor.from_pretrained(model_name)

        print("[engine] model loaded, ready to go")

    def ask(self, image: Image.Image, question: str, history: list = None) -> str:
        """Ask a question about an image.

        history is a list of {"role": ..., "content": [...]} dicts in the
        Qwen chat-template format. Pass in whatever was returned from the
        previous call if you want multi-turn context, otherwise leave it
        as None for a fresh conversation.
        """
        if history is None:
            history = []

        messages = history + [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # generate() gives us the prompt tokens back too, so trim those off
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return answer.strip()
