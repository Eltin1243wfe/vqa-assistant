# Visual Q&A Assistant

A chatbot that looks at an image and answers questions about it, powered by [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), an open-source vision-language model. Upload a photo, ask "what's happening here?" or "how many people are in this picture?", and keep asking follow-ups - it holds context across the conversation.

I built this to get hands-on with multimodal LLMs instead of sticking to text-only stuff. Vision-language models are where a lot of the interesting product work is happening right now, so I wanted something in my portfolio that actually shows I can wire one up end to end - not just call an API, but understand the chat-template format, image preprocessing, and inference pipeline underneath it.

## Demo

examples/screenshot.png

## Why Qwen2.5-VL

I went back and forth between LLaVA and Qwen2.5-VL. Landed on Qwen for a few reasons:

- Genuinely strong for its size - the 3B variant punches well above its weight on visual reasoning benchmarks
- Handles higher-resolution images natively without the aggressive downsampling older VLMs do
- Actively maintained, good docs, clean Hugging Face integration

Swapping to a different model (LLaVA, a bigger Qwen checkpoint, etc.) is a one-line change in `src/engine.py` - the rest of the app doesn't care what's underneath.

## Tech stack

- **Model:** Qwen2.5-VL-3B-Instruct (Hugging Face Transformers)
- **UI:** Gradio
- **Inference:** PyTorch, runs on GPU if you've got one, falls back to CPU (slow but works)

## Project layout

```
vqa-assistant/
├── app.py              # Gradio chatbot UI - main entry point
├── cli.py               # terminal-only version, useful for quick testing
├── src/
│   ├── engine.py         # the actual model wrapper / inference logic
│   └── utils.py          # image preprocessing helpers
├── tests/
│   └── test_utils.py     # unit tests for the non-model logic
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/vqa-assistant.git
cd vqa-assistant
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

You'll need a decent amount of disk space free the first time you run it - it pulls the model weights from Hugging Face (a few GB) and caches them locally.

## Usage

**Web UI:**

```bash
python app.py
```

Opens a local Gradio server (usually `http://127.0.0.1:7860`). Upload an image, ask away.

**Command line:**

```bash
python cli.py my_photo.jpg "what breed of dog is this?"
```

## How it works

1. Image gets loaded and resized down if it's huge (keeps inference reasonably fast and avoids OOM on the model)
2. Question + image get formatted into Qwen's chat-template message format
3. Processor tokenizes text and preprocesses the image together
4. Model generates a response, which gets decoded and shown back in the chat
5. That exchange gets appended to a running history, so follow-up questions have context about both the image and what's already been asked

## Known limitations

- No streaming yet - answers appear all at once rather than token-by-token
- Runs painfully slow on CPU-only machines; a GPU (even a modest one) makes a big difference
- Only handles single images per conversation right now, no multi-image comparisons

## Possible next steps

- Stream tokens as they generate instead of waiting for the full answer
- Add support for comparing two images side by side
- Dockerize it for easier deployment
- Try swapping in Qwen2.5-VL-7B for a quality comparison

## License

MIT - see [LICENSE](LICENSE)
