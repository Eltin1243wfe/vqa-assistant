"""
Quick command-line way to poke at the model without launching the full
Gradio app. Mostly useful for debugging or if you just want to test one
image from the terminal.

    python cli.py path/to/image.jpg "what's happening in this photo?"
"""

import sys

from src.engine import VQAEngine
from src.utils import load_and_prep_image


def main():
    if len(sys.argv) < 3:
        print('usage: python cli.py <image_path> "<question>"')
        sys.exit(1)

    image_path = sys.argv[1]
    question = " ".join(sys.argv[2:])

    image = load_and_prep_image(image_path)
    engine = VQAEngine()

    print(f"\nQ: {question}")
    print("A:", engine.ask(image, question))


if __name__ == "__main__":
    main()
