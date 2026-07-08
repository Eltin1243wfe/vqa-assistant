"""
Keeping these light on purpose - anything that actually touches the model
needs a GPU and a few GB download, which isn't great for CI. So these just
cover the plain-Python helper logic in src/utils.py.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image
from src.utils import load_and_prep_image, format_history_for_display


def test_load_and_prep_image_converts_to_rgb():
    img = Image.new("RGBA", (100, 100))
    result = load_and_prep_image(img)
    assert result.mode == "RGB"


def test_load_and_prep_image_downscales_large_images():
    img = Image.new("RGB", (4000, 2000))
    result = load_and_prep_image(img)
    assert max(result.size) <= 1536


def test_load_and_prep_image_leaves_small_images_alone():
    img = Image.new("RGB", (200, 150))
    result = load_and_prep_image(img)
    assert result.size == (200, 150)


def test_format_history_for_display_pairs_correctly():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "what is this?"}]},
        {"role": "assistant", "content": "a cat"},
        {"role": "user", "content": [{"type": "text", "text": "what color?"}]},
        {"role": "assistant", "content": "orange"},
    ]
    pairs = format_history_for_display(history)
    assert pairs == [("what is this?", "a cat"), ("what color?", "orange")]


def test_format_history_for_display_empty_input():
    assert format_history_for_display([]) == []
