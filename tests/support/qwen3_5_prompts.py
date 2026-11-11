import json
import random

from tests.support.hf_truth import PROMPTS_PATH, tokenize_prompt

_FROZEN = json.loads(PROMPTS_PATH.read_text())
TEXTS = {prompt["name"]: prompt for prompt in _FROZEN["prompts"]}
MAX_IMAGE_TOKENS = 256
_LONG_TEXT = (
    "The survey team logged the tide, the wind and the light at every station along "
    "the coast, and they wrote each reading twice so that nothing would be lost. "
) * 12
# (text prefix, image names, question), in chat content order.
IMAGE_RECIPES = {
    "one-image": ("", ("shapes",), "Describe every shape in this picture and its color."),
    "two-images": ("", ("stripes", "checks"), "How do these two patterns differ?"),
    "image-after-long-text": (_LONG_TEXT, ("shapes",), "Which shape is the largest?"),
    "wide-image": ("", ("wide",), "Describe this picture."),
    "noise-image": ("", ("noise",), "Describe this picture."),
}
_QUESTIONS = (
    "What color is the circle?",
    "How many shapes are there?",
    "Which shape is on the right?",
    "Is there a triangle?",
)


def text_prompts(tokenizer, model_key):
    result = [
        (name, tokenize_prompt(tokenizer, prompt, _FROZEN["chat_template_kwargs"]))
        for name, prompt in TEXTS.items()
    ]
    if result != [(name, prompt["token_ids"][model_key]) for name, prompt in TEXTS.items()]:
        raise ValueError("pinned tokenizer/template token IDs mismatch")
    return result


def question_recipes():
    """One image under several questions, then each question on an image of its own."""

    same = {f"shapes-{i}": ("", ("shapes",), text) for i, text in enumerate(_QUESTIONS)}
    return same | {
        f"{image}-{i}": ("", (image,), text)
        for i, (image, text) in enumerate(zip(("stripes", "checks"), _QUESTIONS, strict=False))
    }


def eviction_recipes():
    """Every image under three questions, one image after another, so a small encoder pool
    holds some repeats and has evicted others."""

    images = ("shapes", "stripes", "checks", "wide", "noise")
    return {
        f"{image}-{i}": ("", (image,), text)
        for i, text in enumerate(_QUESTIONS[:3])
        for image in images
    }


def image_chats(recipes=None):
    """Deterministic images and chats shared by engine and reference processes, of
    ``IMAGE_RECIPES`` unless ``recipes`` are given."""
    from PIL import Image, ImageDraw

    shapes = Image.new("RGB", (448, 320), "white")
    draw = ImageDraw.Draw(shapes)
    draw.ellipse((40, 40, 200, 200), fill="red", outline="black", width=6)
    draw.rectangle((250, 80, 410, 280), fill="blue")
    draw.polygon([(120, 300), (220, 220), (320, 300)], fill="green")
    stripes = Image.new("RGB", (256, 256), "black")
    draw = ImageDraw.Draw(stripes)
    for index in range(0, 256, 32):
        draw.rectangle((index, 0, index + 15, 255), fill="yellow")
    checks = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(checks)
    for row in range(0, 64, 16):
        for column in range(0, 96, 16):
            if (row + column) // 16 % 2:
                draw.rectangle((column, row, column + 15, row + 15), fill="purple")
    images = {
        "shapes": shapes,
        "stripes": stripes,
        "checks": checks,
        "wide": shapes.resize((800, 128)),
        "noise": Image.frombytes(
            "RGB", (320, 224), random.Random(20261008).randbytes(320 * 224 * 3)
        ),
    }
    chats = {}
    for name, (prefix, image_names, question) in (recipes or IMAGE_RECIPES).items():
        content = [{"type": "text", "text": prefix}] if prefix else []
        content.extend({"type": "image", "image": images[key]} for key in image_names)
        content.append({"type": "text", "text": question})
        chats[name] = [{"role": "user", "content": content}]
    return chats


def processed_image_prompts(snapshot, recipes=None):
    from transformers import AutoConfig

    from cti.multimodal import ImageLimits
    from cti.tokenizer import PromptEncoder

    encoder = PromptEncoder.from_checkpoint(
        str(snapshot),
        hf_config=AutoConfig.from_pretrained(snapshot, local_files_only=True),
        limits=ImageLimits(max_images=4, max_image_tokens=MAX_IMAGE_TOKENS),
    )
    prompts = {
        name: encoder.chat(messages, template_kwargs={"enable_thinking": False})
        for name, messages in image_chats(recipes).items()
    }
    for prompt in prompts.values():
        encoder.check(prompt)  # as a request is checked before it reaches the engine
    return prompts
