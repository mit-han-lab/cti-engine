"""Tokenizer worker, prompt encoding and text codecs; the package loads the worker only on
first use of ``tokenize_worker``."""

from cti.utils import lazy_getattr

from .detokenize import IncrementalDetokenizer
from .prompt import PromptEncoder
from .text import load_text_decoder

__getattr__ = lazy_getattr(__name__, {"tokenize_worker": ".server"}, globals())

__all__ = ["IncrementalDetokenizer", "PromptEncoder", "load_text_decoder", "tokenize_worker"]
