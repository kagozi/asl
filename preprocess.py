import re
import string

_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def remove_noise(txt: str) -> str:
    if txt is None:
        return ""
    txt = re.sub(r"\d+", "", str(txt))
    txt = txt.translate(_PUNCT_TABLE)
    txt = " ".join(txt.split())
    return txt


def preprocess_text(text: str) -> str:
    return remove_noise(str(text).lower())


def preprocess_gloss(gloss: str) -> str:
    return remove_noise(str(gloss).upper())
