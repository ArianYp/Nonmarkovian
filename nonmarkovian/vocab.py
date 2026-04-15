TOKENS = ["A", "C", "G", "T", "[M]"]
VOCAB_SIZE = 5  # A,C,G,T,MASK
MASK_IDX = 4
IDX_TO_TOKEN = {i: t for i, t in enumerate(TOKENS)}


def token_to_idx(ch: str) -> int:
    ch = ch.upper()
    if ch == "M" or ch == "[M]":
        return MASK_IDX
    return TOKENS.index(ch)
