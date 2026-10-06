"""Model-free heuristic base score: tokens of sentences with no counterpart get a high score, the rest 0."""
from functools import lru_cache

from .aligner import SentenceAligner
from .segmentation import apply_punct_mask, segment_document


@lru_cache(maxsize=None)
def doc_segments(text_a, unt_a, text_b, unt_b, lang):
    return segment_document(text_a, unt_a, "en"), segment_document(text_b, unt_b, lang)


def heuristic_labels(row: dict, lang: str, aligner: SentenceAligner, unaligned: float = 1.0, complex_: float = 0.0):
    """-> (labels_a, labels_b) token lists with -1 on ASCII punctuation. Only text fields of `row` are used."""
    en, tg = doc_segments(row["text_a"], row["text_a_untokenized"], row["text_b"], row["text_b_untokenized"], lang)
    ta, tb = row["text_a"].split(), row["text_b"].split()
    a, b = [0.0] * len(ta), [0.0] * len(tb)
    for l in aligner.align(en, tg, lang):
        v = unaligned if l.kind.startswith("unaligned") else (complex_ if l.kind != "1-to-1" else 0.0)
        for i in l.en:
            a[en[i].start:en[i].end] = [v] * (en[i].end - en[i].start)
        for j in l.tgt:
            b[tg[j].start:tg[j].end] = [v] * (tg[j].end - tg[j].start)
    return apply_punct_mask(ta, a), apply_punct_mask(tb, b)
