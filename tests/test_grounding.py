"""Answer-grounding signal (anti-hallucination)."""

from agent.loop import _answer_grounding, _content_tokens

CORPUS = "blue widget $19.99 add to cart red gadget $42.00 canberra population 484,630"


def test_empty_answer_is_none():
    assert _answer_grounding("", CORPUS) is None
    assert _answer_grounding("the and of", CORPUS) is None  # all stopwords/short


def test_grounded_answer_scores_high():
    assert _answer_grounding("The Blue Widget costs $19.99.", CORPUS) >= 0.8


def test_hallucinated_number_scores_low():
    # Same words, wrong figure — the weighted number drags it below threshold.
    assert _answer_grounding("The Blue Widget costs $873.00.", CORPUS) < 0.5


def test_grounded_population_scores_high():
    assert _answer_grounding("Canberra population is 484,630.", CORPUS) >= 0.8


def test_numbers_weigh_more_than_words():
    grounded = _answer_grounding("Blue Widget $19.99", CORPUS)
    wrong_num = _answer_grounding("Blue Widget $77777", CORPUS)
    assert grounded > wrong_num


def test_content_tokens_drops_filler():
    toks = _content_tokens("The price is approximately 19.99 dollars")
    assert "the" not in toks and "approximately" not in toks
    assert "19.99" in toks and "dollars" in toks
