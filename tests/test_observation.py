"""Observation serialization (the static-text channel is the key bit)."""

from agent.observation import Observation


def _obs():
    return Observation(
        url="https://shop.example/p",
        title="Widget",
        elements=[
            {"ref": "@e1", "role": "button", "name": "Add to cart"},
            {"ref": "@e2", "role": "textbox", "name": "Qty", "input_type": "number",
             "value": "1"},
        ],
        texts=["Blue Widget", "$19.99", "In stock"],
    )


def test_serialize_includes_elements_and_text():
    s = _obs().serialize()
    assert "@e1" in s and "Add to cart" in s
    assert "PAGE TEXT:" in s
    assert "$19.99" in s  # extraction tasks depend on this


def test_serialize_truncates():
    o = _obs()
    o.texts = [f"line {i}" for i in range(1000)]
    s = o.serialize(max_chars=200)
    assert len(s) <= 260 and "truncated" in s


def test_refs():
    assert _obs().refs == {"@e1", "@e2"}


def test_pagination_detection():
    from agent.observation import _detect_pagination
    # Magento-style next control (whitespace in the accessible name) is detected.
    p = _detect_pagination([
        {"ref": "@e1", "role": "link", "name": "1"},
        {"ref": "@e9", "role": "link", "name": "Page\nNext"},
    ])
    assert p["next_ref"] == "@e9" and p["label"] == "Page Next"
    # A product whose name merely contains "next" is NOT a pagination control.
    assert _detect_pagination(
        [{"ref": "@e1", "role": "link", "name": "Next Gen Console Stand XL"}]
    ) == {}


def test_pagination_surfaced_in_serialize():
    o = Observation(url="u", title="t",
                    elements=[{"ref": "@e9", "role": "link", "name": "Next"}],
                    pagination={"next_ref": "@e9", "label": "Next"})
    assert "PAGINATION:" in o.serialize() and "@e9" in o.serialize()


def test_hash_is_stable_and_sensitive():
    a, b = _obs(), _obs()
    assert a.hash() == b.hash()
    b.elements = b.elements[:1]
    assert a.hash() != b.hash()
