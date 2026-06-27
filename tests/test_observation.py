"""Observation serialization (the static-text channel is the key bit)."""

from agent.observation import Observation, _dedup_substrings, prune_observation


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


# --- AgentOccam-style pruning ----------------------------------------------


def test_dedup_substrings_drops_contained_fragments():
    lines = ["Great product, 5 stars, by Catso", "Catso", "5 stars", "Totally separate"]
    out = _dedup_substrings(lines)
    assert "Great product, 5 stars, by Catso" in out
    assert "Catso" not in out and "5 stars" not in out      # fragments dropped
    assert "Totally separate" in out                         # unrelated kept


def test_prune_keeps_distinct_action_targets():
    # Two distinct "Add to cart" buttons (different products) MUST both survive —
    # pruning never de-duplicates by name.
    o = Observation(
        url="u", title="t",
        elements=[
            {"ref": "@e1", "role": "button", "name": "Add to cart", "rect": {"x": 0, "y": 0}},
            {"ref": "@e2", "role": "button", "name": "Add to cart", "rect": {"x": 0, "y": 90}},
        ],
        texts=["Widget A $5", "$5", "Widget B $9", "$9"],
    )
    pruned = prune_observation(o)
    assert len(pruned.elements) == 2                          # both targets kept
    assert "$5" not in pruned.texts and "$9" not in pruned.texts  # fragments gone
    assert pruned.texts == ["Widget A $5", "Widget B $9"]


def test_prune_drops_exact_duplicate_elements():
    dup = {"ref": "@e1", "role": "link", "name": "Home", "rect": {"x": 1, "y": 1}}
    o = Observation(url="u", title="t", elements=[dup, dict(dup)], texts=[])
    assert len(prune_observation(o).elements) == 1
