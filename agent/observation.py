"""Observation layer: turn a live page into a compact, numbered element list
plus the salient static text on the page.

Primary modality is the accessibility-relevant DOM (deterministic, token-cheap,
robust to markup churn). Each interactable element is stamped with a stable
``data-webagent-ref`` attribute and addressed by the LLM as ``@e1``, ``@e2`` …
so actions can be validated and executed precisely.

Crucially we also extract **leaf static text** (prices, headings, paragraphs,
table cells) that isn't the name of any interactable element — otherwise an
information-extraction task like "find the price" has nothing to read.

Screenshots are the fallback modality (see ``agent.loop``); this module only
captures one when asked.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Injected once per snapshot. Tags interactable/visible elements and gathers
# leaf static text. Hidden/offscreen elements and empty-name noise are dropped
# here so the prompt stays small.
_SNAPSHOT_JS = r"""
(args) => {
  const maxElements = args.maxElements;
  const maxTexts = args.maxTexts;

  const SEL = [
    'a[href]', 'button', 'input', 'select', 'textarea', 'summary',
    '[role=button]', '[role=link]', '[role=checkbox]', '[role=radio]',
    '[role=tab]', '[role=menuitem]', '[role=option]', '[role=switch]',
    '[role=textbox]', '[role=combobox]', '[role=searchbox]',
    '[contenteditable=""]', '[contenteditable=true]', '[onclick]',
  ].join(',');

  const TEXT_SKIP = new Set(['SCRIPT','STYLE','NOSCRIPT','SVG','PATH','HEAD',
    'STYLE','TEMPLATE','IFRAME']);

  const isVisible = (el) => {
    if (el.disabled) return true; // keep disabled so the model sees them
    const style = window.getComputedStyle(el);
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    if (parseFloat(style.opacity || '1') === 0) return false;
    const r = el.getBoundingClientRect();
    if (r.width <= 1 && r.height <= 1) return false;
    return true;
  };

  const accName = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria && aria.trim()) return aria.trim();
    const labelledby = el.getAttribute('aria-labelledby');
    if (labelledby) {
      const parts = labelledby.split(/\s+/)
        .map(id => document.getElementById(id))
        .filter(Boolean).map(n => (n.textContent || '').trim());
      if (parts.join(' ').trim()) return parts.join(' ').trim();
    }
    if (el.labels && el.labels.length) {
      const t = Array.from(el.labels).map(l => (l.textContent || '').trim()).join(' ');
      if (t.trim()) return t.trim();
    }
    const txt = (el.innerText || el.textContent || '').trim();
    if (txt) return txt;
    return (el.getAttribute('placeholder') || el.getAttribute('value')
      || el.getAttribute('alt') || el.getAttribute('title')
      || el.getAttribute('name') || '').trim();
  };

  // Clear any stale refs from a previous snapshot.
  document.querySelectorAll('[data-webagent-ref]').forEach(
    el => el.removeAttribute('data-webagent-ref'));

  const seen = new Set();
  const out = [];
  let i = 0;
  for (const el of document.querySelectorAll(SEL)) {
    if (out.length >= maxElements) break;
    if (seen.has(el)) continue;
    seen.add(el);
    if (!isVisible(el)) continue;
    let name = accName(el);
    if (name.length > 120) name = name.slice(0, 117) + '...';
    const role = el.getAttribute('role')
      || ({A: 'link', BUTTON: 'button', INPUT: (el.type || 'textbox'),
           SELECT: 'select', TEXTAREA: 'textbox', SUMMARY: 'summary'}[el.tagName]
          || el.tagName.toLowerCase());
    const isInput = ['INPUT','TEXTAREA','SELECT'].includes(el.tagName);
    if (!name && !isInput) continue;
    i += 1;
    const ref = 'e' + i;
    el.setAttribute('data-webagent-ref', ref);
    const r = el.getBoundingClientRect();
    const rec = {
      ref: '@' + ref, role, name, tag: el.tagName.toLowerCase(),
      rect: {x: Math.round(r.x), y: Math.round(r.y),
             w: Math.round(r.width), h: Math.round(r.height)},
    };
    if (isInput) {
      rec.input_type = el.type || '';
      if (el.value) rec.value = String(el.value).slice(0, 80);
      if (el.placeholder) rec.placeholder = el.placeholder;
      if (el.tagName === 'SELECT') {
        rec.options = Array.from(el.options).slice(0, 30).map(o => o.text.trim());
      }
      if (el.checked !== undefined && (el.type === 'checkbox' || el.type === 'radio'))
        rec.checked = !!el.checked;
    }
    if (el.disabled) rec.disabled = true;
    out.push(rec);
  }

  // Static text not already captured as an interactable element's name.
  const interactNames = new Set(out.map(e => e.name));
  const texts = [];
  const seenText = new Set();

  // Pass 1: coherent "record" blocks — a review / product / list row emitted as
  // ONE unit so the model can associate fields (author + rating + body) instead
  // of seeing them as disconnected lines. Mark each record's subtree covered.
  const RECORD_SEL = '[class*="review" i],[class*="item" i],[class*="product" i],' +
    'li,tr,[role=listitem],[role=row]';
  const recordRoots = new Set();
  const covered = new Set();
  for (const el of document.querySelectorAll(RECORD_SEL)) {
    if (el.querySelector(RECORD_SEL)) continue;   // leaf-most record only
    if (!isVisible(el)) continue;
    const t = (el.innerText || '').replace(/\s+/g, ' ').trim();
    if (t.length < 15 || t.length > 500) continue;
    recordRoots.add(el);
    for (const d of el.querySelectorAll('*')) covered.add(d);
  }

  // Pass 2: DOM-order walk. Emit record blocks whole; for everything else take
  // its OWN direct text (so plain <div>/<span> content is captured too), but
  // skip anything already inside an emitted record to avoid duplication.
  for (const el of document.querySelectorAll('body *')) {
    if (texts.length >= maxTexts) break;
    if (TEXT_SKIP.has(el.tagName)) continue;
    let t;
    if (recordRoots.has(el)) {
      t = (el.innerText || '').replace(/\s+/g, ' ').trim();
    } else {
      if (covered.has(el)) continue;
      t = '';
      for (const n of el.childNodes) {
        if (n.nodeType === 3) t += n.textContent;   // text node only
      }
      t = t.replace(/\s+/g, ' ').trim();
    }
    if (t.length < 2) continue;
    if (!isVisible(el)) continue;
    if (t.length > 400) t = t.slice(0, 397) + '...';
    if (seenText.has(t) || interactNames.has(t)) continue;
    seenText.add(t);
    texts.push(t);
  }

  return {
    url: location.href,
    title: document.title,
    elements: out,
    texts: texts,
    truncated: out.length >= maxElements,
  };
}
"""


@dataclass
class Observation:
    url: str
    title: str
    elements: list[dict[str, Any]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    truncated: bool = False
    pagination: dict[str, Any] = field(default_factory=dict)  # {"next_ref", "label"}
    screenshot_b64: Optional[str] = None

    @property
    def refs(self) -> set[str]:
        return {e["ref"] for e in self.elements}

    def serialize(self, max_chars: int = 8000) -> str:
        """Compact text for the prompt: element list + salient page text."""
        lines = [f"URL: {self.url}", f"TITLE: {self.title}", "ELEMENTS:"]
        for e in self.elements:
            parts = [e["ref"], f"<{e['role']}>"]
            name = e.get("name", "")
            if name:
                parts.append(f'"{name}"')
            if e.get("input_type"):
                parts.append(f"[type={e['input_type']}]")
            if "value" in e:
                parts.append(f"[value={e['value']!r}]")
            elif e.get("placeholder"):
                parts.append(f"[placeholder={e['placeholder']!r}]")
            if "options" in e:
                opts = ", ".join(e["options"][:10])
                parts.append(f"[options: {opts}]")
            if e.get("checked") is not None:
                parts.append("[checked]" if e["checked"] else "[unchecked]")
            if e.get("disabled"):
                parts.append("[disabled]")
            lines.append("  " + " ".join(parts))
        if self.truncated:
            lines.append("  … (element list truncated)")
        if self.pagination.get("next_ref"):
            lines.append(
                f"PAGINATION: more pages exist — this is one page of a longer list. "
                f"Use {self.pagination['next_ref']} "
                f"({self.pagination.get('label', 'Next')!r}) to see further pages "
                f"before concluding 'none' or 'all'."
            )
        if self.texts:
            lines.append("PAGE TEXT:")
            for t in self.texts:
                lines.append(f"  - {t}")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n  … (observation truncated)"
        return text

    def hash(self) -> str:
        basis = self.url + "|" + "|".join(
            f"{e['ref']}:{e.get('role')}:{e.get('name','')}" for e in self.elements
        ) + "|TEXT:" + str(len(self.texts))
        return hashlib.sha1(basis.encode("utf-8", "ignore")).hexdigest()[:12]


# A "next page" control — exact arrow/word forms, or an aria/name flagged "next
# page". Kept tight to avoid matching product names that merely contain "next".
_NEXT_EXACT = {"next", "next page", "next »", "»", "›", "→", ">", "older", "older posts"}
_NEXT_RE = re.compile(r"\bnext\s+page\b", re.I)


def _detect_pagination(elements: list[dict[str, Any]]) -> dict[str, Any]:
    for e in elements:
        if e.get("role") not in ("link", "button"):
            continue
        name = (e.get("name") or "").strip()
        low = name.lower()
        if low in _NEXT_EXACT or _NEXT_RE.search(low) or (
            "next" in low and len(name) <= 12
        ):
            label = re.sub(r"\s+", " ", name).strip() or "Next"
            return {"next_ref": e["ref"], "label": label}
    return {}


def snapshot(page: Any, *, max_elements: int = 120, max_texts: int = 100) -> Observation:
    """Run the snapshot script against a Playwright page and build an Observation."""
    data = page.evaluate(
        _SNAPSHOT_JS, {"maxElements": max_elements, "maxTexts": max_texts}
    )
    return Observation(
        url=data["url"],
        title=data["title"],
        elements=data["elements"],
        texts=data.get("texts", []),
        truncated=data["truncated"],
        pagination=_detect_pagination(data["elements"]),
    )
