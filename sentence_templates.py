"""
Sentence templates: maps a sequence of confirmed signs to a full spoken
sentence. Actual template DATA lives in sentence_data.json (same folder)
-- edit that file to add/remove combinations, no code changes or rebuild
required, even after packaging this into an .exe with PyInstaller.

(Data file is named sentence_data.json, not sentence_templates.json, on
purpose -- keeping it clearly different from this .py filename makes
error messages and tracebacks easier to tell apart while debugging.)

Matching rule used by main.py / cv_main.py: checks if a template's word
sequence appears as a CONTIGUOUS, IN-ORDER subsequence of the most recent
confirmed signs (the word buffer). Order matters. Longer templates are
checked first so more specific matches win over shorter/more general ones.
"""

import json
import os
import sys


def _get_base_dir():
    # Same logic as main.py's get_base_dir() -- when frozen into an exe by
    # PyInstaller, sys.executable points at the exe itself, so this looks
    # for sentence_data.json sitting NEXT TO the exe, not baked inside it.
    # That's what makes it editable after building.
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _load_templates():
    base_dir = _get_base_dir()
    json_path = os.path.join(base_dir, "sentence_data.json")
    if not os.path.exists(json_path):
        print(f"ERROR: sentence_data.json not found at {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        data = json.load(f)

    templates = {}
    for entry in data.get("templates", []):
        words = tuple(entry["words"])
        templates[words] = entry["sentence"]
    return templates


# Loaded once when the module is imported. If you edit
# sentence_data.json, restart the program to pick up changes.
SENTENCE_TEMPLATES = _load_templates()


def match_sentence(word_buffer):
    """word_buffer: list of confirmed sign strings, most recent last.
    Returns the matched sentence string, or None if no template matches.
    Checks longer templates first so more specific matches win over
    shorter/more general ones."""
    for template in sorted(SENTENCE_TEMPLATES.keys(), key=len, reverse=True):
        n = len(template)
        if n > len(word_buffer):
            continue
        if tuple(word_buffer[-n:]) == template:
            return SENTENCE_TEMPLATES[template]
    return None