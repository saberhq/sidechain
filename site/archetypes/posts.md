---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
date: {{ .Date }}
draft: true
description: ""    # one sentence for a stranger — the search snippet and the card text
short: ""          # optional: short tab title (e.g. "batch effects")
# image: cover.jpg   # beside index.md in a bundle (posts/<slug>/index.md), or /images/… under site/static/
# links:
#   - label: Code
#     url: https://github.com/saberhq/sidechain
---

Intro paragraph.

## Section

Body. A ```mermaid fence renders as a diagram. Link within the site root-relative
(`[posts](/posts/)`); the /sidechain/ prefix is added at build time.
