---
title: "{{ replace .File.ContentBaseName "-" " " | title }}"
date: {{ .Date }}
draft: true
description: ""    # one sentence for a stranger — the search snippet and the card text
short: ""          # optional: short tab title (e.g. "batch effects")
# image: cover.jpg    # the photo banner — cut by scripts/banner.py into this post's bundle
# caption: "2024-11-03"   # under the banner: the date the photo was taken (place/name is Saber's call)
# images: [social.jpg]    # 1.91:1 crop banner.py writes beside cover.jpg — the LinkedIn/OG card
# pinned: true        # heads the landing-page rail regardless of date (one post at a time)
# links:
#   - label: Code
#     url: https://github.com/saberhq/sidechain
---

Intro paragraph.

## Section

Body. A ```mermaid fence renders as a diagram. Link within the site root-relative
(`[posts](/posts/)`); the /sidechain/ prefix is added at build time.
