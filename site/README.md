# site/ — saberhq.com/sidechain

Sidechain's public page: the scored submissions, what the project is, and the write-ups. A sibling of
[saberhq.com](https://saberhq.com): the same type and colours, its own wordmark and layout, built and deployed from
this repo. Notes: `private/code-notes/site/README.md.md`.

## How it is built

Themeless [Hugo](https://gohugo.io/). Every template is in `layouts/`; every colour, size and font comes from the
four token files in `assets/tokens/`. `assets/site.css` builds on them: the `sq-` rules are saberhq.com's chrome
carried over, the `sc-` rules are Sidechain's additions. The accent is carmine, `--accent-2` in the tokens, which
`site.css` re-points `--accent` at; re-measure contrast before changing a colour (4.5:1 for text on its ground).

**The kit.** The header, the post page and the post card are saberhq.com's, shared word for word with it: the
tokens, `assets/header.css`, `assets/post.css`, `assets/post-card.css`, `data/header.yaml`, `data/areas.yaml` and
`layouts/_partials/saberhq/`. saberhq.com holds the originals; these are copies, never edited here.
`../scripts/kit-pull` copies them from saberhq.com's latest release and records it in `kit.lock.json`; it refuses
while a file in saberhq.com's shared folders is in none of its lists. `../tests/test_site_kit.py` fails when a copy
differs from that release or from the lock. `post-card.css` and `post.css` load before `site.css`, so this site's rules override them; `header.css`
loads after it. The kit reads two settings in `hugo.toml`: `params.area` and `params.contentHistory`.

```
site/
├── hugo.toml              baseURL carries the /sidechain/ path
├── content/_index.md      landing page copy + the front matter the sections read
├── content/posts/         write-ups (markdown; see archetypes/posts.md)
├── content/*.md           standalone pages: the models page, the control-cells page
├── data/                  submissions.json — generated, see below; header.yaml + areas.yaml — the kit
├── layouts/               home, posts list, post, 404; partials (saberhq/ is the kit), render hooks, shortcodes
├── assets/                tokens/ + header.css, post.css, post-card.css (the kit); site.css, standing.js, controls.js; data/field.json — generated
└── static/                favicon and social card (scripts/wordmark.py); data/controls/ — generated
```

## Working on it

```bash
../scripts/preview.sh                   # BOTH public surfaces: this site AND the GitHub README
hugo server --source site --port 1414   # http://localhost:1414/sidechain/  (live reload)
hugo --source site --minify             # builds into site/public/ (gitignored)

hugo new --source site posts/my-post.md          # a draft post from the archetype
hugo server -D --source site --port 1414         # preview drafts too
uv run python scripts/wordmark.py            # README wordmark + favicon + social card
uv run python scripts/banner.py photo.jpg --slug my-post   # post banner from a photo
scripts/kit-pull --check                     # is the kit still saberhq.com's? (without --check: copy it in)
uv run python scripts/chroma_css.py          # code-block styles, after a Hugo upgrade
```

The preview runs on port 1414 (also in `.claude/launch.json`) because saberhq.com's `hugo server` owns 1313.

## For search engines and agents

Every page carries one JSON-LD block (`layouts/_partials/jsonld.html`): the author, the site, the code, and the page as
a `WebPage`, `CollectionPage`, `BlogPosting` or, on the control-cells page, a `Dataset`. The build also writes
`llms.txt` (`layouts/home.llms.txt`) and an `index.md` beside each page's `index.html` (`layouts/page.markdown.md`; a
shortcode used in a page needs a `*.markdown.md` twin). After each deploy, `../scripts/indexnow.py` sends the page list
to the IndexNow search engines. `../tests/test_site_seo.py` checks the built pages.

## Posts and pages

A post goes in `content/posts/`, as a plain file or as a page bundle with its images beside it:

```
content/posts/my-post.md              # plain post
content/posts/my-post/index.md        # page bundle: put its images beside it
content/posts/my-post/chart.png       #   and reference them as ![x](chart.png)
```

Front matter: `title`, `date`, `description` (the card text and the search snippet), `draft`, optional `short`,
`image` (with `image_alt`, and `image_light`, `image_light_alt`, `caption_light` for a light-theme photo), `caption`,
`links`, and `lastmod` (set it when a published post is edited; the post then prints "updated <date>" linked to the
file's history). A ```mermaid fence renders as a diagram, and `{{</* corpus */>}}` drops in the
corpus-coverage figure. Any page with `rail: true` in its front matter gets a card in the landing page's rail
(`layouts/_partials/page-card.html`: its `date`, `description`, and optionally `card_title`, `badge` and `cta`); a
`pinned: true` item heads the list.

Two ways a finished post stays invisible, both silent:

- **`draft: true`**: flip it to `false` to publish (`hugo server -D` previews).
- **A `date` in the future (UTC)**: Hugo drops future-dated content, and the site only rebuilds on a push, so the
  post stays missing until the next push after that date. Late-evening US dates are already tomorrow in UTC.

**The photo banner.** `image: cover.jpg` in the front matter, cut to a fixed 5:2 frame by `scripts/banner.py`, which
also strips EXIF and writes `social.jpg`, the 1.91:1 crop for link previews (list it as `images: [social.jpg]`). The
caption is the date the photo was taken (`caption: "2024-11-03"`).

**Links and images.** Write in-site links root-relative (`[posts](/posts/)`): the `/sidechain/` prefix is added at
build time for markdown links and images, **not** for raw HTML `<a>`/`<img>`, which must spell out `/sidechain/…`.

## What is generated

Never hand-edited; each generator's `--check` reports drift without writing.

- **`data/submissions.json` and the README's table**: `scripts/standings.py`, from our submission records and the
  leaderboard snapshots (`python -m sidechain.eval.leaderboard`). An entry's rank is the one in the first snapshot
  that contains it. The models page's table (`{{</* board */>}}`) paints the file's `board`; the rules are in the
  script.
- **`assets/data/field.json`**: the same script; the latest whole-field snapshot: the number of teams, the score at
  rank 100 and Sidechain's row.
- **`data/coverage.json`** (not in the tree; the `corpus` shortcode reads it):
  `uv run python -m sidechain.data.coverage --out site/data/coverage.json`, from the ingest caches.
- **`static/data/controls/`**: `uv run python scripts/site_controls.py`, field by field, so nothing the control-cells
  page does not draw reaches it, and a string outside the page's vocabulary stops the build. With no data on disk
  the page says so and the build still passes.

The plot and the rows are drawn at build time, so the page reads with scripts off; `assets/standing.js` adds the
plot's readout.

## How it deploys

`.github/workflows/site.yml` builds `site/` on every push to `main` that touches it and publishes with GitHub Pages,
which serves `saberhq/sidechain` at `saberhq.com/sidechain` under the account's custom domain. The one-time
prerequisite is Pages enabled on this repo with *GitHub Actions* as the source, a setting the workflow token cannot
flip itself:

```bash
gh api -X POST repos/saberhq/sidechain/pages -f build_type=workflow
```

Dark mode follows saberhq.com's: the `saberhq-theme` key in `localStorage`, shared because the two share an origin.
