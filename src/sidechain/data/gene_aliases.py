"""One source of truth for retired HGNC gene symbols seen in our corpora.

The authority is **HGNC** (``rest.genenames.org``): a pair is listed here only when exactly one
APPROVED current symbol came back from both ``fetch/prev_symbol/<OLD>`` and
``fetch/alias_symbol/<OLD>``, and ``fetch/symbol/<NEW>`` lists ``<OLD>`` among its
``prev_symbol``. QARS is the one deliberate exception to "exactly one" (see below). The nine pairs
carried since 2026-09-03 were re-checked under the same rule and carry the same records.

Verified 2026-09-15. Evidence, one record per pair, all 48 -- HGNC id, Ensembl id, the date the
symbol changed, and the ``prev_symbol`` list of the current symbol that confirms the rule::

    ~/data/sidechain/runs/geometry_gate/t78_alias_20260915/alias_verified.json

Why it exists: ESM2's ``protein_embeddings.pt`` is keyed on CURRENT symbols (all 48 values are
keys there; none of the 48 old symbols is), while several corpora -- and the 18,533-symbol 2026
gene axis -- still spell targets the old way (38 of these 48 old symbols are on that axis, and
none of the current ones). Without the bridge those targets resolve to nothing --
``scripts/build_pert_features.py`` would hand ``cell_load`` a label it zero-fills, and
``scripts/esm2_geometry_gate.py`` would silently drop the row.

**Direction matters.** This table maps a corpus (or 2026-axis) spelling ONTO the embedding table.
It is NOT a map onto the 2026 axis: applying it that way turns hits into misses, because the axis
speaks the old dialect. No pair has both spellings on the axis, so nothing here can merge two
distinct 2026 genes -- ``tests/test_geometry_gate.py`` pins that, along with no self-map, no
chain, and no two old symbols sharing a current one.

Two pairs worth keeping notes on:

  QARS is a retired symbol on TWO genes -- QARS1 (HGNC:9751) and EPRS1 (HGNC:3418). The mapping
  below is the correct one, pinned deliberately rather than resolved by scanning an alias field,
  because a generic prev_symbol lookup returns both. EPRS -> EPRS1 is listed separately.

  CCDC130 -> YJU2B (HGNC:28118, ENSG00000104957) was renamed 2021-03-26; "YJU2 splicing factor
  homolog B" is the same spliceosome NTC protein under a new name, not a different gene.

Consumers import it as ``ALIAS``; keys are the old symbol, values the current one, alphabetical.
"""

from __future__ import annotations

#: Old (retired) HGNC symbol -> current HGNC symbol. 48 pairs, verified 2026-09-15.
RETIRED_SYMBOLS: dict[str, str] = {
    "AARS": "AARS1",
    "ATP5MD": "ATP5MK",
    "C12orf45": "NOPCHAP1",
    "C5orf30": "MACIR",
    "C9orf16": "BBLN",
    "CARS": "CARS1",
    "CCDC130": "YJU2B",
    "CCDC84": "CENATAC",
    "CD3EAP": "POLR1G",
    "DARS": "DARS1",
    "EPRS": "EPRS1",
    "FAM207A": "SLX9",
    "FGFR1OP": "CEP43",
    "GARS": "GARS1",
    "H2AFX": "H2AX",
    "H2AFZ": "H2AZ1",
    "H3F3A": "H3-3A",
    "HARS": "HARS1",
    "HIST1H2AB": "H2AC4",
    "HIST1H2AE": "H2AC8",
    "HIST1H2AI": "H2AC13",
    "HIST1H2BB": "H2BC3",
    "HIST1H2BC": "H2BC4",
    "HIST1H2BE": "H2BC6",
    "HIST1H2BJ": "H2BC11",
    "HIST1H2BL": "H2BC13",
    "HIST1H2BM": "H2BC14",
    "HIST1H2BN": "H2BC15",
    "HIST2H2AA3": "H2AC18",
    "HIST2H2AC": "H2AC20",
    "HIST2H2BE": "H2BC21",
    "HIST2H2BF": "H2BC18",
    "HIST2H3A": "H3C15",
    "HIST2H3D": "H3C13",
    "IARS": "IARS1",
    "KARS": "KARS1",
    "LARS": "LARS1",
    "MARS": "MARS1",
    "NARS": "NARS1",
    "QARS": "QARS1",
    "RARS": "RARS1",
    "TARS": "TARS1",
    "TWISTNB": "POLR1F",
    "VARS": "VARS1",
    "WARS": "WARS1",
    "WDR92": "DNAAF10",
    "YARS": "YARS1",
    "ZNRD1": "POLR1H",
}
