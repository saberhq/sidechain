"""Retired HGNC gene symbols mapped to their current ones (authority: HGNC, rest.genenames.org).

    from sidechain.data.gene_aliases import RETIRED_SYMBOLS
    current = RETIRED_SYMBOLS.get(symbol, symbol)

Direction matters: it maps a corpus (or 2026-axis) spelling onto current-symbol tables, never onto the 2026 gene
axis, which spells these genes the old way. Keys are old symbols, values current ones, alphabetical. Reads and
writes nothing. Notes: private/code-notes/src/sidechain/data/gene_aliases.py.md
"""

from __future__ import annotations

#: Old (retired) HGNC symbol -> current HGNC symbol.
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
