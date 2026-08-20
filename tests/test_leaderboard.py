"""The leaderboard snapshot parses the RSC payload Arc embeds in the page.

Pins the contract against a minimal synthetic page so a change in the site's
payload shape fails here rather than silently yielding an empty board.
"""
import json

import pytest

from sidechain.eval import leaderboard as lb

ENTRY = {
    "id": "x", "teamName": "t", "modelName": "m", "rank": 1, "scoreAvg": 0.1234,
    "scorePds": 0.6, "scoreMse": 0.0, "scoreNmae": 0.1, "scoreFid": -0.04,
    "scoreReach": 0.09, "scoreJac": 0.007, "pdsCosine": 0.78,
}


def _page(payload: str) -> str:
    # A JS string literal inside a <script>: quotes escaped exactly as Next.js does.
    literal = json.dumps(payload)[1:-1]
    return (
        "<html><body><div>Loading...</div>"
        f'<script>self.__next_f.push([1,"{literal}"])</script>'
        "</body></html>"
    )


def test_parses_all_three_boards_and_totals():
    payload = (
        '13:["$","div",null,{"children":["$","$L19",null,{'
        f'"initialLiveLeaderboardEntries":[{json.dumps(ENTRY)}],'
        '"initialLiveLeaderboardNumEntries":6,'
        '"initialFinalLeaderboardEntries":[],"initialFinalLeaderboardNumEntries":0,'
        '"initialGeneralistEntries":[],"initialGeneralistNumEntries":0}]}]\n'
    )
    boards = lb.parse_boards(_page(payload))
    assert set(boards) == {"live", "final", "generalist"}
    assert boards["live"]["total"] == 6
    assert boards["live"]["entries"][0]["scoreAvg"] == pytest.approx(0.1234)
    assert boards["final"]["entries"] == []


def test_missing_payload_raises_rather_than_returning_empty():
    with pytest.raises(ValueError):
        lb.parse_boards("<html><body>Loading...</body></html>")


def test_table_has_one_row_per_entry_and_the_six_members():
    table = lb.format_table([ENTRY])
    assert "pds" in table and "jac" in table
    assert table.count("\n") == 2  # header, rule, one row
    assert "0.1234" in table
