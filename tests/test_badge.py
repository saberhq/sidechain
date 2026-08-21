"""The README badge reports the team's live rank, and says so honestly when it can't."""
import json

from sidechain.eval import badge as b


def _row(team, rank):
    return {"teamName": team, "modelName": "m", "rank": rank, "scoreAvg": 0.1}


def test_reports_rank_of_total_and_matches_case_insensitively():
    board = {"entries": [_row("Vivai", 1), _row("sidechain", 9)], "total": 95}
    out = b.badge(board, "Sidechain")
    assert out == {"schemaVersion": 1, "label": "live rank", "message": "#9 of 95", "color": "brightgreen"}


def test_best_row_wins_when_a_team_has_several():
    board = {"entries": [_row("Sidechain", 30), _row("Sidechain", 12)], "total": 95}
    assert b.badge(board, "Sidechain")["message"] == "#12 of 95"
    assert b.badge(board, "Sidechain")["color"] == "blue"


def test_absent_from_embedded_rows_means_below_them_not_unranked():
    board = {"entries": [_row("Vivai", 1)], "total": 95}
    assert b.badge(board, "Sidechain")["message"] == "below #1 of 95"


def test_absent_from_a_fully_embedded_board_is_unranked():
    board = {"entries": [_row("Vivai", 1)], "total": 1}
    out = b.badge(board, "Sidechain")
    assert out["message"] == "unranked" and out["color"] == "lightgrey"


def test_cli_reads_a_snapshot_and_writes_shields_json(tmp_path, capsys):
    snap = tmp_path / "lb.json"
    snap.write_text(json.dumps({"live": {"entries": [_row("Sidechain", 2)], "total": 41}}))
    out = tmp_path / "badges" / "leaderboard.json"
    assert b.main(["--team", "Sidechain", "--snapshot", str(snap), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["message"] == "#2 of 41"
    assert capsys.readouterr().out.strip() == "#2 of 41"


def test_cli_fails_loudly_on_a_missing_board(tmp_path):
    snap = tmp_path / "lb.json"
    snap.write_text(json.dumps({"live": {"entries": [], "total": 0}}))
    assert b.main(["--team", "Sidechain", "--board", "final", "--snapshot", str(snap)]) == 1
