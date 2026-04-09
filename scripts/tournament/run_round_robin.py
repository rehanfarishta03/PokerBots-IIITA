#!/usr/bin/env python3
"""Run a static round-robin tournament for qualified submissions."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from tournament_utils import discover_submission_bots, run_isolated_match, validate_submission


@dataclass
class LeaderboardEntry:
    bot_id: str
    total_bankroll: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    matches: int = 0


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run round robin among qualified bots")
    parser.add_argument("--repo-root", default=".", help="Path to repository root")
    parser.add_argument("--submissions-root", default="submission", help="Submission root directory")
    parser.add_argument("--baseline-path", default="python_skeleton", help="Baseline bot for qualification")
    parser.add_argument(
        "--qualification-rounds",
        type=int,
        default=300,
        help="Hands against baseline for qualification",
    )
    parser.add_argument(
        "--qualification-threshold",
        type=int,
        default=1,
        help="Minimum submission bankroll to qualify",
    )
    parser.add_argument("--match-rounds", type=int, default=600, help="Hands per finals match")
    parser.add_argument("--output-dir", default="tournament_results", help="Directory for results/logs")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    submissions_root = (repo_root / args.submissions_root).resolve()
    baseline = (repo_root / args.baseline_path).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    logs_dir = output_dir / "logs"

    bots = discover_submission_bots(submissions_root)

    qualification_rows: list[dict] = []
    qualified: list[tuple[str, Path]] = []

    if not baseline.is_dir():
        print(f"Baseline path not found: {baseline}")
        return 2

    for bot in bots:
        validation = validate_submission(bot, repo_root)
        row = {
            "bot_id": bot.bot_id,
            "path": bot.path.as_posix(),
            "validation_ok": validation.ok,
            "validation_errors": " | ".join(validation.errors),
            "qualified": False,
            "baseline_bankroll": 0,
            "submission_bankroll": 0,
            "notes": "",
        }

        if not validation.ok:
            qualification_rows.append(row)
            continue

        bot_abs = (repo_root / bot.path).resolve()
        q_match = run_isolated_match(
            repo_root=repo_root,
            player_1_source=baseline,
            player_2_source=bot_abs,
            output_dir=logs_dir / "qualification",
            player_1_name="BASELINE",
            player_2_name=bot.bot_id.replace("/", "_"),
            num_rounds=args.qualification_rounds,
            timeout_seconds=1200,
        )

        row["baseline_bankroll"] = q_match.player_1_bankroll
        row["submission_bankroll"] = q_match.player_2_bankroll

        if not q_match.ok:
            row["notes"] = q_match.failure_reason or "qualification match failed"
            qualification_rows.append(row)
            continue

        if q_match.player_2_bankroll >= args.qualification_threshold:
            row["qualified"] = True
            qualified.append((bot.bot_id, bot_abs))
        else:
            row["notes"] = (
                f"below threshold {args.qualification_threshold}: {q_match.player_2_bankroll}"
            )

        qualification_rows.append(row)

    _write_csv(
        output_dir / "qualification.csv",
        qualification_rows,
        [
            "bot_id",
            "path",
            "validation_ok",
            "validation_errors",
            "qualified",
            "baseline_bankroll",
            "submission_bankroll",
            "notes",
        ],
    )

    leaderboard: dict[str, LeaderboardEntry] = {
        bot_id: LeaderboardEntry(bot_id=bot_id) for bot_id, _ in qualified
    }
    match_rows: list[dict] = []

    for (bot_a_id, bot_a_path), (bot_b_id, bot_b_path) in combinations(qualified, 2):
        finals_match = run_isolated_match(
            repo_root=repo_root,
            player_1_source=bot_a_path,
            player_2_source=bot_b_path,
            output_dir=logs_dir / "finals",
            player_1_name=bot_a_id.replace("/", "_"),
            player_2_name=bot_b_id.replace("/", "_"),
            num_rounds=args.match_rounds,
            timeout_seconds=1800,
        )

        row = {
            "bot_a": bot_a_id,
            "bot_b": bot_b_id,
            "match_ok": finals_match.ok,
            "bot_a_bankroll": finals_match.player_1_bankroll,
            "bot_b_bankroll": finals_match.player_2_bankroll,
            "winner": "",
            "log_path": finals_match.log_path.as_posix() if finals_match.log_path else "",
            "notes": finals_match.failure_reason or "",
        }

        if finals_match.ok:
            a_entry = leaderboard[bot_a_id]
            b_entry = leaderboard[bot_b_id]
            a_entry.total_bankroll += finals_match.player_1_bankroll
            b_entry.total_bankroll += finals_match.player_2_bankroll
            a_entry.matches += 1
            b_entry.matches += 1

            if finals_match.player_1_bankroll > finals_match.player_2_bankroll:
                row["winner"] = bot_a_id
                a_entry.wins += 1
                b_entry.losses += 1
            elif finals_match.player_2_bankroll > finals_match.player_1_bankroll:
                row["winner"] = bot_b_id
                b_entry.wins += 1
                a_entry.losses += 1
            else:
                row["winner"] = "DRAW"
                a_entry.draws += 1
                b_entry.draws += 1

        match_rows.append(row)

    _write_csv(
        output_dir / "matches.csv",
        match_rows,
        [
            "bot_a",
            "bot_b",
            "match_ok",
            "bot_a_bankroll",
            "bot_b_bankroll",
            "winner",
            "log_path",
            "notes",
        ],
    )

    leaderboard_rows = [
        {
            "bot_id": entry.bot_id,
            "total_bankroll": entry.total_bankroll,
            "wins": entry.wins,
            "losses": entry.losses,
            "draws": entry.draws,
            "matches": entry.matches,
        }
        for entry in sorted(
            leaderboard.values(),
            key=lambda e: (e.total_bankroll, e.wins, -e.losses),
            reverse=True,
        )
    ]

    _write_csv(
        output_dir / "results.csv",
        leaderboard_rows,
        ["bot_id", "total_bankroll", "wins", "losses", "draws", "matches"],
    )

    summary = {
        "total_submissions_discovered": len(bots),
        "qualified_count": len(qualified),
        "qualification_rounds": args.qualification_rounds,
        "qualification_threshold": args.qualification_threshold,
        "finals_match_rounds": args.match_rounds,
        "outputs": {
            "qualification_csv": str((output_dir / "qualification.csv").relative_to(repo_root)),
            "matches_csv": str((output_dir / "matches.csv").relative_to(repo_root)),
            "results_csv": str((output_dir / "results.csv").relative_to(repo_root)),
            "logs_dir": str((output_dir / "logs").relative_to(repo_root)),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))

    if len(qualified) < 2:
        print("Warning: fewer than 2 qualified bots; finals round robin did not run full pairings.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
