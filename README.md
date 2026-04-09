# MIT Pokerbots Engine
MIT Pokerbots engine and skeleton bots in Python, Java, and C++.

This is the reference implementation of the engine for playing vanilla Texas hold'em. **Do not update** this repo to implement the yearly game variant! Instead, create a new repo within this organization called mitpokerbots/engine-yyyy.

Improvements which help the engine generalize to new variants, run faster, handle bot subprocesses more safely, etc. should be incorporated into this repo.

## Dependencies
 - python>=3.5
 - cython (pip install cython)
 - eval7 (pip install eval7)
 - Java>=8 for java_skeleton
 - C++17 for cpp_skeleton
 - boost for cpp_skeleton (`sudo apt install libboost-all-dev`)

## Linting
Use pylint.

## Automated Tournament Pipeline

This repository includes a two-stage tournament flow:

1. PR-time qualification in GitHub Actions.
2. Post-deadline static round robin run manually.

### Phase 1: PR Qualification (Gatekeeper)

Workflow file: `.github/workflows/submission-qualification.yml`

Submission paths must follow one of:

- `submission/<roll_no>/python_bot/`
- `submission/<roll_no>/cpp_bot/`

Each changed submission is validated for required files and then matched against a baseline bot in an isolated temporary sandbox.

Default qualification parameters:

- baseline bot: `python_skeleton`
- rounds per qualification match: `300`
- minimum submission bankroll to pass: `>= 1`

The workflow posts a sticky PR comment with:

- per-submission validation outcome
- bankroll results vs baseline
- pass/fail verdict

It also uploads `.qualification/` as an artifact with logs and JSON summaries.

### Phase 2: Static Round Robin (Finals)

Run manually after the deadline:

```bash
python scripts/tournament/run_round_robin.py \
	--repo-root . \
	--submissions-root submission \
	--baseline-path python_skeleton \
	--qualification-rounds 300 \
	--qualification-threshold 1 \
	--match-rounds 600 \
	--output-dir tournament_results
```

This script:

1. Discovers all bots under `submission/<roll_no>/(python_bot|cpp_bot)`.
2. Re-validates and re-qualifies bots against baseline.
3. Runs every unique finals pair among qualified bots.
4. Tracks total bankroll and W/L/D statistics.
5. Writes outputs:
	 - `tournament_results/qualification.csv`
	 - `tournament_results/matches.csv`
	 - `tournament_results/results.csv`
	 - detailed logs in `tournament_results/logs/`

### Security and Isolation Notes

- Untrusted bot code is run inside the GitHub Actions runner (CI) or the local execution host.
- Every match runs in a fresh temporary sandbox directory that contains only:
	- `engine.py`
	- an auto-generated `config.py`
	- copied bot directories for the two participants
- Engine game clock and player timeouts remain enforced via `config.py`.
