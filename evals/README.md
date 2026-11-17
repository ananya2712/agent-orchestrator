# evals

Dev-only live scenario suite. The grader does not run it.

These scenarios exercise `POST /chat`. They call the real OpenAI API, so they need
`OPENAI_API_KEY` in `.env`, and each full run costs a few cents.

| File | Purpose |
|---|---|
| `scenarios.py` | 92 scenarios: 7 services × 6 test types (happy path, disambiguation, missing info, out of scope, multi-step, injection), plus error paths (C7), overlapping capabilities (C8), prompts no tool matches (C9), conditional execution (C10), conflicting instructions (C11), result ranking (C12), metamorphic paraphrase groups (C13), and cross-service extras. |
| `run_scenarios.py` | Runner: reset → optional setup → `/chat` → state checks; prints a scorecard and a service × category table. |
| `results/` | Written on each run (git-ignored): `latest.json` (full responses) and `latest.log` (orchestrator log). |

Run from the repo root:

```bash
backend/.venv/bin/python -I evals/run_scenarios.py                   # all 92 scenarios
backend/.venv/bin/python -I evals/run_scenarios.py --category C6     # one test type
backend/.venv/bin/python -I evals/run_scenarios.py --service slack   # one service
backend/.venv/bin/python -I evals/run_scenarios.py --only linear.C2  # id prefix
backend/.venv/bin/python -I evals/run_scenarios.py -v                # print every reply
```
