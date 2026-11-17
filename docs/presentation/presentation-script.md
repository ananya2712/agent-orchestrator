# Presentation script — POST /chat orchestrator (Loom, ≤ 5 min)

Target: about 4:45. Spoken words: about 540 at a relaxed pace, plus about 40 s
of demo wait time. Lines in *italics* are actions, not words to say.

The full-length slide-only version (about 4.7 min, no demo) is in the deck's
speaker notes. Use this script for the Loom, because the brief asks for
"a couple of scenarios running live".

## Before you record

1. Terminal 1: `pnpm run dev` (server on http://127.0.0.1:8000). Keep it visible: the turn log shows routing and tool calls.
2. Terminal 2: `pnpm run cli -- reset`, then do one warm-up chat so the first demo is not slow. Run `pnpm run cli -- reset` again after it.
3. Make the terminal font large. Open the deck in slideshow mode.
4. Do a dry run of the four demo commands. Check that the replies match the "point out" lines below, and change those lines if they do not.

---

## Slide 1 — Cover · 0:00–0:20

Hi! This is my POST /chat orchestrator. The task: take a plain-English request
and get it done across 191 mock tools in seven services, with the right tools,
as few calls as possible, and a fast reply. My approach fits in one line: one
router call picks the tools, and one bounded loop runs them.

## Slide 2 — Architecture · 0:20–0:50

The code is split into small layers, and each one has a single job. solution.py
is the endpoint; it just wires things together. The router picks the tools, the
orchestrator runs the agent loop, and the executor runs one tool call safely.
Only the model adapter knows about OpenAI. The rule throughout: the model makes
the judgment calls, and code enforces the hard limits.

## Slide 3 — Data flow · 0:50–1:25

Sending all 191 tool schemas would cost about 72 thousand tokens on every call.
So the router reads a compact index instead, about 3.4 thousand tokens, and
picks up to 12 tools. Code then adds the lookups a write depends on, like
listing Slack channels before a post. Then the loop: the model asks for calls,
the executor runs them in order, and the results go back, until there's a final
answer. If any limit hits, one last call writes an honest answer.

## Slide 4 — Request path · 1:25–1:40

Here's that same turn, file by file: solution, config, router, and then the
orchestrator, which goes back and forth between the model adapter and the
executor until the model has an answer.

---

## Live demo · 1:40–3:25

*Switch to the terminal.*

Let's see it run. I'll reset the mock state first.

*`pnpm run cli -- reset`*

### Demo 1 — single service (about 25 s)

*`pnpm run cli -- chat "What conversations do I have in Slack?"`*

A simple one first. Point out: one tool call, `slack_list_conversations`, and
it's right there in the tool_calls log.

### Demo 2 — cross-service write (about 40 s)

*`pnpm run cli -- chat "Find the most recent email about the product launch and post a summary to the launch-war-room Slack channel."`*

Now a cross-service write. Point out: it searches Gmail first, looks up the
Slack channel, and only then posts. The calls run in order, and each one is in
the log. The channel lookup was there because code added it, not because the
model guessed an ID.

### Demo 3 — ambiguity (about 25 s)

*`pnpm run cli -- chat "Schedule a 30-minute meeting with everyone on the project next week."`*

This one is ambiguous: which project? Point out: there's more than one, so it
asks instead of guessing, and nothing was written.

### Demo 4 — error path (about 20 s)

*`pnpm run cli -- chat "Delete budget_2025.xlsx from my Drive."`*

And an error path. Point out: that file doesn't exist, so it says so. It
doesn't pretend it deleted anything.

*Switch back to the slides.*

---

## Slide 5 — Key design decisions · 3:25–3:50

A few decisions shaped this. Routing with one LLM call over a compact index,
because it's cheap and understands intent across services. A plain bounded
loop instead of a framework. Tool output is treated as data, never as an
instruction, and all seven planted injection attacks were ignored. And when a
target is ambiguous, the agent asks, as you just saw.

## Slide 6 — Code changes · 3:50–4:00

The deck has a one-line summary of what each file does, so I'll move on.

## Slide 7 — Tests · 4:00–4:25

I tested at two levels. A dry run with a scripted model needs no API, and all
43 tests pass. Then a live matrix of 92 scenarios that runs like the grader:
each of the seven services gets the same six test types, from happy paths to
prompt injection, plus cross-cutting tests like conditional requests and
paraphrases.

## Slide 8 — Errors and results · 4:25–4:55

Each round of tests found something real, and each got a small, targeted fix:
invalid router names, missing lookups, invented search syntax, and the agent
refusing simple knowledge questions. In the final full run, 88 of 92 pass, at
3.6 seconds median. Of the four misses, three are known limits I've documented,
and one was an over-strict check that's now fixed. Thanks for watching!
