Loom Link: https://www.loom.com/share/5eb53f0bd88745fc9c5cff351c1d1d1c

Goal -> Build a multi-tool harness supporting 7 services. I used a test-driven approach to build the router, tool adapter and orchestrator. 

## TOOL ROUTING
Giving every tool to the model would consume an enormous number of tokens per call. Instead, I implemented a three-step architecture:
1. Router: A structured-output LLM selects <=12 tools from a compacted catalog. A schema ensures that selected names match the registry exactly. Router considers 6 most recent messages. I designed the router's prompt to maximize OpenAI's automatic prompt caching.
   
2. Deterministic tool expansion: Code adds necessary additional tools(eg. search tools), and tools needed for cross-service queries.
   
3. Agent: Receives only the selected schemas and can request additional tools through request_tools when necessary.
   
I explored three alternatives: passing all tool schemas directly to the model, embedding-based retrieval, and service-level routing. However, each had limitations, including high token costs, the risk of missing required tools, and large tool catalogs within individual services (GitHub alone has 86 tools).

| Metric | Value |
|---|---|
| Router latency | p50 1.0 s, max 3.4 s; median **26%** of a 4.3 s turn |
| Router prompt cache hit | **92%** of router input tokens (eg. 4,352 of 4,557) |
| Input tokens per turn | about 10.3K (router 4.6K + agent 5.7K over all steps) |
| Versus all schemas every step | ≥ 144K per turn (72K × median 2 steps), about **14×** more |
| Metamorphic routing consistency | Router gave 5/5 paraphrases of "open PRs" the **same** tools |

## ORCHESTRATION LOOP
A response contains either a final answer or tool calls. The harness validates each tool and its arguments, executes the registered function, and returns a structured result with the original call ID. It continues until the model answers or reaches a configured limit.

The harness bounds rounds and tool calls to prevent infinite loops. Unknown tools, invalid arguments, and execution failures become errors so the model can recover when possible.

I execute calls sequentially for deterministic ordering.

## What I would do with 8 more hours?
Add deterministic checks before executing conditional write operations. Currently, the agent relies on the LLM to evaluate conditions, which can lead to incorrect decisions.
I would also integrate the test suite into CI and run repeated evaluations to measure routing consistency, latency, and regression rates. I may also introduce an LLM as a judge environment to review how the harness performs.

## Trade-offs and testing
I followed a test-driven development approach, using failures to refine the architecture. For example, incorrect tool names led to enum-constrained routing, while missing lookup tools motivated deterministic companion-tool expansion.

For orchestration, I chose sequential tool execution over parallel execution. Since the mocked tools execute in under a millisecond, parallelism offered limited benefits while introducing potential race conditions and nondeterministic logs.

These were the cases looked at - 
C1 · Happy path
C2 · Disambiguation
C3 · Missing info
C4 · Out of scope
C5 · Multi-step
C6 · Injection
C7 · Error path
C8 · Overlapping services
C9 · No tool match
C10 · Conditional
C11 · Conflicting instructions
C12 · Ranking
C13 · Paraphrases
