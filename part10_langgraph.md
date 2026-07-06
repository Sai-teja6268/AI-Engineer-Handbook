# Part 10: LangGraph — Stateful AI Workflows

> *"LangChain taught you how to chain LLM calls in a line. LangGraph teaches you how to build AI systems that think in loops — systems that reason, act, observe, correct, and decide. It is the difference between a script and a brain."*

---

## Table of Contents

- [Chapter 1: LangGraph Fundamentals](#chapter-1-langgraph-fundamentals)
- [Chapter 2: State Management](#chapter-2-state-management)
- [Chapter 3: Conditional Edges and Routing](#chapter-3-conditional-edges-and-routing)
- [Chapter 4: ReAct Agent Pattern](#chapter-4-react-agent-pattern)
- [Chapter 5: Multi-Agent Systems](#chapter-5-multi-agent-systems)
- [Chapter 6: Human-in-the-Loop](#chapter-6-human-in-the-loop)
- [Chapter 7: Persistence and Checkpointing](#chapter-7-persistence-and-checkpointing)
- [Chapter 8: Production Patterns](#chapter-8-production-patterns)

---

# Chapter 1: LangGraph Fundamentals

---

## 1. Introduction

### What Is LangGraph?

**LangGraph** is a library for building stateful, multi-step AI workflows using a **directed graph** model. It extends LangChain by enabling:

- **Cycles**: Loops where an agent can retry, reflect, and correct itself
- **Persistent state**: A shared state object that accumulates information as it flows through nodes
- **Conditional routing**: Dynamic branching based on intermediate outputs
- **Human-in-the-loop**: Pause execution at any point for human review or input
- **Multi-agent coordination**: Multiple AI agents collaborating on a task

LangGraph uses **StateGraph** as its core abstraction — a computational graph where:
- **Nodes** are Python functions or LangChain Runnables
- **Edges** define how control flows between nodes
- **State** is a shared dictionary that flows through every node

### Why Does LangGraph Exist?

LCEL (LangChain's pipe operator) is excellent for **linear or parallel** pipelines. But production AI systems frequently need:
- An agent that loops until it gets a good answer
- A workflow that branches based on whether a document is relevant
- A system that asks a human for approval before taking an irreversible action
- Multiple specialized agents (researcher, writer, reviewer) coordinating on a task

These requirements cannot be expressed cleanly in LCEL. LangGraph was built specifically for these **stateful, cyclic, conditional** workflows.

---

## 2. Historical Motivation

### The Limitation of Linear Chains

In 2023, as engineers built more sophisticated AI applications, they consistently hit the ceiling of linear chain frameworks:

**Problem 1**: ReAct agents need to loop — call a tool, observe output, decide to call another tool or answer. LCEL's `AgentExecutor` handled this with an internal loop, but it was opaque, hard to debug, and difficult to extend with custom logic.

**Problem 2**: Multi-step workflows needed branching. "Is this document relevant? If yes, summarize it. If no, fetch another one." This required complex conditional logic that LCEL couldn't express cleanly.

**Problem 3**: Long-running workflows needed to be paused and resumed — for human approval, for batch processing, for system restarts. LCEL had no concept of persistence.

LangGraph (released early 2024) addressed all three: explicit graph structure with named nodes, conditional edges for branching, and checkpointers for persistence. Within months, it became the standard for building production AI agents and multi-agent systems.

---

## 3. Real-World Analogy

### The Control Flow Graph (CFG) of AI

Computer science students learn about **Control Flow Graphs** — directed graphs where nodes are code blocks and edges represent possible execution paths. An if-else becomes two edges from one node. A while loop creates a back edge (cycle).

LangGraph applies the same idea to AI workflows:
- **Node** = a step in your AI workflow (call the LLM, search the web, query the DB)
- **Edge** = possible transitions between steps
- **State** = the "registers" that carry values across steps
- **Conditional edge** = the `if` statement that chooses which step to go to next
- **Cycle** = the `while` loop that keeps the agent reasoning until it reaches an answer

You're not writing an AI script; you're designing an AI control flow graph.

---

## 4. Visual Mental Model

### LangGraph Core Concepts

```mermaid
flowchart TD
    subgraph "LangGraph StateGraph"
        START((START)) --> A["Node: agent\n(LLM decides what to do)"]
        A --> B{Conditional Edge:\nDoes agent want to use a tool?}
        B -->|Yes: tool_call| C["Node: tools\n(Execute the tool)"]
        B -->|No: final answer| END((END))
        C --> A
    end

    subgraph "State (flows through every node)"
        S["AgentState = TypedDict
        messages: Annotated[list, add_messages]
        retrieved_docs: list
        final_answer: str"]
    end

    A -.->|reads/writes| S
    C -.->|reads/writes| S
```

### Node, Edge, State — The Three Primitives

```mermaid
graph LR
    subgraph "Three Primitives"
        A["🔷 Node\nPython function\nInput: State → Output: State update"]
        B["→ Edge\nDirect connection\nAlways goes A→B"]
        C["⟶ Conditional Edge\nFunction decides\nwhich node to go to"]
        D["📦 State\nShared TypedDict\nAll nodes read from and write to it"]
    end

    A --> B --> A2["Another Node"]
    A --> C --> A3["Node A or Node B"]
    D -.->|flows through| A
    D -.->|flows through| A2
    D -.->|flows through| A3
```

---

## 5. Internal Working

### The StateGraph Lifecycle

```python
from langgraph.graph import StateGraph, START, END

# 1. Define State (shared across all nodes)
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class MyState(TypedDict):
    messages: Annotated[list, add_messages]  # Append-only list
    step_count: int                           # Simple value

# 2. Create the graph
graph = StateGraph(MyState)

# 3. Add nodes (functions that take state, return state update)
def my_node(state: MyState) -> dict:
    # Process state
    return {"step_count": state["step_count"] + 1}

graph.add_node("my_node", my_node)

# 4. Add edges (define flow)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)

# 5. Compile (creates a runnable)
app = graph.compile()

# 6. Invoke
result = app.invoke({"messages": [], "step_count": 0})
```

### How State Updates Work

Each node returns a **partial state update** — a dict containing only the keys it wants to update. LangGraph **merges** these updates into the current state.

**Key insight**: The `Annotated[list, add_messages]` type annotation tells LangGraph to use the `add_messages` reducer instead of replacing the list. This enables append behavior: every node that returns `{"messages": [new_message]}` **appends** to the messages list rather than replacing it.

```python
# Without Annotated: REPLACES the list
messages: list  # node returns {"messages": [x]} → messages = [x]

# With Annotated + add_messages: APPENDS to the list
messages: Annotated[list, add_messages]  # node returns {"messages": [x]} → messages = [...prev, x]
```

This is how conversation history accumulates naturally in a graph-based agent.

---

## 6. Mathematical Intuition

### Graph Theory Foundation

A LangGraph StateGraph is formally a **directed graph** $G = (V, E)$ where:
- $V$ = set of nodes (including `START` and `END`)
- $E \subseteq V \times V$ = set of directed edges
- $S$ = shared state space
- $f_v: S \rightarrow \Delta S$ = each node is a function mapping state to a state update

**Execution semantics**: Starting at `START`, the graph executes nodes in topological order (for DAGs) or in cycles (for cyclic graphs with conditional edges). Each node receives the current accumulated state and returns a partial update that is merged into the state.

**Reducers**: For each key in the state, a **reducer** function defines how updates are merged:
- Default: replace (last write wins)
- `add_messages`: append to list
- Custom: any commutative/associative function (for parallel nodes)

---

## 7. Implementation

### Complete LangGraph Hello World to Production

```python
"""
LangGraph: from Hello World to a full production graph.
pip install langgraph langchain-openai
"""

import asyncio
import operator
from typing import TypedDict, Annotated, List, Literal, Optional, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ─── 1. Minimal Graph: Hello World ───────────────────────────────────────────

def hello_langgraph():
    """Simplest possible LangGraph: one node that calls an LLM."""

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    llm = ChatOpenAI(model="gpt-4o-mini")

    def chatbot_node(state: State):
        """Single node: call LLM with current message history."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}  # Appends to message history

    graph = StateGraph(State)
    graph.add_node("chatbot", chatbot_node)
    graph.add_edge(START, "chatbot")
    graph.add_edge("chatbot", END)

    app = graph.compile()

    # Invoke
    result = app.invoke({
        "messages": [HumanMessage(content="What is LangGraph?")]
    })
    print(result["messages"][-1].content)
    return app


# ─── 2. Graph with Tools (ReAct-style) ───────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city. Use when asked about weather."""
    # Mock implementation
    return f"The weather in {city} is 22°C, sunny with light breeze."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. e.g., '15 * 8.5'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def build_tool_graph():
    """Graph with LLM agent node + tool execution node."""

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]

    tools = [get_weather, calculate]
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(tools)

    # Node 1: Agent decides whether to use a tool or answer directly
    def agent_node(state: AgentState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    # Node 2: Execute tool calls (LangGraph's built-in ToolNode)
    tool_node = ToolNode(tools)

    # Routing function: should we call tools or stop?
    def should_use_tools(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        # If the last message has tool calls, go to tools
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "end"

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")  # After tools → back to agent (the cycle!)

    app = graph.compile()
    return app


# ─── 3. Visualizing the Graph ─────────────────────────────────────────────────

def visualize_graph(app):
    """
    LangGraph graphs are self-documenting.
    Use .get_graph().draw_mermaid() to get a Mermaid diagram.
    """
    mermaid_diagram = app.get_graph().draw_mermaid()
    print(mermaid_diagram)
    # Output is valid Mermaid syntax you can paste into mermaid.live


# ─── 4. Streaming Graph Execution ─────────────────────────────────────────────

async def stream_graph_execution():
    """
    Stream graph execution: see each node's output as it runs.
    Modes: "values" (full state after each node), "updates" (just the changes)
    """
    app = build_tool_graph()

    # Stream: see state after every node execution
    async for event in app.astream(
        {"messages": [HumanMessage(content="What's the weather in Tokyo?")]},
        stream_mode="updates",  # Show only what each node changes
    ):
        for node_name, state_update in event.items():
            print(f"\n[Node: {node_name}]")
            for msg in state_update.get("messages", []):
                print(f"  {type(msg).__name__}: {str(msg.content)[:100]}")


# ─── 5. Graph Introspection ───────────────────────────────────────────────────

def inspect_graph(app):
    """
    Inspect graph structure programmatically.
    Useful for testing and documentation.
    """
    graph_obj = app.get_graph()
    
    print("Nodes:", list(graph_obj.nodes.keys()))
    print("Edges:", [(e.source, e.target) for e in graph_obj.edges])

    # Draw ASCII representation
    print(graph_obj.draw_ascii())
```

---

## 8. Production Architecture

```mermaid
flowchart TD
    subgraph "LangGraph Production Stack"
        A[FastAPI Server] --> B[LangGraph App\ncompiled StateGraph]
        B --> C[Checkpointer\nPostgresSaver]
        C --> D[(PostgreSQL\nThread checkpoints)]

        B --> N1[Node: Planner\nDecomposes task]
        B --> N2[Node: Researcher\nSearches knowledge]
        B --> N3[Node: Writer\nGenerates content]
        B --> N4[Node: Reviewer\nChecks quality]

        B --> E[LangSmith\nTrace every node]
    end
```

---

## 9. Tradeoffs

| Feature | LangGraph | LCEL Chains | AgentExecutor |
|---|---|---|---|
| Cycles / loops | ✅ Native | ❌ No | ✅ Internal |
| State management | ✅ TypedDict | ❌ Manual | ❌ Opaque |
| Conditional branching | ✅ Explicit | ❌ Limited | ❌ None |
| Human-in-the-loop | ✅ interrupt_before/after | ❌ No | ❌ No |
| Persistence | ✅ Checkpointers | ❌ No | ❌ No |
| Debuggability | ✅ Node-by-node | Medium | ❌ Black box |
| Learning curve | High | Low | Low |

---

## 10. Common Mistakes

❌ **Forgetting `Annotated[list, add_messages]` for message history**: Using `messages: list` instead causes each node to **replace** the entire message list rather than appending. Your agent will "forget" all previous messages.

❌ **Adding edges to END without conditional logic**: If a node might want to loop back OR finish, you need a conditional edge. A direct edge to END means the node always terminates.

❌ **Not compiling before invoking**: `graph.compile()` is required. The graph object itself is not runnable.

❌ **Mutating state directly inside nodes**: Never do `state["messages"].append(x)`. Return a dict update instead: `return {"messages": [x]}`. LangGraph handles the merge.

---

## 11. Interview Preparation

**Junior**: "LangGraph uses a graph model to build AI workflows. You define nodes (Python functions that process state) and edges (connections between nodes). The shared state flows through every node. Unlike LangChain's linear chains, LangGraph supports cycles — an agent can loop back and call itself multiple times."

**Mid-level**: "LangGraph's three core concepts: (1) State — a TypedDict shared across all nodes, with reducers controlling how updates are merged (add_messages appends, default replaces); (2) Nodes — functions that take State and return a partial state update; (3) Conditional edges — functions that decide which node to go to next based on current state. The cycle `agent → tools → agent` is the ReAct loop: the agent calls a tool, gets the result, and decides whether to call another tool or give the final answer."

**Senior**: "I choose LangGraph over LCEL whenever the workflow needs: (1) Cycles — agent reasoning loops; (2) Conditional routing — branch based on intermediate outputs; (3) State persistence — resume workflows after interruption; (4) Human-in-the-loop — interrupt before irreversible actions; (5) Multi-agent coordination — multiple specialized agents collaborating. The graph structure forces explicit thinking about state transitions, which makes complex workflows debuggable and testable in ways that are impossible with opaque agent executors."

---

## 12. Follow-up Questions

**Q1: What is the difference between `add_edge` and `add_conditional_edges`?**
> `add_edge(a, b)`: After node `a`, always go to node `b`. `add_conditional_edges(a, condition_fn, {output: node})`: After node `a`, call `condition_fn(state)` to decide where to go. The dict maps condition_fn's return values to node names.

**Q2: What are reducers and why do they matter?**
> A reducer is a function that merges a node's state update into the current state. Default reducer: replacement (new value overwrites old). `add_messages`: appends new messages to the existing list. Custom reducers: any associative merge function. They matter because in parallel node execution, multiple nodes may write to the same state key simultaneously — the reducer defines how those writes are combined.

**Q3: What is `interrupt_before` and `interrupt_after`?**
> Both are arguments to `graph.compile()`. `interrupt_before=["node_name"]` pauses execution BEFORE the named node runs — the human can review the state and decide to continue or modify it. `interrupt_after=["node_name"]` pauses AFTER — useful for reviewing a node's output before proceeding. This is the foundation of human-in-the-loop workflows.

---

## 13. Revision Sheet

- **StateGraph** = directed graph with named nodes, edges, and shared state
- **State** = TypedDict with reducers; `Annotated[list, add_messages]` for append behavior
- **Node** = `fn(state: State) → dict` (returns partial state update, never mutates)
- **Edge** = `add_edge(from, to)` (unconditional) or `add_conditional_edges(from, fn, map)` (conditional)
- **Cycle** = back-edge from "tools" to "agent" creates the ReAct loop
- **compile()** = required before invoking; returns a Runnable
- **stream_mode** = "values" (full state) or "updates" (delta per node)

---

---

# Chapter 2: State Management

---

## 1. Introduction

### What Is State in LangGraph?

**State** is the memory of your graph — the data structure that flows through every node and accumulates information as the workflow progresses. It is the single most important design decision in any LangGraph application.

A well-designed state:
- Contains exactly the data each node needs
- Uses appropriate reducers so parallel nodes don't conflict
- Is serializable (for checkpointing and persistence)
- Documents the purpose of each field

Poor state design is the root cause of most LangGraph bugs: state fields being overwritten unexpectedly, missing data when a node needs it, or state growing unbounded.

---

## 2. Visual Mental Model

```mermaid
graph TD
    subgraph "State Design Patterns"
        A["Simple State\nFor linear graphs"] --> S1["class State(TypedDict):\n  messages: Annotated[list, add_messages]\n  step: int"]

        B["Agent State\nFor tool-using agents"] --> S2["class AgentState(TypedDict):\n  messages: Annotated[list, add_messages]\n  next_action: str\n  tool_results: list\n  final_answer: Optional[str]"]

        C["Multi-Agent State\nFor complex systems"] --> S3["class TeamState(TypedDict):\n  task: str\n  research_notes: Annotated[list, operator.add]\n  drafts: list\n  approved: bool\n  agent_scratchpads: dict"]
    end
```

---

## 3. Implementation

### State Design Patterns

```python
"""
LangGraph State: design patterns for every complexity level.
"""

import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any, Literal
from langchain_core.messages import BaseMessage, AnyMessage
from langgraph.graph.message import add_messages


# ─── Pattern 1: Minimal Chatbot State ────────────────────────────────────────

class ChatState(TypedDict):
    """Minimal state for a simple chatbot."""
    messages: Annotated[List[AnyMessage], add_messages]


# ─── Pattern 2: RAG Agent State ──────────────────────────────────────────────

class RAGState(TypedDict):
    """State for a RAG agent that retrieves and generates."""
    messages: Annotated[List[AnyMessage], add_messages]
    
    # Query processing
    original_query: str
    rephrased_query: Optional[str]
    query_type: Literal["factual", "conversational", "analytical", "unknown"]
    
    # Retrieval
    retrieved_docs: List[Dict]     # List of retrieved document dicts
    reranked_docs: List[Dict]      # After reranking
    retrieval_quality: Literal["good", "poor", "unknown"]
    
    # Generation
    generation_attempts: int       # How many times we've tried to generate
    final_answer: Optional[str]
    answer_is_faithful: Optional[bool]
    citations: List[str]


# ─── Pattern 3: Multi-Agent Collaboration State ───────────────────────────────

class ResearchTeamState(TypedDict):
    """State for a multi-agent research + writing team."""
    task: str                           # The original task
    
    # Research phase
    search_queries: List[str]           # Generated search queries
    research_notes: Annotated[          # Each agent appends their notes
        List[str], operator.add
    ]
    sources: Annotated[List[str], operator.add]  # Accumulated sources
    
    # Writing phase
    outline: Optional[str]
    draft: Optional[str]
    revision_notes: List[str]
    revision_count: int
    
    # Review phase
    review_score: Optional[int]   # 1-10
    review_feedback: Optional[str]
    approved: bool
    
    # Control flow
    next_agent: Optional[str]
    iteration: int


# ─── Pattern 4: Custom Reducer ────────────────────────────────────────────────

def merge_unique(existing: List, update: List) -> List:
    """
    Custom reducer: merge lists keeping only unique items.
    Useful for accumulating retrieved doc IDs across parallel retrievals.
    """
    existing_set = {str(item) for item in existing}
    result = list(existing)
    for item in update:
        if str(item) not in existing_set:
            result.append(item)
            existing_set.add(str(item))
    return result


class DeduplicatedState(TypedDict):
    """State with deduplicated document accumulation."""
    messages: Annotated[List[AnyMessage], add_messages]
    doc_ids: Annotated[List[str], merge_unique]   # No duplicates


# ─── Pattern 5: Private State Fields (Langgraph 0.2+) ────────────────────────

from dataclasses import dataclass, field
from langgraph.graph import add_messages as _add_messages


@dataclass
class AgentStateWithPrivate:
    """
    Using dataclass for state (alternative to TypedDict).
    Allows default values and cleaner field definitions.
    """
    messages: List[AnyMessage] = field(default_factory=list)
    retrieved_docs: List[Dict] = field(default_factory=list)
    step_count: int = 0
    _internal_scratchpad: str = ""   # Private: not exposed in API response


# ─── State Validation ─────────────────────────────────────────────────────────

def validate_state(state: RAGState) -> bool:
    """
    Validate state integrity before entering a critical node.
    Use as a guard at the start of expensive nodes.
    """
    assert "original_query" in state and state["original_query"], \
        "original_query must be set before retrieval"
    assert isinstance(state.get("retrieved_docs", []), list), \
        "retrieved_docs must be a list"
    assert state.get("generation_attempts", 0) < 5, \
        "Too many generation attempts — possible infinite loop"
    return True


# ─── State Update Patterns ────────────────────────────────────────────────────

def node_that_updates_correctly(state: RAGState) -> dict:
    """
    CORRECT: Return a partial dict with only changed fields.
    LangGraph merges this into the current state.
    """
    new_query = f"[Rephrased] {state['original_query']}"
    return {
        "rephrased_query": new_query,      # Update this field
        "query_type": "factual",            # Update this field
        # DON'T include fields you're not changing
    }


def node_that_updates_WRONG(state: RAGState) -> dict:
    """
    WRONG: Never copy and return the full state.
    This breaks reducers (overwrites accumulated values).
    """
    state_copy = dict(state)
    state_copy["rephrased_query"] = "new query"
    return state_copy  # BAD: This will REPLACE retrieved_docs, messages, etc.
```

---

## 4. Interview Preparation

**Mid-level**: "State in LangGraph is a TypedDict with reducer annotations. The most critical annotation is `Annotated[list, add_messages]` which appends to the message list instead of replacing it. For parallel nodes that all write to the same list, I use `Annotated[list, operator.add]`. Nodes return partial dicts — only the fields they change — and LangGraph merges them using the reducers."

**Senior**: "State design is the first thing I design in any LangGraph project. I think about: (1) What data does each node need to read? (2) What data does each node produce? (3) Where do nodes write to the same field simultaneously (requiring reducers)? (4) Is all state serializable for checkpointing? For multi-agent systems, I separate concerns into sub-states — the global state has minimal shared fields, and each agent has its own scratchpad. This prevents tight coupling between agents."

---

---

# Chapter 3: Conditional Edges and Routing

---

## 1. Introduction

### What Are Conditional Edges?

**Conditional edges** are the `if-else` of LangGraph. After a node executes, a routing function reads the current state and decides which node to execute next. This enables:
- **Branching**: Take different paths based on query type, retrieval quality, etc.
- **Looping**: Go back to a previous node for retry or refinement
- **Early stopping**: Skip remaining nodes when the answer is already satisfactory
- **Error routing**: Send failures to a fallback node

---

## 2. Visual Mental Model

### Conditional Routing Patterns

```mermaid
flowchart TD
    subgraph "Pattern 1: Binary Branch"
        A[Classify Query] --> CR1{relevant?}
        CR1 -->|yes| B[Retrieve + Generate]
        CR1 -->|no| C[Polite Decline]
        B --> E1((END))
        C --> E1
    end

    subgraph "Pattern 2: Multi-Way Router"
        D[Classify Query Type] --> CR2{type?}
        CR2 -->|factual| E[RAG Pipeline]
        CR2 -->|sql| F[Text-to-SQL]
        CR2 -->|chat| G[Direct LLM]
        CR2 -->|image| H[Vision Model]
        E --> E2((END))
        F --> E2
        G --> E2
        H --> E2
    end

    subgraph "Pattern 3: Quality Loop"
        I[Generate Answer] --> CR3{quality ok?}
        CR3 -->|good| J((END))
        CR3 -->|poor, tries < 3| I
        CR3 -->|poor, tries >= 3| K[Fallback Answer]
        K --> J
    end
```

---

## 3. Implementation

### Routing Patterns

```python
"""
LangGraph conditional edges: routing patterns for production systems.
"""

import asyncio
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import json


# ─── State ────────────────────────────────────────────────────────────────────

class RouterState(TypedDict):
    messages: Annotated[list, add_messages]
    query_type: Optional[str]          # Assigned by router node
    retrieval_quality: Optional[str]   # "good" or "poor"
    generation_count: int              # Number of generation attempts
    final_answer: Optional[str]


# ─── Nodes ────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def query_router_node(state: RouterState) -> dict:
    """
    Classify the query type to determine the processing pipeline.
    """
    query = state["messages"][-1].content

    class QueryClassification(BaseModel):
        type: Literal["factual", "conversational", "sql", "chitchat"]
        confidence: float = Field(ge=0.0, le=1.0)

    classifier = llm.with_structured_output(QueryClassification)

    result = classifier.invoke([
        {"role": "system", "content": "Classify this user query into one of: factual, conversational, sql, chitchat"},
        {"role": "user", "content": query},
    ])

    return {"query_type": result.type}


def rag_retrieve_node(state: RouterState) -> dict:
    """Retrieve documents from vector store."""
    # Mock retrieval
    query = state["messages"][-1].content
    # In production: actual retriever call
    mock_docs = [
        {"text": f"Relevant information about: {query}", "score": 0.92},
        {"text": f"Additional context for: {query}", "score": 0.87},
    ]
    quality = "good" if mock_docs[0]["score"] > 0.8 else "poor"
    return {"retrieval_quality": quality}


def rag_generate_node(state: RouterState) -> dict:
    """Generate answer from retrieved context."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on context only."),
        ("human", "{query}"),
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": state["messages"][-1].content})
    return {
        "final_answer": answer,
        "generation_count": state.get("generation_count", 0) + 1,
        "messages": [AIMessage(content=answer)],
    }


def sql_generate_node(state: RouterState) -> dict:
    """Generate SQL query and execute it."""
    # Mock SQL generation
    answer = "SELECT * FROM orders WHERE status = 'pending' LIMIT 10"
    return {"final_answer": answer, "messages": [AIMessage(content=answer)]}


def direct_chat_node(state: RouterState) -> dict:
    """Direct LLM response without retrieval."""
    response = llm.invoke(state["messages"])
    return {"messages": [response], "final_answer": response.content}


def chitchat_node(state: RouterState) -> dict:
    """Handle small talk."""
    response = "I'm here to help with factual questions and data analysis. How can I assist you today?"
    return {"messages": [AIMessage(content=response)], "final_answer": response}


def fallback_node(state: RouterState) -> dict:
    """Called after too many failed generation attempts."""
    response = "I wasn't able to find a reliable answer. Please rephrase your question or provide more context."
    return {"messages": [AIMessage(content=response)], "final_answer": response}


# ─── Routing Functions ────────────────────────────────────────────────────────

def route_by_query_type(state: RouterState) -> str:
    """Multi-way router: directs to appropriate processing pipeline."""
    query_type = state.get("query_type", "conversational")
    routing_map = {
        "factual":        "retrieve",
        "sql":            "sql_generate",
        "conversational": "direct_chat",
        "chitchat":       "chitchat",
    }
    return routing_map.get(query_type, "direct_chat")


def route_after_retrieval(state: RouterState) -> str:
    """Check retrieval quality before generating."""
    if state.get("retrieval_quality") == "good":
        return "generate"
    return "direct_chat"  # Fallback to direct LLM if retrieval is poor


def route_after_generation(state: RouterState) -> str:
    """
    Quality gate: check if the answer is good enough.
    Loop back to generate if not, but cap at 3 attempts.
    """
    if state.get("final_answer"):
        return END  # Good answer, done

    attempts = state.get("generation_count", 0)
    if attempts >= 3:
        return "fallback"  # Too many attempts, give up

    return "generate"  # Try again


# ─── Build Full Router Graph ──────────────────────────────────────────────────

def build_full_router_graph():
    """
    Complete multi-path routing graph.
    Demonstrates all conditional edge patterns.
    """
    graph = StateGraph(RouterState)

    # Add all nodes
    graph.add_node("router", query_router_node)
    graph.add_node("retrieve", rag_retrieve_node)
    graph.add_node("generate", rag_generate_node)
    graph.add_node("sql_generate", sql_generate_node)
    graph.add_node("direct_chat", direct_chat_node)
    graph.add_node("chitchat", chitchat_node)
    graph.add_node("fallback", fallback_node)

    # Entry point
    graph.add_edge(START, "router")

    # Multi-way routing after classification
    graph.add_conditional_edges(
        "router",
        route_by_query_type,
        {
            "retrieve":    "retrieve",
            "sql_generate": "sql_generate",
            "direct_chat": "direct_chat",
            "chitchat":    "chitchat",
        }
    )

    # Quality gate after retrieval
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {"generate": "generate", "direct_chat": "direct_chat"},
    )

    # Quality loop after generation
    graph.add_conditional_edges(
        "generate",
        route_after_generation,
        {"generate": "generate", "fallback": "fallback", END: END},
    )

    # Terminal nodes → END
    graph.add_edge("sql_generate", END)
    graph.add_edge("direct_chat", END)
    graph.add_edge("chitchat", END)
    graph.add_edge("fallback", END)

    app = graph.compile()
    return app


# ─── LangGraph Command (Modern Routing, v0.3+) ────────────────────────────────

from langgraph.types import Command

def modern_routing_node(state: RouterState) -> Command:
    """
    Modern routing with Command: node returns both state updates AND routing.
    Cleaner than separate routing functions for simple cases.
    """
    query = state["messages"][-1].content
    
    # Decide route based on simple heuristic
    if "?" in query and len(query) > 20:
        next_node = "retrieve"
    else:
        next_node = "direct_chat"

    return Command(
        goto=next_node,              # Where to route
        update={"query_type": "factual"},  # State update
    )
```

---

## 4. Interview Preparation

**Mid-level**: "Conditional edges in LangGraph are implemented as routing functions — they take the current state and return a string indicating which node to go to next. `add_conditional_edges(from_node, routing_fn, {output: node})` maps the function's return values to node names. This enables: query type routing (factual → RAG, SQL → text-to-SQL), quality loops (regenerate if poor quality, up to 3 attempts), and early stopping (if confidence is high, skip reranking)."

**Senior**: "I use three routing patterns in production: (1) Multi-way router at graph entry — classify query type once, route to the appropriate sub-graph; (2) Quality loop — generate, evaluate, regenerate if needed with a hard iteration cap (prevents infinite loops); (3) Cascading fallback — try primary, fall back to simpler approach if primary fails. The modern `Command` return type (LangGraph 0.3+) lets nodes return both state updates AND routing decisions, which eliminates the need for separate routing functions for simple cases."

---

---

# Chapter 4: ReAct Agent Pattern

---

## 1. Introduction

### What Is ReAct?

**ReAct** (Reason + Act) is the foundational pattern for AI agents. Published by Yao et al. (2022), it interleaves:
- **Reasoning**: The LLM thinks about what to do next
- **Acting**: The LLM executes an action (tool call)
- **Observing**: The LLM reads the tool's output
- **Repeat**: Until a final answer is reached

LangGraph implements ReAct naturally: the `agent` node reasons, the `tools` node acts, and the back-edge creates the observation loop.

---

## 2. Visual Mental Model

```mermaid
stateDiagram-v2
    [*] --> Reason
    Reason --> Act : LLM decides to use tool
    Reason --> FinalAnswer : LLM has enough info
    Act --> Observe : Tool returns result
    Observe --> Reason : Add result to messages
    FinalAnswer --> [*]

    note right of Reason
        LLM reads all messages
        (history + tool results)
        and decides next step
    end note

    note right of Act
        Execute tool:
        web search, calculator,
        DB query, API call
    end note
```

---

## 3. Implementation

### Custom ReAct Agent from Scratch

```python
"""
ReAct agent built from scratch with LangGraph.
Shows exactly what's happening inside create_react_agent().
"""

from typing import TypedDict, Annotated, List, Literal, Union
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import json
import asyncio


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """Search the web for current information. Returns top 3 results."""
    # Mock — replace with Tavily, SerpAPI, etc.
    return f"[Web Search: {query}]\nResult 1: ...\nResult 2: ...\nResult 3: ..."


@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for company-specific information."""
    return f"[KB Search: {query}]\nRelevant document: ..."


@tool
def python_repl(code: str) -> str:
    """Execute Python code for data analysis. Returns stdout output."""
    import io, contextlib
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"__builtins__": {"print": print, "range": range, "sum": sum, "len": len}})
        return output.getvalue() or "(no output)"
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─── State ────────────────────────────────────────────────────────────────────

class ReActState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    iteration: int  # Track how many ReAct loops we've done


# ─── Custom ReAct Agent ───────────────────────────────────────────────────────

def build_react_agent_from_scratch():
    """
    Build ReAct agent manually to understand what's happening.
    For production, use langgraph.prebuilt.create_react_agent.
    """
    tools = [search_web, search_knowledge_base, python_repl, get_current_time]
    
    # Bind tools to LLM (enables function/tool calling)
    llm_with_tools = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(
        tools,
        tool_choice="auto",  # LLM decides when to use tools
    )

    system_prompt = """You are a helpful research assistant.
Use tools to gather information before answering.
Think step by step about what information you need.
After gathering enough information, provide a comprehensive final answer."""

    # ─── Node 1: Agent (Reason) ───────────────────────────────────────────────
    def agent_node(state: ReActState) -> dict:
        """The reasoning node: LLM reads all messages and decides next step."""
        messages = [
            {"role": "system", "content": system_prompt},
            *[m for m in state["messages"]],
        ]
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "iteration": state.get("iteration", 0) + 1,
        }

    # ─── Node 2: Tools (Act + Observe) ───────────────────────────────────────
    # LangGraph's ToolNode handles tool execution + wrapping in ToolMessage
    tool_node = ToolNode(tools)

    # ─── Routing: Should we call tools or give final answer? ─────────────────
    def should_continue(state: ReActState) -> Literal["tools", "__end__"]:
        """
        Check the last message:
        - Has tool_calls → go to tools node (Act)
        - No tool_calls → agent has final answer → END
        
        Also enforce max iterations to prevent infinite loops.
        """
        if state.get("iteration", 0) >= 10:
            return "__end__"  # Hard stop after 10 iterations

        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
        return "__end__"

    # ─── Build Graph ──────────────────────────────────────────────────────────
    graph = StateGraph(ReActState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    graph.add_edge("tools", "agent")  # The ReAct cycle: tools → agent → tools → ...

    app = graph.compile()
    return app


# ─── Production ReAct with prebuilt helper ────────────────────────────────────

def build_production_react_agent():
    """
    Production ReAct agent using LangGraph's built-in helper.
    Equivalent to the manual version above, but with built-in optimizations.
    """
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage

    tools = [search_web, search_knowledge_base, python_repl, get_current_time]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    app = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SystemMessage(content="""You are a helpful research assistant.
Use tools to gather information. Think step by step.
Always verify information before including it in your final answer."""),
        # state_modifier allows pre-processing state before LLM call
        # checkpointer for persistence (covered in Chapter 7)
    )

    return app


# ─── Streaming ReAct Execution ────────────────────────────────────────────────

async def stream_react_agent(query: str):
    """Stream a ReAct agent's execution, showing each node's output."""
    app = build_production_react_agent()
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)

    async for event in app.astream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode="updates",
    ):
        for node_name, updates in event.items():
            print(f"\n[Node: {node_name.upper()}]")
            for msg in updates.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"  🔧 Tool Call: {tc['name']}({json.dumps(tc['args'])[:100]})")
                elif isinstance(msg, ToolMessage):
                    print(f"  📋 Tool Result: {msg.content[:150]}...")
                elif isinstance(msg, AIMessage):
                    print(f"  🤖 AI: {msg.content[:200]}...")


# asyncio.run(stream_react_agent("What is today's date and what is 2024 + 2025?"))
```

---

## 4. Interview Preparation

**Mid-level**: "The ReAct pattern in LangGraph is a two-node cycle: `agent → tools → agent`. The `agent` node calls the LLM (which is bound with tools). If the LLM responds with a tool call, the conditional edge routes to `tools`. The `ToolNode` executes the tool and adds a `ToolMessage` with the result. Control returns to `agent`, which now has the tool result in its message history and can reason further. The cycle continues until the LLM gives a final answer without tool calls."

**Senior**: "I always implement two safety constraints on ReAct agents: (1) iteration cap — the routing function checks `state['iteration'] >= N` and routes to END regardless of tool calls; (2) tool timeout — ToolNode wraps each tool in a try-except with a timeout. Without these, a buggy tool that always errors can loop infinitely. For production, I prefer `create_react_agent` over the manual implementation because it handles edge cases in tool calling that are subtle to get right."

---

---

# Chapter 5: Multi-Agent Systems

---

## 1. Introduction

### What Are Multi-Agent Systems?

A **multi-agent system** is a LangGraph application where multiple specialized AI agents collaborate on a task, each handling a specific aspect:

- **Researcher agent**: Searches and gathers information
- **Writer agent**: Drafts content from gathered research
- **Reviewer agent**: Evaluates quality and provides feedback
- **Supervisor agent**: Orchestrates which agent works next

Multi-agent systems excel when a task is too complex for one agent, requires specialized tools/prompts, or benefits from parallel processing.

---

## 2. Visual Mental Model

### Supervisor Pattern

```mermaid
flowchart TD
    subgraph "Supervisor Multi-Agent System"
        USER[User Task] --> SUP[Supervisor Agent\nOrchestrates which agent to call]
        
        SUP -->|"next: researcher"| RES[Researcher Agent\nWeb search + KB search]
        SUP -->|"next: writer"| WRI[Writer Agent\nCreates draft from notes]
        SUP -->|"next: reviewer"| REV[Reviewer Agent\nScores quality]
        SUP -->|"next: FINISH"| END((End))
        
        RES --> SUP
        WRI --> SUP
        REV --> SUP
    end

    subgraph "Shared Team State"
        S["task: str\nresearch_notes: list\ndraft: str\nreview_score: int\nnext: str"]
    end
```

### Hierarchical Pattern

```mermaid
flowchart TD
    USER[User] --> META[Meta Supervisor\nHigh-level orchestration]
    META --> RESEARCH_TEAM[Research Supervisor\nManages researcher sub-agents]
    META --> WRITING_TEAM[Writing Supervisor\nManages writer sub-agents]
    RESEARCH_TEAM --> R1[Researcher 1\nWeb search]
    RESEARCH_TEAM --> R2[Researcher 2\nAcademic papers]
    WRITING_TEAM --> W1[Drafter\nCreates outline + draft]
    WRITING_TEAM --> W2[Editor\nPolishes text]
```

---

## 3. Implementation

### Supervisor Multi-Agent System

```python
"""
Multi-agent system with supervisor orchestration.
Implements: researcher, writer, reviewer, supervisor pattern.
"""

import asyncio
from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import operator
from pydantic import BaseModel


# ─── State ────────────────────────────────────────────────────────────────────

class TeamState(TypedDict):
    task: str                                              # Original task
    messages: Annotated[List[BaseMessage], add_messages]   # Conversation history
    research_notes: Annotated[List[str], operator.add]     # Accumulated notes
    draft: Optional[str]                                   # Written draft
    review_score: Optional[int]                            # Quality score 1-10
    review_feedback: Optional[str]                         # Reviewer's feedback
    revision_count: int                                    # Number of revisions
    next: Optional[str]                                    # Which agent to call next


# ─── LLM Setup ────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
llm_structured = ChatOpenAI(model="gpt-4o", temperature=0)


# ─── Agent Nodes ──────────────────────────────────────────────────────────────

def researcher_agent(state: TeamState) -> dict:
    """
    Research agent: searches for information and adds notes.
    Has access to web search and knowledge base tools.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research specialist.
Research the given task thoroughly.
Return a comprehensive bullet-point list of key findings.
Focus on facts, statistics, and specific details."""),
        ("human", "Task: {task}\n\nPrevious research notes:\n{notes}"),
    ])

    chain = prompt | llm | StrOutputParser()
    notes = chain.invoke({
        "task": state["task"],
        "notes": "\n".join(state.get("research_notes", [])) or "None yet",
    })

    return {
        "research_notes": [notes],  # operator.add appends to list
        "messages": [AIMessage(content=f"[Researcher] Completed research:\n{notes[:200]}...")],
    }


def writer_agent(state: TeamState) -> dict:
    """
    Writer agent: creates a draft from research notes.
    """
    research_context = "\n\n".join(state.get("research_notes", []))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional writer.
Create a well-structured, clear draft based on the research notes.
The draft should be comprehensive, well-organized, and engaging."""),
        ("human", "Task: {task}\n\nResearch notes:\n{research}\n\nReviewer feedback (if any): {feedback}"),
    ])

    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "task": state["task"],
        "research": research_context or "No research available",
        "feedback": state.get("review_feedback") or "First draft",
    })

    return {
        "draft": draft,
        "revision_count": state.get("revision_count", 0) + 1,
        "messages": [AIMessage(content=f"[Writer] Draft created ({len(draft)} chars)")],
    }


def reviewer_agent(state: TeamState) -> dict:
    """
    Reviewer agent: evaluates draft quality and provides feedback.
    """
    class ReviewOutput(BaseModel):
        score: int = Field(ge=1, le=10, description="Quality score 1-10")
        feedback: str
        approved: bool = Field(description="True if score >= 7 and ready to publish")

    reviewer_llm = llm_structured.with_structured_output(ReviewOutput)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a quality reviewer. Evaluate this draft rigorously."),
        ("human", "Task: {task}\n\nDraft:\n{draft}\n\nRate quality 1-10, provide specific feedback."),
    ])

    review = (prompt | reviewer_llm).invoke({
        "task": state["task"],
        "draft": state.get("draft", ""),
    })

    return {
        "review_score": review.score,
        "review_feedback": review.feedback,
        "messages": [AIMessage(content=f"[Reviewer] Score: {review.score}/10. {review.feedback[:200]}")],
    }


def supervisor_agent(state: TeamState) -> dict:
    """
    Supervisor: orchestrates which agent should work next.
    Makes decisions based on the current state of the workflow.
    """
    class SupervisorDecision(BaseModel):
        next: Literal["researcher", "writer", "reviewer", "FINISH"]
        reasoning: str

    supervisor_llm = llm_structured.with_structured_output(SupervisorDecision)

    status = f"""Task: {state['task']}
Research notes: {len(state.get('research_notes', []))} entries
Draft: {'Yes' if state.get('draft') else 'No'}
Review score: {state.get('review_score', 'Not reviewed')}
Revision count: {state.get('revision_count', 0)}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a project supervisor. 
Decide which agent should work next.
Rules:
- researcher: if no research notes yet, or researcher has been called < 2 times for complex tasks
- writer: if research is done but no draft, OR reviewer gave feedback and revisions < 3
- reviewer: if a draft exists and hasn't been reviewed yet, or after writer revision
- FINISH: if reviewer score >= 8 OR revision_count >= 3 (don't over-revise)"""),
        ("human", "Current status:\n{status}\n\nWho should work next?"),
    ])

    decision = (prompt | supervisor_llm).invoke({"status": status})

    return {
        "next": decision.next,
        "messages": [AIMessage(content=f"[Supervisor] Next: {decision.next}. {decision.reasoning[:100]}")],
    }


# ─── Build Supervisor Graph ────────────────────────────────────────────────────

def build_supervisor_system():
    """Build the multi-agent supervisor system."""
    graph = StateGraph(TeamState)

    # Add all agent nodes
    graph.add_node("supervisor", supervisor_agent)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("reviewer", reviewer_agent)

    # All agents report back to supervisor after completing
    graph.add_edge(START, "supervisor")
    graph.add_edge("researcher", "supervisor")
    graph.add_edge("writer", "supervisor")
    graph.add_edge("reviewer", "supervisor")

    # Supervisor routes to the right agent
    def route_supervisor(state: TeamState) -> str:
        next_agent = state.get("next", "researcher")
        if next_agent == "FINISH":
            return END
        return next_agent

    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "researcher": "researcher",
            "writer": "writer",
            "reviewer": "reviewer",
            END: END,
        }
    )

    app = graph.compile()
    return app


# ─── Network Pattern: Peer-to-peer agents ────────────────────────────────────

async def demo_supervisor():
    """Run the multi-agent system on a sample task."""
    app = build_supervisor_system()

    initial_state = {
        "task": "Write a comprehensive guide on HNSW algorithm for vector databases",
        "messages": [],
        "research_notes": [],
        "draft": None,
        "review_score": None,
        "review_feedback": None,
        "revision_count": 0,
        "next": None,
    }

    print("Starting multi-agent workflow...\n")

    async for event in app.astream(initial_state, stream_mode="updates"):
        for node_name, updates in event.items():
            msgs = updates.get("messages", [])
            for msg in msgs:
                print(msg.content[:200])

    final_state = app.invoke(initial_state)
    print(f"\n{'='*60}")
    print(f"Final Review Score: {final_state['review_score']}/10")
    print(f"Revisions: {final_state['revision_count']}")
    print(f"\nFinal Draft (first 500 chars):\n{final_state['draft'][:500]}...")
```

---

## 4. Interview Preparation

**Mid-level**: "Multi-agent systems in LangGraph use the supervisor pattern: a supervisor agent decides which specialized agent to call next based on the current state. All agents report back to the supervisor after completing their task. The supervisor's routing decision is stored in the state (`next` field) and the conditional edge reads it to route control."

**Senior**: "I use multi-agent systems when a task has clearly separable specialist roles that benefit from different prompts, tools, and expertise. The key architectural decision: supervisor (single orchestrator, more predictable) vs. network (agents communicate peer-to-peer, more flexible but harder to debug). I always add iteration caps to both the supervisor (max routing decisions) and each specialist (max attempts). I monitor agent-to-agent handoffs in LangSmith to identify coordination failures — when the supervisor routes to the wrong agent or when specialists produce output that doesn't satisfy downstream agents."

---

---

# Chapter 6: Human-in-the-Loop

---

## 1. Introduction

### What Is Human-in-the-Loop?

**Human-in-the-Loop (HITL)** is the capability to pause an automated AI workflow at a designated checkpoint, present the current state to a human for review, allow the human to approve, reject, or modify it, and then resume the workflow accordingly.

This is critical for:
- **High-stakes actions**: Sending emails, deleting records, financial transactions
- **Quality gates**: Review a generated document before publishing
- **Ambiguity resolution**: Ask a human when the AI is uncertain
- **Compliance**: Regulatory requirements for human oversight

---

## 2. Visual Mental Model

```mermaid
stateDiagram-v2
    [*] --> PlanAction
    PlanAction --> PAUSED : interrupt_before("execute")
    
    state PAUSED {
        Human reviews the planned action
        Human approves / modifies / rejects
    }

    PAUSED --> Execute : Human approves\n(graph.invoke with Command(resume=...))
    PAUSED --> Replan : Human rejects\n(route to alternative node)
    Execute --> Done
    Replan --> PlanAction

    Done --> [*]
```

---

## 3. Implementation

```python
"""
Human-in-the-Loop with LangGraph checkpointing.
"""

from typing import TypedDict, Annotated, List, Optional, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


# ─── State ────────────────────────────────────────────────────────────────────

class HITLState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    planned_action: Optional[str]   # What the AI plans to do
    action_result: Optional[str]    # Result after execution
    human_approved: Optional[bool]  # Human approval decision
    human_feedback: Optional[str]   # Human's modification/feedback


# ─── Pattern 1: interrupt() — Modern HITL (LangGraph 0.3+) ────────────────────

def build_hitl_graph_modern():
    """
    Modern HITL using interrupt() function.
    The node itself calls interrupt() to pause and wait for human input.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    checkpointer = MemorySaver()

    def plan_node(state: HITLState) -> dict:
        """Plan an action based on the user's request."""
        last_msg = state["messages"][-1].content
        plan = f"I will: Send an email to all customers about: {last_msg}"
        return {
            "planned_action": plan,
            "messages": [AIMessage(content=f"Planned action: {plan}")],
        }

    def human_review_node(state: HITLState) -> dict:
        """
        HITL node: pause and wait for human approval.
        
        interrupt() raises a special exception that LangGraph catches.
        The graph is PAUSED here. When resumed with Command(resume=value),
        interrupt() returns that value and execution continues.
        """
        # Present the plan to the human
        print(f"\n[HUMAN REVIEW REQUIRED]")
        print(f"Planned action: {state['planned_action']}")
        print("Please approve (yes/no) or provide modified action:")

        # This PAUSES the graph and waits for human input
        # Value provided by human via: graph.invoke(Command(resume="yes"), config)
        human_input: str = interrupt(
            value={
                "planned_action": state["planned_action"],
                "question": "Do you approve this action? (yes/no or provide modification)",
            }
        )

        # After resume, human_input contains what was passed to Command(resume=...)
        approved = human_input.lower() in ["yes", "y", "approve"]
        modification = human_input if not approved else None

        return {
            "human_approved": approved,
            "human_feedback": modification,
        }

    def execute_node(state: HITLState) -> dict:
        """Execute the approved action."""
        action = state["planned_action"]
        result = f"✅ Executed: {action}"
        return {
            "action_result": result,
            "messages": [AIMessage(content=result)],
        }

    def revision_node(state: HITLState) -> dict:
        """Revise the plan based on human feedback."""
        feedback = state.get("human_feedback", "No specific feedback")
        new_plan = f"[Revised based on feedback: {feedback}]"
        return {
            "planned_action": new_plan,
            "messages": [AIMessage(content=f"Revised plan: {new_plan}")],
        }

    def reject_node(state: HITLState) -> dict:
        """Handle rejected actions."""
        return {
            "action_result": "❌ Action rejected by human reviewer",
            "messages": [AIMessage(content="Action was rejected. Task cancelled.")],
        }

    def route_after_review(state: HITLState) -> str:
        if state.get("human_approved"):
            return "execute"
        elif state.get("human_feedback"):
            return "revise"
        return "reject"

    # Build graph
    graph = StateGraph(HITLState)
    graph.add_node("plan", plan_node)
    graph.add_node("human_review", human_review_node)  # HITL node
    graph.add_node("execute", execute_node)
    graph.add_node("revise", revision_node)
    graph.add_node("reject", reject_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"execute": "execute", "revise": "revise", "reject": "reject"},
    )
    graph.add_edge("revise", "human_review")  # Loop back for revised plan review
    graph.add_edge("execute", END)
    graph.add_edge("reject", END)

    # IMPORTANT: Checkpointer enables state persistence across the pause
    app = graph.compile(checkpointer=checkpointer)
    return app


# ─── Running HITL Graph ───────────────────────────────────────────────────────

def demo_hitl():
    """
    Demonstrate the full HITL workflow:
    1. Start the graph → it pauses at human_review
    2. Human approves → graph continues to execute
    """
    app = build_hitl_graph_modern()
    
    # Thread config: unique ID per workflow instance
    config = {"configurable": {"thread_id": "hitl-demo-001"}}

    # Step 1: Start the graph
    print("Step 1: Starting workflow...")
    initial_state = {
        "messages": [HumanMessage(content="Send a product update email to all customers")],
        "planned_action": None,
        "action_result": None,
        "human_approved": None,
        "human_feedback": None,
    }

    # Graph will run until it hits interrupt() in human_review_node
    result = app.invoke(initial_state, config=config)
    print(f"Graph paused. Current state: {result}")

    # At this point, the graph is PAUSED.
    # In production: send the interrupt value to a UI, wait for user response.
    # Here we simulate human approval:

    print("\nStep 2: Human approves the action...")
    
    # Resume the graph with human's decision
    # Command(resume=value) provides the value that interrupt() will return
    final_result = app.invoke(
        Command(resume="yes"),  # Human approved
        config=config,
    )
    
    print(f"\nFinal result: {final_result['action_result']}")
    return final_result


# ─── Pattern 2: interrupt_before (compile-time) ───────────────────────────────

def build_hitl_graph_compiletime():
    """
    Alternative: specify interrupt nodes at compile time.
    Simpler than using interrupt() function, but less flexible.
    """
    checkpointer = MemorySaver()

    class SimpleState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        plan: Optional[str]

    def plan_node(state):
        return {"plan": "Send email to all users", "messages": [AIMessage(content="Planned")]}

    def execute_node(state):
        return {"messages": [AIMessage(content=f"Executed: {state['plan']}")]}

    graph = StateGraph(SimpleState)
    graph.add_node("plan", plan_node)
    graph.add_node("execute", execute_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", END)

    # interrupt_before="execute" pauses BEFORE execute_node runs
    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["execute"],
    )

    # Usage:
    config = {"configurable": {"thread_id": "demo"}}
    state = {"messages": [HumanMessage(content="send email")], "plan": None}

    # First call: runs plan_node, stops BEFORE execute_node
    app.invoke(state, config=config)

    # Second call (after human review): resumes from execute_node
    # Can update state before resuming:
    app.update_state(config, {"plan": "[Modified plan based on review]"})
    final = app.invoke(None, config=config)  # None = continue from checkpoint
    return final
```

---

## 4. Interview Preparation

**Mid-level**: "Human-in-the-loop in LangGraph works through `interrupt()`. When a node calls `interrupt(value)`, the graph pauses and the interrupt value is surfaced to the caller. The graph state is saved to the checkpointer. When the human responds, `graph.invoke(Command(resume=human_response), config)` resumes the graph from where it paused, with `interrupt()` returning the human's response."

**Senior**: "I build HITL into every agent that can take irreversible actions — sending messages, modifying records, triggering payments. The pattern: plan node (what to do), review node (interrupt for human approval), execute node (do it). I expose the interrupt value through a webhook or WebSocket to the frontend, with a timeout (e.g., 24 hours) after which the action is auto-rejected. The checkpointer (PostgresSaver in production) persists the graph state so the approval flow survives server restarts."

---

---

# Chapter 7: Persistence and Checkpointing

---

## 1. Introduction

### What Is Checkpointing?

**Checkpointing** is LangGraph's persistence mechanism: the ability to save the complete graph state at every step, enabling:
- **Resumability**: Long-running workflows can be resumed after interruption
- **Human-in-the-loop**: Required for HITL — state must survive the pause
- **Branching**: Explore alternative paths from the same checkpoint
- **Debugging**: Replay a workflow from any saved checkpoint
- **Multi-turn sessions**: Conversation state persists across requests

---

## 2. Visual Mental Model

```mermaid
flowchart TD
    subgraph "Without Checkpointing"
        A[Node A] --> B[Node B] --> C[Node C]
        NOTE1["If server crashes at Node B:\n❌ All work lost\n❌ Must restart from scratch"]
    end

    subgraph "With Checkpointing (MemorySaver/PostgresSaver)"
        D[Node A] --> CP1[Checkpoint\nSaved] --> E[Node B] --> CP2[Checkpoint\nSaved] --> F[Node C]
        NOTE2["If server crashes at Node B:\n✅ Resume from Node B checkpoint\n✅ Zero work lost"]
    end
```

---

## 3. Implementation

### Checkpointer Types and Usage

```python
"""
LangGraph checkpointers: from in-memory to production PostgreSQL.
"""

from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command


# ─── State ────────────────────────────────────────────────────────────────────

class CheckpointState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    step: int


# ─── Simple Graph for Demonstration ──────────────────────────────────────────

def build_checkpointed_graph(checkpointer):
    """Build a simple graph with checkpointing enabled."""
    llm = ChatOpenAI(model="gpt-4o-mini")

    def chat_node(state: CheckpointState) -> dict:
        response = llm.invoke(state["messages"])
        return {
            "messages": [response],
            "step": state.get("step", 0) + 1,
        }

    graph = StateGraph(CheckpointState)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    # Checkpointer attached at compile time
    app = graph.compile(checkpointer=checkpointer)
    return app


# ─── 1. MemorySaver (Development) ────────────────────────────────────────────

def demo_memory_saver():
    """
    MemorySaver: in-memory checkpointing.
    
    - Zero setup, no external dependencies
    - Data lost on process restart
    - Perfect for development and testing
    - Supports full HITL and multi-turn conversations
    """
    saver = MemorySaver()
    app = build_checkpointed_graph(saver)

    # thread_id identifies this conversation session
    config = {"configurable": {"thread_id": "user-session-001"}}

    # Turn 1: Initial message
    state1 = {"messages": [HumanMessage(content="My name is Alice")], "step": 0}
    result1 = app.invoke(state1, config=config)
    print(f"Turn 1: {result1['messages'][-1].content[:100]}")

    # Turn 2: Follow-up — graph REMEMBERS Alice from Turn 1
    state2 = {"messages": [HumanMessage(content="What's my name?")], "step": 0}
    result2 = app.invoke(state2, config=config)
    print(f"Turn 2: {result2['messages'][-1].content}")  # Should mention Alice

    # Inspect checkpoints
    history = list(app.get_state_history(config))
    print(f"Checkpoints saved: {len(history)}")


# ─── 2. SQLite Saver (Local Persistent Storage) ───────────────────────────────

def demo_sqlite_saver(db_path: str = "checkpoints.db"):
    """
    SQLiteSaver: file-based persistence.
    
    - Persists across process restarts
    - Single-machine only (no distributed support)
    - Great for local apps and small teams
    - pip install langgraph-checkpoint-sqlite
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    with SqliteSaver.from_conn_string(db_path) as saver:
        app = build_checkpointed_graph(saver)

        config = {"configurable": {"thread_id": "persistent-session"}}
        state = {"messages": [HumanMessage(content="Hello!")], "step": 0}
        result = app.invoke(state, config=config)
        print(result["messages"][-1].content)


# ─── 3. PostgreSQL Saver (Production) ─────────────────────────────────────────

async def demo_postgres_saver():
    """
    PostgresSaver: distributed production persistence.
    
    - Survives server restarts and deployments
    - Supports multiple server instances
    - Production-grade durability
    - pip install langgraph-checkpoint-postgres
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    async with AsyncPostgresSaver.from_conn_string(
        "postgresql://user:password@localhost:5432/langchain_checkpoints"
    ) as saver:
        # Create required tables (only needed once)
        await saver.setup()

        app = build_checkpointed_graph(saver)
        config = {"configurable": {"thread_id": "prod-session-001"}}
        state = {"messages": [HumanMessage(content="Hello!")], "step": 0}
        result = await app.ainvoke(state, config=config)
        print(result["messages"][-1].content)


# ─── 4. State History and Time Travel ─────────────────────────────────────────

def demo_time_travel(app, config: dict):
    """
    LangGraph's 'time travel': replay from any historical checkpoint.
    
    Use cases:
    - Debug a failed workflow by replaying from before the failure
    - Try an alternative path from a decision point
    - Roll back to a previous state and try a different prompt
    """
    # Get all checkpoints (newest first)
    history = list(app.get_state_history(config))
    print(f"Total checkpoints: {len(history)}")

    for i, checkpoint in enumerate(history):
        print(f"  [{i}] Step {checkpoint.values.get('step', '?')}: "
              f"{checkpoint.config['configurable']['checkpoint_id']}")

    # Time travel: restart from checkpoint 2
    if len(history) >= 3:
        old_checkpoint_config = history[-3].config  # 3rd most recent

        # Resume from this old checkpoint
        # All subsequent steps will be re-executed from this point
        result = app.invoke(
            None,   # Don't add new input; resume from checkpoint
            config=old_checkpoint_config,
        )
        print(f"Resumed from checkpoint: {result}")


# ─── 5. Branching from a Checkpoint ──────────────────────────────────────────

def demo_branching(app, config: dict):
    """
    Branch: create alternative conversation paths from the same checkpoint.
    
    Use case: A/B test different system prompts or approaches
    starting from the same conversation history.
    """
    history = list(app.get_state_history(config))
    
    if not history:
        return

    # Take the 2nd checkpoint as our branch point
    branch_point = history[-2]
    branch_config = branch_point.config

    # Branch A: Continue with original state (implicit)
    result_a = app.invoke(
        {"messages": [HumanMessage(content="Path A: Tell me more")]},
        config=branch_config,
    )

    # Branch B: Create a new thread from the same checkpoint
    branch_config_b = {"configurable": {
        "thread_id": f"{config['configurable']['thread_id']}-branch-b",
        "checkpoint_ns": "",
    }}
    app.update_state(branch_config_b, branch_point.values)
    result_b = app.invoke(
        {"messages": [HumanMessage(content="Path B: Alternative approach")]},
        config=branch_config_b,
    )

    print(f"Branch A: {result_a['messages'][-1].content[:100]}")
    print(f"Branch B: {result_b['messages'][-1].content[:100]}")
```

---

## 4. Interview Preparation

**Mid-level**: "LangGraph checkpointers save the complete graph state after every node execution. I use MemorySaver for development (in-memory, zero setup), SqliteSaver for single-machine persistent storage, and PostgresSaver for production (distributed, survives deployments). The thread_id in the config config identifies a conversation session — passing the same thread_id on subsequent calls retrieves the previous state and appends to it."

**Senior**: "Checkpointing is the foundation of three critical capabilities: (1) HITL — the graph state must survive the pause between human review and approval, which can be hours or days; (2) Multi-turn chat — conversation history persists across HTTP requests; (3) Long-running agent workflows — a 30-minute research agent can resume after a server restart. For PostgreSQL, I partition checkpoints by thread_id and add a TTL to manage storage. I also enable 'time travel' in the admin UI — I can replay any failed workflow from any historical checkpoint to debug it."

---

---

# Chapter 8: Production Patterns

---

## 1. Introduction

This chapter covers the essential patterns for deploying LangGraph applications in production: FastAPI integration, streaming, multi-tenant configuration, monitoring, and deployment architecture.

---

## 2. Implementation

### Production LangGraph Service

```python
"""
Production LangGraph: FastAPI + checkpointing + streaming + observability.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from typing import TypedDict, Annotated, List
import logging

logger = logging.getLogger(__name__)


# ─── Application Setup ────────────────────────────────────────────────────────

class AppState:
    agent: any = None
    checkpointer: MemorySaver = None


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize LangGraph resources at startup."""
    logger.info("Building LangGraph agent...")

    app_state.checkpointer = MemorySaver()  # Replace with PostgresSaver in production
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    app_state.agent = create_react_agent(
        model=llm,
        tools=[],  # Add your tools here
        checkpointer=app_state.checkpointer,
    )

    logger.info("Agent ready.")
    yield
    logger.info("Shutdown complete.")


app = FastAPI(title="LangGraph Production API", lifespan=lifespan)


# ─── Request Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str   # Session identifier
    stream: bool = True


class ChatResponse(BaseModel):
    response: str
    thread_id: str


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with session persistence via checkpointing.
    
    thread_id persists conversation across requests.
    Same thread_id on subsequent calls → agent remembers previous messages.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    inputs = {"messages": [HumanMessage(content=request.message)]}

    if request.stream:
        async def generate() -> AsyncIterator[str]:
            try:
                async for event in app_state.agent.astream(
                    inputs,
                    config=config,
                    stream_mode="messages",  # Stream individual message tokens
                ):
                    # event = (message_chunk, metadata)
                    if isinstance(event, tuple):
                        msg_chunk, metadata = event
                        if hasattr(msg_chunk, "content") and msg_chunk.content:
                            data = json.dumps({"token": msg_chunk.content, "done": False})
                            yield f"data: {data}\n\n"

                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"

            except Exception as e:
                logger.error(f"Stream error: {e}")
                error_data = json.dumps({"error": str(e), "done": True})
                yield f"data: {error_data}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    else:
        result = await app_state.agent.ainvoke(inputs, config=config)
        last_msg = result["messages"][-1]
        return ChatResponse(
            response=last_msg.content,
            thread_id=request.thread_id,
        )


@app.get("/sessions/{thread_id}/history")
async def get_history(thread_id: str):
    """Get conversation history for a session."""
    config = {"configurable": {"thread_id": thread_id}}
    state = app_state.agent.get_state(config)
    
    if not state.values:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = state.values.get("messages", [])
    return {
        "thread_id": thread_id,
        "message_count": len(messages),
        "messages": [
            {"role": msg.type, "content": msg.content[:200]}
            for msg in messages
        ],
    }


@app.delete("/sessions/{thread_id}")
async def clear_session(thread_id: str):
    """Clear a conversation session."""
    # With MemorySaver: remove from store
    # With PostgresSaver: delete from DB
    return {"cleared": True, "thread_id": thread_id}


@app.get("/health")
async def health():
    return {"status": "healthy", "agent_ready": app_state.agent is not None}


# ─── Production: PostgreSQL Checkpointer Setup ────────────────────────────────

async def setup_production_checkpointer():
    """
    Production checkpointer using PostgreSQL.
    Replace MemorySaver with this for persistent, distributed state.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql://user:pass@localhost:5432/langgraph_prod"
    )

    checkpointer = AsyncPostgresSaver.from_conn_string(DATABASE_URL)
    await checkpointer.setup()  # Create required tables
    return checkpointer


# ─── Production: Multi-tenant Configuration ───────────────────────────────────

def get_tenant_config(tenant_id: str, thread_id: str) -> dict:
    """
    Multi-tenant configuration.
    thread_id includes tenant_id to ensure isolation.
    """
    return {
        "configurable": {
            "thread_id": f"{tenant_id}:{thread_id}",  # Namespace by tenant
        },
        "tags": [f"tenant:{tenant_id}"],
        "metadata": {"tenant_id": tenant_id},
    }
```

---

## 3. Common Mistakes

❌ **Not setting `max_iterations` in ReAct agents**: Without an iteration cap, a buggy tool that always errors causes an infinite loop that exhausts your API budget.

❌ **Using MemorySaver in production**: MemorySaver stores everything in RAM — data is lost on restart, and it grows unbounded. Use PostgresSaver with TTL.

❌ **Not handling `interrupt()` returns in HITL graphs**: After `interrupt()`, the `Command(resume=value)` must be passed to the next `invoke()` call with the SAME config (thread_id). Using a different config restarts from scratch.

❌ **Not compiling with checkpointer for multi-turn chat**: Calling `graph.compile()` without a checkpointer means the graph has NO memory between invocations. Each call is stateless.

❌ **Returning full state from nodes**: Nodes should return only changed fields. Returning a full state copy breaks reducers.

---

## 4. Interview Preparation

**Junior**: "LangGraph builds stateful AI workflows using a graph model. Nodes process data, edges define flow, and the state accumulates information. With a checkpointer, the graph remembers everything across multiple requests."

**Mid-level**: "For production LangGraph: (1) PostgresSaver for checkpointing — persists state across restarts; (2) `stream_mode='messages'` for streaming individual tokens; (3) thread_id per user session for conversation isolation; (4) `interrupt_before=["execute"]` or `interrupt()` for HITL; (5) iteration caps on all loops. The key insight: LangGraph turns a series of API calls into a managed state machine."

**Senior**: "LangGraph in production is fundamentally state machine engineering. I design: (1) State schema first — what does every node need to read/write? (2) Error recovery paths — what happens when a node fails? (3) HITL checkpoints — which nodes require human approval? (4) Parallelism — which nodes can run concurrently using `Send` API? (5) Observability — LangSmith traces every node with full state diffs. The PostgreSQL schema for checkpoints is critical — I add indexes on thread_id and created_at, and a TTL cleanup job to manage storage."

---

## 5. Follow-up Questions

**Q1: What is the difference between LangGraph's `stream_mode="values"` and `stream_mode="messages"`?**
> `stream_mode="values"`: Yields the complete state after each node finishes. Good for seeing what changed after each step. `stream_mode="messages"`: Yields individual message tokens as they're generated (like SSE streaming from an LLM). Good for user-facing streaming responses. `stream_mode="updates"`: Yields only the state changes (delta) from each node. Most efficient for debugging.

**Q2: What is the `Send` API and when do you use it?**
> `Send` is LangGraph's mechanism for dynamic parallelism — sending the same or different work to multiple node instances simultaneously. Instead of one call to "researcher", you can `Send` to 5 researcher instances in parallel, each with a different search query. Useful for: parallel web searches, map-reduce patterns, fan-out processing. The reducer in the state (using `operator.add`) combines their results.

**Q3: How do you debug a LangGraph workflow that's behaving unexpectedly?**
> Three techniques: (1) `stream_mode="updates"` — print state changes after every node to see what's being modified; (2) LangSmith traces — full visibility into every node's input and output; (3) Time travel — `app.get_state_history(config)` to see all checkpoints, then replay from a specific point to reproduce and debug the issue.

**Q4: What is the difference between create_react_agent and building manually?**
> `create_react_agent` is a convenience factory that builds the standard `agent → tools → agent` graph with sensible defaults. Build manually when you need: custom state fields, custom routing logic (not just tool vs. no-tool), additional nodes (e.g., a memory compression step), or different behavior after tool failures.

**Q5: How does LangGraph handle errors in nodes?**
> By default, node exceptions propagate up to the caller. For resilient graphs: (1) Use try-except inside the node function and return an error state; (2) Add fallback routing — `add_conditional_edges` that routes to an error recovery node based on error state; (3) Use `.with_retry()` on individual LLM calls within nodes; (4) Add a dedicated `handle_error` node as a catch-all.

**Q6: What is LangGraph Studio and when would you use it?**
> LangGraph Studio is a visual IDE for LangGraph — it renders your graph as an interactive diagram, lets you step through execution node by node, inspect state at each step, edit state and re-run, and test HITL flows interactively. Use it during development for: debugging unexpected routing, testing state transitions, validating HITL flows before writing tests.

---

## 6. Practical Scenario

### Scenario: Autonomous Customer Support Agent with HITL

**Context**: An e-commerce company wants an AI agent that can: research customer issues, draft refund decisions, require human approval for refunds over $500, and send resolution emails.

```python
class SupportState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    customer_id: str
    order_id: str
    issue: str
    research_notes: List[str]
    refund_amount: Optional[float]
    decision: Optional[str]
    human_approved: Optional[bool]

# Graph: research → decide → [HITL if >$500] → execute → notify
# interrupt() fires before execute if refund_amount > 500
# Human reviews: approve, reduce amount, or reject
```

**Result**: 80% of cases (< $500 refunds) handled fully automatically in < 30 seconds. 20% (≥ $500) go to human review, resolved in < 4 hours. Zero missed refunds, zero unauthorized large refunds.

---

## 7. Revision Sheet

```
Core Concepts:
□ StateGraph = directed graph with shared TypedDict state
□ Node = fn(State) → dict (partial state update, never mutate)
□ add_edge = unconditional flow; add_conditional_edges = routing function
□ Annotated[list, add_messages] = append reducer; default = replace
□ compile(checkpointer=...) = enables persistence + HITL

ReAct Pattern:
□ agent → tools → agent cycle (loop until no tool calls)
□ ToolNode handles tool execution + ToolMessage wrapping
□ iteration cap prevents infinite loops

Checkpointers:
□ MemorySaver = development; PostgresSaver = production
□ thread_id identifies session; same thread_id = continued conversation
□ Time travel: replay from any checkpoint

HITL:
□ interrupt(value) pauses graph; returns human's response
□ Command(resume=value) resumes graph
□ Requires checkpointer

Multi-Agent:
□ Supervisor pattern: supervisor → agent → supervisor → ...
□ All agents report back; supervisor decides next
□ Team state has operator.add reducers for parallel writes

Production:
□ PostgresSaver for production persistence
□ stream_mode="messages" for SSE streaming
□ thread_id namespaced per tenant for isolation
□ LangSmith tracing on all graphs
□ max_iterations on all cycles
```

---

## 8. Mini Project: Research + Writing Agent

Build a complete research and writing system:
1. **Input**: A topic (e.g., "LangGraph vs. LangChain comparison")
2. **Researcher agent**: 3 parallel web searches with `Send` API, RRF fusion of results
3. **Planner agent**: Creates an outline from research notes
4. **Writer agent**: Writes each section in parallel
5. **Reviewer agent**: Scores quality (1-10), provides feedback
6. **HITL**: Interrupt before publishing if score < 8
7. **Publisher node**: Saves to Notion / posts to blog
8. **Persistence**: PostgresSaver, resume if server restarts
9. **API**: FastAPI with SSE streaming of progress
10. **Monitoring**: LangSmith traces + Prometheus metrics per node

---

# Summary: Part 10 — LangGraph

This part built complete LangGraph mastery — from graph fundamentals to production multi-agent systems:

| Chapter | Core Skill |
|---|---|
| **1. Fundamentals** | StateGraph, Nodes, Edges, State reducers, compile(), stream modes |
| **2. State Management** | TypedDict patterns, Annotated reducers, custom merge functions, validation |
| **3. Conditional Routing** | Routing functions, multi-way routers, quality loops, Command API |
| **4. ReAct Pattern** | Agent→Tools cycle, ToolNode, iteration caps, streaming |
| **5. Multi-Agent** | Supervisor pattern, TeamState, peer coordination, parallel agents |
| **6. Human-in-the-Loop** | interrupt(), Command(resume=), interrupt_before, approval workflows |
| **7. Checkpointing** | MemorySaver, SqliteSaver, PostgresSaver, time travel, branching |
| **8. Production** | FastAPI integration, SSE streaming, multi-tenancy, PostgreSQL, monitoring |

### When to Choose LangGraph

```
Simple single-pass pipeline?         → LCEL chains
Need loops / retry logic?            → LangGraph
Need conditional branching?          → LangGraph
Need human approval steps?           → LangGraph (interrupt)
Need persistent sessions?            → LangGraph (checkpointer)
Need multiple specialized agents?    → LangGraph (multi-agent)
Need state to survive server restart? → LangGraph (PostgresSaver)
```

---

*End of Part 10 — LangGraph*
