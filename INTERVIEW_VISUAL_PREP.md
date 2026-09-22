# Visual Tech Explainer — Baltic Agent Systems Interview
> "Assume I know nothing" edition. Every technology: what it is, why it exists, why YOU chose it, what the alternative was, what the limitation is, and what to say out loud.

---

## HOW TO USE THIS IN 4 HOURS

- Read one section at a time. Each section is self-contained.
- The **💬 Say this out loud** lines are your verbal answers — practice them.
- The **⚠️ Limitation** lines are for when they probe deeper.
- The **🆚 Why not X** boxes are for comparative questions.

---

# SECTION 1 — THE BIG PICTURE: What is RAG?

## The analogy that always works

Imagine you're on a quiz show. The question is: *"What projects has Pragna done?"*

Two options:
1. **Without RAG**: Ask a random person on the street. They'll make something up that sounds plausible. This is a regular LLM — it hallucinates.
2. **With RAG**: Hand that person a folder of Pragna's actual notes, then ask them to answer **using only what's in the folder**. They can't make things up. This is RAG.

**RAG = Search Engine + Language Model**
- The search engine finds the right notes (retrieval)
- The language model writes a natural-language answer from those notes (generation)

```
QUESTION → [Find relevant notes] → [LLM reads notes + writes answer] → ANSWER
             ^^^^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^
             That's "Retrieval"                 That's "Generation"
```

## Why RAG instead of fine-tuning?

| | RAG (what you built) | Fine-tuning (what you didn't) |
|---|---|---|
| **Update knowledge** | Edit .md file, restart | Retrain the model (days + $$$) |
| **Cost** | Free to update | Expensive GPU time |
| **Swap LLM** | One env var change | Start over |
| **Explain what it retrieved** | Yes — citations shown | Black box |
| **Risk of making things up** | Low (grounded in your notes) | Higher (model "remembers" vaguely) |

**💬 Say this out loud:**
> "I chose RAG over fine-tuning because my knowledge base changes — I add projects, update opinions. With RAG I edit a Markdown file and restart. With fine-tuning I'd need a new training run every time. For a portfolio, RAG is the only sensible choice."

---

# SECTION 2 — EMBEDDINGS

## The analogy

Words are meaningless to computers. Numbers are not. Embeddings convert words into numbers in a specific way: **similar meaning = similar numbers**.

Think of a map. London and Paris are physically close. Tokyo is far away. Embeddings create a "meaning map" where:
- "What's your work experience?" and "Tell me about your background?" are **close** on the map
- "What's your experience?" and "What's the weather in Tokyo?" are **far** on the map

Your embedding model (`all-MiniLM-L6-v2`) converts any sentence into **384 numbers** (a "vector"). Two sentences with similar meaning get similar vectors.

## What does 384 dimensions mean?

A single number = 1 dimension (a line).
Two numbers = 2 dimensions (an X,Y coordinate on a map).
384 numbers = 384-dimensional space (impossible to visualise, but the math works the same).

Each of the 27 chunks in your knowledge base is stored as 384 numbers. Each incoming question is also converted to 384 numbers. Then you measure "which chunk is closest?"

## Why `all-MiniLM-L6-v2`?

| Model | Size | Speed | Quality | Your use case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` ✅ | 80 MB | Very fast on CPU | Good | ✅ Portfolio, CPU-only |
| `all-mpnet-base-v2` | 420 MB | Slower | Better | Overkill for 27 chunks |
| OpenAI `text-embedding-3-small` | API call | Fast | Excellent | $0.02 per 1M tokens — adds cost + dependency |

**💬 Say this out loud:**
> "The MiniLM model runs on CPU in milliseconds, takes 80MB in the Docker image, and was pretrained on over a billion sentence pairs. For 27 chunks, it's more than sufficient. The quality uplift from a larger model would be invisible at this scale."

**⚠️ Limitation:**
> "The model doesn't update. If I add very domain-specific jargon it hasn't seen, the embeddings won't capture the meaning well. For that, a domain-fine-tuned embedder or hybrid search (semantic + keyword) would help."

---

# SECTION 3 — COSINE SIMILARITY (the retrieval math)

## The analogy

Two arrows drawn from the same centre point. If they point in the same direction → score = 1.0. If they're at 90° to each other → score = 0. If they point opposite → score = -1.0.

Your 384-number vector IS an arrow. Finding the "most similar chunk" = finding the chunk whose arrow points most in the same direction as the question's arrow.

```
Question vector:  [0.1, 0.8, 0.2, ..., 0.5]   ← 384 numbers
Chunk 1 vector:   [0.1, 0.7, 0.3, ..., 0.4]   ← similarity: 0.92 ✅ (top hit)
Chunk 2 vector:   [0.9, 0.1, 0.1, ..., 0.2]   ← similarity: 0.21 ❌
```

## How it works in the code

```python
# In RAGIndex.retrieve():
scores = (norm_matrix @ qn.T).ravel()   # Matrix multiply = cosine similarity for all 27 chunks at once
idx = np.argsort(-scores)[:k]           # Sort highest to lowest, take top-k
```

`@` is NumPy's matrix multiply. It computes the similarity of the question against ALL 27 chunks simultaneously — one operation, sub-millisecond.

**💬 Say this out loud:**
> "Cosine similarity measures the angle between two vectors. The NumPy matrix multiply computes it against all chunks in one shot. For 27 chunks it's instant. Known limitation: I re-normalise the matrix on every query call — I should pre-normalise it once at startup."

**🆚 Why NumPy, not a vector database (FAISS, Pinecone, Qdrant)?**
> "At 27 chunks, a vector database adds a dependency, a service to run, and a network call for zero performance benefit. NumPy is already in the dependency list. I'd switch to pgvector at ~1,000+ chunks — and the docker-compose.yml already has the pgvector image ready."

---

# SECTION 4 — FASTAPI

## The analogy

FastAPI is the front door of your building. When a request arrives (someone typing a question), FastAPI:
1. Looks at the URL path (`/api/chat`)
2. Routes it to the right room (the `chat()` function)
3. Checks the ID at the door (Pydantic validates the input)
4. Takes the response back and hands it out

## Why FastAPI specifically?

| Framework | Async | Auto docs | Validation | Speed | Your choice |
|---|---|---|---|---|---|
| **FastAPI** ✅ | ✅ Native | ✅ Auto | ✅ Pydantic | ⚡ Very fast | ✅ |
| Flask | ❌ (add-on) | ❌ Manual | ❌ Manual | Medium | Too simple |
| Django | ❌ (ASGI add-on) | ❌ Manual | ORM only | Slower | Too heavy |
| Express (Node.js) | ✅ | ❌ | ❌ Manual | Fast | Wrong language |

**💬 Say this out loud:**
> "FastAPI's async model means while the LLM is generating (which takes 3–8 seconds), the server can handle other requests. Flask would block. Django REST Framework is designed for CRUD apps, not streaming AI APIs. FastAPI was built for exactly this use case."

## What is `async def` / `await`?

Imagine a single waiter at a restaurant.
- **Synchronous (bad)**: Waiter takes order → stands in kitchen watching food cook → brings it out. No other tables served.
- **Asynchronous (what you built)**: Waiter takes order → goes to the kitchen → while food cooks, serves other tables → picks up food when ready.

`async def chat()` = this function can pause when waiting for the LLM and let other requests run. `await llm.stream(prompt)` = "pause here while waiting for the LLM response."

---

# SECTION 5 — PYDANTIC

## The analogy

Pydantic is the customs officer at an airport. Before any data enters your code:
- Is it the right type? (string, not a number)
- Is it the right length? (max 2000 chars)
- Is it a valid value? (k must be 1–8, not 100)

If not → instant rejection with a clear error message. Your code never sees bad data.

```python
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)   # ← Pydantic enforces this
    k: int = Field(default=4, ge=1, le=8)                      # ← ge=greater-or-equal, le=less-or-equal
```

If someone sends `k=500` → Pydantic returns a 422 error before your function even runs.

**💬 Say this out loud:**
> "Pydantic v2 is written in Rust, so validation is extremely fast. It's not just type checking — it's contract enforcement at the boundary of my system. Bad data never reaches my retrieval or LLM logic."

---

# SECTION 6 — SERVER-SENT EVENTS (SSE)

## The analogy

Imagine two ways to get sports scores:
1. **REST/polling**: You call the stadium every 10 seconds: "What's the score?" This is inefficient.
2. **SSE**: You call once and the stadium keeps sending you updates as they happen. That's SSE.

Your chatbot uses SSE. The browser makes ONE request, and the server sends events over that connection for the entire answer duration.

## Your event structure

```
event: citations
data: {"citations": [{"title": "Who I am", "score": 0.87}, ...]}

event: token
data: {"text": "I"}

event: token
data: {"text": " built"}

... (one event per word/token)

event: done
data: {"latency_ms": 2340, "provider": "groq"}
```

The browser's JavaScript loops: `while (true) { const chunk = await reader.read(); append to DOM; }`

**Why citations FIRST?** Users see proof of retrieval before the answer starts generating. This is a UX and transparency decision — it shows how RAG works.

**🆚 SSE vs WebSocket:**
| | SSE | WebSocket |
|---|---|---|
| Direction | Server → Client only | Both ways |
| Complexity | Simple (plain HTTP) | Handshake, stateful |
| Your use case | LLM streams words to browser | Not needed |
| **Winner for chat** | ✅ SSE | ❌ Overkill |

**💬 Say this out loud:**
> "SSE is one-directional — server to client. For chat, the client sends one question and receives one streaming response. That's SSE's exact use case. WebSocket would add complexity (connection state management, heartbeats) with no benefit."

---

# SECTION 7 — LLM PROVIDERS (Groq, Anthropic, Ollama, Echo)

## The Protocol pattern (structural typing)

Instead of making all LLMs inherit from one class (inheritance), Python's `Protocol` says: "I don't care what family you come from. If you have a `.name` and an `async stream()` method, you qualify."

Think of a power outlet. Different devices (laptop, phone, lamp) all have different internals. But as long as the plug fits the socket, it works. The socket doesn't care what's inside.

```
LLM Protocol = the socket shape
AnthropicLLM, GroqLLM, OllamaLLM, EchoLLM = different devices with the right plug
```

## Why each provider exists

| Provider | What it is | Why in the code |
|---|---|---|
| **Groq** | Custom chip (LPU) that runs Llama 3.3 70B absurdly fast | Production default — fast + cheap |
| **Anthropic** | Hosted LLM models | Quality fallback |
| **Ollama** | Runs open-source models LOCALLY | Zero cost, zero internet needed |
| **Echo** | Returns the retrieved context as-is | Testing without API keys |

**💬 Say this out loud:**
> "The pluggable design means adding a fifth provider — say, Google Gemini — is 20 lines. I add a class with `.name` and `stream()`, add 'gemini' to the config literal, and one branch in the factory. No other file changes."

**⚠️ Real limitation:**
> "The current Protocol takes a single prompt string. Anthropic's API has a dedicated `system` parameter that gives the model stronger grounding. To fix this, I'd refactor the Protocol to accept `system` and `user` separately."

---

# SECTION 8 — DOCKER (multi-stage build)

## The analogy

Building a house:
- **Stage 1 (Builder)**: Construction site. All the heavy machinery, lumber, tools. Messy.
- **Stage 2 (Runtime)**: The finished house. Clean, minimal, only what you need to live in.

The Docker build downloads packages and ML models (heavy, messy). The runtime image copies only the finished result. The construction machinery never ships to the user.

## Why bake ML models into the image?

If models download at container startup:
1. Cold start takes 30–60 seconds (Railway kills it as unhealthy)
2. If Hugging Face is down, your app is down
3. Every new instance re-downloads models

With baked models:
1. Startup = instant (model is already on disk)
2. No internet needed at runtime
3. The image layer is immutable and cached

```dockerfile
# Builder stage — downloads everything
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Runtime stage — just copies the result
COPY --from=builder /root/.local /home/app/.local
```

## The non-root user

`RUN useradd -m -u 1000 app` + `USER app`

Why? If a hacker exploits a vulnerability in your app, they get access as the `app` user — not as `root`. Root access = full machine control. App user = limited damage.

**💬 Say this out loud:**
> "Two-stage build keeps the image small — no build tools in production. Models are baked in so startup is instant and there's no runtime dependency on Hugging Face. Non-root user is standard container security."

---

# SECTION 9 — SECURITY

## The four layers

### Layer 1: Rate Limiting (slowapi)
**What**: A counter per IP address. Chat endpoint: 20 requests/minute. If you hit 21, you get a 429 error.

**Why**: Without it, someone could write a script that calls `/api/chat` 10,000 times and cost you hundreds of dollars in LLM API fees.

### Layer 2: Input Sanitisation
**What**: Before your message reaches the LLM, regex patterns scan it for phrases like "ignore all previous instructions" and replace them with `[redacted]`.

**Why**: Prompt injection — an attacker tries to override your system prompt to make the chatbot say something harmful, leak secrets, or impersonate someone else.

**Limitation to state honestly:**
> "Regex patterns block the obvious attacks. A sophisticated adversary using paraphrasing or multi-step injection can still get through. Full defence requires output filtering, a moderation layer, and LLM self-awareness — this is baseline hygiene, not a complete solution."

### Layer 3: Security Headers
```
X-Frame-Options: DENY           → stops your site being embedded in an iframe (clickjacking)
X-Content-Type-Options: nosniff → stops browsers guessing file types (MIME sniffing attacks)
Content-Security-Policy         → tells browsers: only run scripts from MY domain
Referrer-Policy                 → controls what URL info is sent to external sites
Permissions-Policy              → blocks the page from accessing camera/mic/location
```

### Layer 4: CORS + Trusted Hosts
CORS: Only allow requests from known origins in production (not from random.com).
Trusted hosts: Reject requests with unknown `Host:` headers — stops certain server-side request forgery attacks.

**💬 Say this out loud:**
> "All security code is in one file — `security.py`. That's a deliberate design choice. Security posture should be reviewable in one read, not scattered across the codebase. A new engineer should be able to open one file and understand the entire attack surface."

---

# SECTION 10 — DATABASE (SQLAlchemy 2.0 async)

## What you store and WHY

```python
class Conversation(Base):
    question:    str   # The cleaned question (never the raw injection text)
    answer:      str   # Full LLM response
    citations:   str   # JSON list of which chunks were retrieved
    latency_ms:  int   # How long the full response took
    provider:    str   # Which LLM answered (groq/anthropic/echo)
```

**Why log this?**
- You can see what visitors are actually asking → improve the knowledge base
- You can see when retrieval fails → add the right chunk
- You can see latency → know if Groq is slower on certain queries
- You never log IP addresses or user identity → privacy by design

**Why async SQLAlchemy?**
Without async: writing a DB row freezes the entire server for 5–10ms.
With async: writing happens in the background while the server handles other requests.

**💬 Say this out loud:**
> "Logging is intentionally best-effort — if the DB write fails, chat still works. The `_try_log_conversation()` function catches all exceptions and swallows them silently. Logging should never break the main user flow."

---

# SECTION 11 — THE KNOWLEDGE BASE (knowledge_base.md)

## Why Markdown with `## CHUNK:` markers?

**Option A: Sliding window chunking** — split every 500 characters, overlapping by 50. Common approach.
**Problem**: A 3-sentence story might be split mid-sentence. The retrieved chunk is half a thought.

**Option B: Your approach** — semantic chunks with explicit `## CHUNK:` markers.
**Benefit**: Each chunk is a complete, coherent thought. The "Bank of America story" chunk contains the entire story, not half of it.

```markdown
## CHUNK: A story from Bank of America
Built a fraud-detection integration. The hard part wasn't the JWT plumbing...
```

The regex `r"^##\s*CHUNK:\s*(.+)$"` finds all these markers and splits the document at each one.

**💬 Say this out loud:**
> "I own the chunking boundaries. I know that 'Who I am' should be one chunk, 'Bank of America story' another. Sliding window would mix them. Explicit markers give me full control over what the retrieval unit contains."

---

# SECTION 12 — THE AI CODING QUESTION (the full answer)

## What they're actually asking

They're not asking "did you type every character." They're asking: **"Do you understand what you built, or are you a prompt-copier who can't extend it?"**

## The honest, confident three-part answer

**Part 1 — Acknowledge honestly:**
> "Yes, I used AI as a coding tool. The same way every engineer uses Stack Overflow, documentation, compiler hints, and code review. The question isn't whether tools were used — it's whether I understand the output."

**Part 2 — Prove ownership with specifics:**
> "Here's what I mean by understanding: I know there's a dead-code bug on line 65 of `main.py` — the docs-enabled logic reduces to just `settings.api_docs_enabled` regardless of environment. I know the system prompt goes as a user message, not in Anthropic's `system` field, and why that's suboptimal. I know cosine similarity is re-normalised on every query call and should be pre-computed. I know exactly why I have 27 chunks and why adding a 28th requires understanding retrieval, not just adding a Markdown section."

**Part 3 — Reframe the question:**
> "The value I add is architectural judgment. AI can generate a FastAPI endpoint. AI doesn't decide whether to use SSE or WebSocket, whether to use NumPy or pgvector, whether to bake models into Docker or download at runtime — and explain why with tradeoff analysis. I made those decisions. I can defend all of them."

## If they push harder: "But if AI writes the code, why hire you?"

Point to decisions that require understanding:
1. Not using LangChain — and the exact reasoning (dependency cost, debuggability)
2. All-MiniLM-L6-v2 — and why not a larger model (CPU, startup time, Docker size)
3. 27 semantic chunks vs sliding-window — and why (coherent retrieval units)
4. Security in one file — and why (auditability)
5. SSE not WebSocket — and why (one-directional, simpler)
6. Echo provider — and why (testability without API keys)
7. Best-effort DB logging — and why (chat can't fail because of logging)
8. Citations streamed FIRST — and why (UX + RAG transparency)

Each of these is a judgment call that requires understanding the system. Copying AI output doesn't give you these.

---

# QUICK-REFERENCE CHEATSHEET

| If they ask about... | Key phrase to use |
|---|---|
| RAG | "Search engine + language model. The LLM reads your notes, doesn't invent." |
| Embeddings | "Converts meaning into 384 numbers. Similar meaning = similar numbers." |
| Cosine similarity | "Angle between two vectors. Closest angle = most relevant chunk." |
| Why not LangChain | "200+ deps, abstraction I can't debug, 175 lines solved this without it." |
| Why NumPy not pgvector | "27 chunks. Sub-millisecond. pgvector is ready in docker-compose for scale." |
| FastAPI vs Flask | "Flask has no native async. Streaming LLM calls need async." |
| SSE vs WebSocket | "Chat is one-directional. SSE is simpler with identical UX." |
| Multi-stage Docker | "Small image, instant startup, no Hugging Face dependency at runtime." |
| Rate limiting | "20 req/min per IP on chat — prevents API cost abuse." |
| Prompt injection | "Regex baseline hygiene. Not bulletproof — output filtering is the next layer." |
| Did AI write this | "AI is a tool. I made the architectural decisions. I can defend all of them." |
| Why not fine-tuning | "Expensive, drifts, locked in. RAG: edit a file, restart, done." |
| System prompt placement | "Current limitation — sent as user message, not Anthropic's `system` field." |

---

*Good luck. You built something real and you understand it. That combination is rarer than it sounds.*
