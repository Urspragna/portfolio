# 🎓 AI Portfolio Learning Guide for Beginners

> **Start here** if you're new to AI, Python web apps, or this codebase. Every term explained, every step shown.

---

## Table of Contents
1. [Core Concepts](#core-concepts) — Key AI terms explained
2. [How to Start](#how-to-start) — First 10 minutes
3. [The RAG Pipeline](#the-rag-pipeline) — The heart of this project
4. [Code Walkthrough](#code-walkthrough) — Reading the actual files
5. [Experiments to Try](#experiments-to-try) — Hands-on learning
6. [Deploy to Production](#deploy-to-production) — Get a live URL
7. [Glossary](#glossary) — Reference for all terms

---

## Core Concepts

### What is RAG?

**RAG = Retrieval-Augmented Generation**

Imagine you're asking an AI chatbot: *"What is Pragna's experience with fraud detection?"*

Without RAG, the AI would guess from its training data (and probably get it wrong).

**With RAG**, the system does this:
1. **Retrieve** — Search through Pragna's CV to find relevant passages
2. **Augment** — Glue those passages into the prompt as context
3. **Generate** — Ask the LLM to answer *based on that context only*

**Real-world analogy:** You're studying for an exam. Instead of relying on what you memorized, you look up the answer in your textbook first, then explain it in your own words. That's RAG.

**Why RAG?**
- ✅ Answers are grounded in real data (no hallucination)
- ✅ Easy to update (just edit the knowledge base file)
- ✅ Works with any LLM (Claude, Ollama, GPT, etc.)
- ✅ Cheap (no expensive fine-tuning)

---

### What is an LLM?

**LLM = Large Language Model**

A neural network trained on billions of words. It's really good at predicting the next word in a sequence.

**Examples:**
- Claude (Anthropic) — smart, safe, costs money
- GPT-4 (OpenAI) — very smart, costs money
- Llama 3 (Meta, via Ollama) — decent, free, runs locally on your computer

**In this project:** You can swap LLMs with one environment variable. No code changes needed.

---

### What is an Embedding?

**Embedding = A numeric representation of meaning**

Words and sentences are turned into lists of numbers that capture *meaning*.

**Example:**
```
"Pragna works with Python" → [0.12, -0.43, 0.91, 0.22, ...]  (384 numbers)
"Pragna codes in Python"   → [0.11, -0.44, 0.89, 0.23, ...]  (similar!)
"Llamas are fluffy"        → [0.02, 0.15, -0.88, -0.12, ...] (very different)
```

**Why embeddings?**
- Similar sentences have similar embeddings
- You can measure distance: `distance(sent1, sent2)` = how similar they are
- This is how RAG *retrieves* the right context

**In this project:**
- Uses `sentence-transformers` (open-source, 384-dim vectors)
- Falls back to hash-based embeddings if the model can't download

---

### What is FastAPI?

**FastAPI = A Python web framework**

It lets you build web servers that:
- Handle HTTP requests (GET, POST, etc.)
- Return JSON responses
- Auto-generate interactive API documentation
- Run asynchronously (handle many requests at once)

**Real-world analogy:** Think of it as a butler. You define what endpoints exist (`POST /api/chat`), and FastAPI routes incoming requests to the right handler.

**In this project:**
```python
@router.post("/api/chat")  # When someone POSTs to /api/chat, this runs
async def chat(body: ChatRequest) -> StreamingResponse:
    ...
```

---

### What is Async?

**Async = Asynchronous programming**

Normally, code runs line-by-line. If one line takes 2 seconds (waiting for a database), the whole program pauses.

With async, the program can do something else while waiting.

**Example:**
```python
# Without async (slow)
response = call_llm()  # Wait 3 seconds...
send_to_db()           # Wait 1 second...
# Total: 4 seconds

# With async (fast)
async def main():
    response = await call_llm()  # Start, don't wait
    send_to_db()                 # Do this while waiting
    # Total: 3 seconds (happens in parallel!)
```

**In this project:** FastAPI uses async so multiple users can chat at once without blocking each other.

---

### What is Streaming (SSE)?

**SSE = Server-Sent Events**

Instead of waiting for the whole answer, the server *streams* words as they come.

**Example:**
```
User:     "Tell me about Pragna's RAG experience"
Server:   *waits for LLM*
LLM:      "Pragna built a..." (yields first chunk)
Browser:  Shows: "Pragna built a..."
LLM:      "...from-scratch RAG..." (yields next chunk)
Browser:  Shows: "Pragna built a...from-scratch RAG..."
(and so on, like ChatGPT/Claude in the browser)
```

**Why streaming?**
- UX feels instant (words appear as they generate)
- Better for slow networks
- Matches what users expect from ChatGPT

---

### What is a Vector Database?

**Vector DB = A database optimized for storing and searching embeddings**

Regular databases search by exact match (`WHERE name = "Pragna"`).

Vector DBs search by *similarity* (`find embeddings closest to this embedding`).

**In this project:**
- **Dev:** Embeddings stored in memory (fast, simple)
- **Prod:** PostgreSQL + pgvector extension (persistent, scales)

---

### What is Pydantic?

**Pydantic = A Python library for data validation**

Ensures incoming data is the right shape before your code runs.

**Example:**
```python
class ChatRequest(BaseModel):
    message: str  # Must be a string
    k: int = 4    # Must be an integer, defaults to 4

# If frontend sends: {"message": "hi", "k": "wrong"}
# Pydantic raises an error before your code sees it
```

**Why Pydantic?**
- Type safety (catch bugs early)
- Auto-generate API docs
- Serialize/deserialize cleanly

---

### What is SQLAlchemy?

**SQLAlchemy = A Python ORM for databases**

ORM = Object-Relational Mapping. It translates between Python objects and database rows.

**Example:**
```python
# Instead of writing SQL:
# INSERT INTO conversations (user_id, message) VALUES (1, "hello")

# You write Python:
conversation = Conversation(user_id=1, message="hello")
session.add(conversation)
session.commit()
```

**In this project:**
- Logs chat history (optional)
- Works with SQLite (dev) or PostgreSQL (prod)
- Uses async sessions for non-blocking DB calls

---

## How to Start

### Step 1: Set Up (2 minutes)

```bash
# 1. You're already in the right directory (it's already activated above)
# Verify Python is ready:
python --version  # Should be 3.11+

# 2. Install dependencies
pip install -r requirements.txt

# This installs all the libraries you need:
# - fastapi, uvicorn (web server)
# - sentence-transformers (embeddings)
# - sqlalchemy (database)
# - pydantic (validation)
# - torch, transformers (AI models)
# - and more...
```

### Step 2: Run the App (1 minute)

```bash
uvicorn app.main:app --reload
```

You'll see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 3: Explore (5 minutes)

| URL | What it is |
|-----|-----------|
| `http://localhost:8000` | Portfolio homepage with chat widget |
| `http://localhost:8000/lab` | AI Learning Lab (4 live demos) |
| `http://localhost:8000/docs` | Interactive API docs (Swagger UI) |

**Try it:**
1. Open `http://localhost:8000` in your browser
2. Type: *"What does Pragna do?"*
3. Watch it retrieve context and answer

---

## The RAG Pipeline

This is where the magic happens. Let's trace a question from input to answer.

### Question: *"What's Pragna's experience with fraud detection?"*

#### Step 1: CHUNK (app/rag.py)

The knowledge base (`data/knowledge_base.md`) is split into logical units:

```markdown
## CHUNK: A story from Bank of America
The fraud-detection win at BoA wasn't one dramatic moment — it was a year 
of slow integration...
```

Why chunking?
- If you chunk by character count, a sentence might split in half ❌
- If you chunk by heading, you get coherent topics ✅

**Code:**
```python
def chunk_markdown(md: str) -> list[Chunk]:
    matches = list(CHUNK_RE.finditer(md))  # Find all "## CHUNK:" headings
    chunks = []
    for m in matches:
        # Extract title and text between this chunk and the next
        chunks.append(Chunk(title=title, text=text))
    return chunks
```

#### Step 2: EMBED (app/rag.py)

Each chunk is converted to a 384-dimensional vector using `sentence-transformers`:

```
Chunk: "The fraud-detection win at BoA..."
         ↓
Embedder.encode(chunk.text)
         ↓
Vector: [0.12, -0.43, 0.91, ..., 0.22]  (384 numbers)
```

These vectors are stored in memory (or pgvector in production).

**Code:**
```python
class Embedder:
    def encode(self, texts: list[str]) -> np.ndarray:
        # Uses SentenceTransformer or falls back to hashing
        test = self._sbert.encode(["probe"], convert_to_numpy=True)
        # Returns: numpy array of shape (n_texts, 384)
```

#### Step 3: RETRIEVE (app/rag.py)

Your question is *also* embedded, then compared to all chunk embeddings:

```
User question: "What's Pragna's experience with fraud detection?"
                ↓
Embedder.encode("What's Pragna's experience with fraud detection?")
                ↓
Question vector: [0.10, -0.45, 0.88, ..., 0.24]
                ↓
Compare to all chunk vectors using cosine similarity
                ↓
Top 4 most similar chunks:
  1. "A story from Bank of America" (similarity: 0.94)
  2. "My engineering philosophy" (similarity: 0.71)
  3. "What I'm working on this week" (similarity: 0.68)
  4. ... (similarity: 0.55)
```

**Code:**
```python
def retrieve(self, query: str, k: int = 4) -> list[Hit]:
    # Embed the query
    q_vec = self.embedder.encode([query])[0]  # Get first result
    
    # Compare to all chunks using cosine similarity
    scores = np.dot(self.vectors, q_vec)  # Dot product = similarity
    
    # Get top-k
    top_k_idx = np.argsort(scores)[-k:][::-1]  # Sort descending
    return [Hit(chunk=self.chunks[i], score=scores[i]) for i in top_k_idx]
```

#### Step 4: AUGMENT (app/rag.py)

The retrieved chunks are stitched into a prompt:

```
System: You are a helpful assistant answering questions about Pragna's background.

Context passages:
- [Bank of America] The fraud-detection win at BoA...
- [Engineering philosophy] The boring parts — logging, RBAC...

Question: What's Pragna's experience with fraud detection?

Answer:
```

This is the full prompt sent to the LLM.

**Code:**
```python
def build_prompt(question: str, hits: list[Hit], max_chars: int) -> str:
    context_lines = []
    for hit in hits:
        context_lines.append(f"[{hit.chunk.title}] {hit.chunk.text}")
    
    context = "\n".join(context_lines)[:max_chars]  # Trim if too long
    
    return f"""System: You are helpful...

Context passages:
{context}

Question: {question}

Answer:"""
```

#### Step 5: GENERATE (app/llm.py)

The prompt is sent to an LLM:

```python
class AnthropicLLM:
    async def stream(self, prompt: str) -> AsyncIterator[str]:
        # Send to Anthropic Claude API
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", ..., json={"prompt": prompt}) as r:
                async for chunk in r.aiter_text():
                    yield chunk  # Stream each token as it arrives
```

#### Step 6: STREAM TO USER (app/routers/chat.py)

Each token is sent to the browser as a server-sent event:

```python
@router.post("/api/chat")
async def chat(body: ChatRequest):
    hits = index.retrieve(body.message, k=body.k)
    prompt = build_prompt(body.message, hits)
    
    async def event_stream() -> AsyncIterator[bytes]:
        # Send citations first
        yield _sse("citations", {"citations": citations})
        
        # Then stream the answer token-by-token
        async for token in llm.stream(prompt):
            yield _sse("token", {"text": token})
    
    return StreamingResponse(event_stream())
```

Browser receives events and displays them:
```javascript
const eventSource = new EventSource("/api/chat");
eventSource.addEventListener("token", (e) => {
    const {text} = JSON.parse(e.data);
    // Append text to screen
});
```

---

## Code Walkthrough

### File: `app/main.py`

**What it does:** Wires the whole app together.

```python
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup, once on shutdown."""
    print("[app] starting up")
    
    # 1. Initialize database
    await init_db()
    
    # 2. Build the RAG index from knowledge_base.md
    chunks = chunk_markdown(load_file("data/knowledge_base.md"))
    embeddings = Embedder(settings).encode([c.text for c in chunks])
    index = RAGIndex(chunks, embeddings)
    app.state.rag_index = index
    
    # 3. Instantiate the LLM (Claude, Ollama, or Echo)
    app.state.llm = make_llm(settings)
    
    yield  # App runs here
    
    print("[app] shutting down")

app = FastAPI()
# Mount static files (index.html, lab.html)
app.mount("/", StaticFiles(directory=".", html=True))
# Include routers (chat, demos)
app.include_router(chat_router)
app.include_router(demos_router)
```

**Key takeaway:** Startup code builds the index *once*, then reuses it for all requests.

---

### File: `app/config.py`

**What it does:** Environment-based configuration using Pydantic.

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    environment: str = "development"
    llm_provider: str = "echo"  # "anthropic" or "ollama" or "echo"
    anthropic_api_key: str | None = None
    embedding_provider: str = "sbert"
    
    class Config:
        env_file = ".env"  # Reads from .env file
```

**How to use:**
```bash
# Create .env
echo "LLM_PROVIDER=anthropic" > .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Now run the app
uvicorn app.main:app --reload
# Settings will read from .env
```

---

### File: `app/rag.py`

**What it does:** The entire RAG pipeline (all 6 steps above).

Key classes:
- `Chunk` — A retrieval unit (title + text)
- `Embedder` — Turns text into vectors
- `RAGIndex` — Stores chunks + embeddings, does retrieval
- `Hit` — A retrieved chunk + similarity score

Key functions:
- `chunk_markdown()` — Split knowledge base
- `build_prompt()` — Assemble the RAG prompt

---

### File: `app/routers/chat.py`

**What it does:** `POST /api/chat` endpoint that streams answers.

```python
@router.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """
    Input: {"message": "...", "k": 4}
    Output: Server-Sent Events stream of tokens
    """
    index = request.app.state.rag_index  # Loaded on startup
    llm = request.app.state.llm
    
    # 1. Retrieve top-k chunks
    hits = index.retrieve(body.message, k=body.k)
    
    # 2. Build prompt
    prompt = build_prompt(body.message, hits)
    
    # 3. Stream from LLM
    async def event_stream():
        yield _sse("citations", {"citations": citations})
        async for token in llm.stream(prompt):
            yield _sse("token", {"text": token})
    
    return StreamingResponse(event_stream())
```

---

### File: `app/routers/demos.py`

**What it does:** Four ML demo endpoints for learning.

```python
@router.get("/api/demo/embed")      # Embedding demo
@router.get("/api/demo/sentiment")  # Sentiment analysis demo
@router.get("/api/demo/tokenise")   # Tokenization demo
@router.get("/api/demo/retrieve")   # RAG retrieval demo
```

---

### File: `app/db.py`

**What it does:** Database models and session management.

```python
class Conversation(Base):
    """Stores chat history (optional)."""
    __tablename__ = "conversations"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str]
    message: Mapped[str]
    response: Mapped[str]
    created_at: Mapped[datetime]
```

---

### File: `data/knowledge_base.md`

**What it does:** The content that gets RAG-retrieved.

Each chunk is marked with `## CHUNK: <title>`:

```markdown
## CHUNK: Who I am, in my own voice
I'm Pragna. I'm a software engineer who cares about...

---

## CHUNK: A story from Bank of America
The fraud-detection win at BoA wasn't one dramatic moment...
```

**To update the chatbot:**
1. Edit this file
2. Restart the app (`uvicorn app.main:app --reload`)
3. The index rebuilds automatically

---

## Experiments to Try

### Experiment 1: Add a New Chunk

**Goal:** See how the chatbot immediately learns new information.

1. Open `data/knowledge_base.md`
2. Add a new chunk:
   ```markdown
   ## CHUNK: My favorite programming language
   Python is my favorite because it's readable, has great libraries, 
   and gets out of your way when you need to ship.
   
   ---
   ```
3. Save the file
4. Restart the app (stop and re-run `uvicorn app.main:app --reload`)
5. Ask the chatbot: *"What's your favorite programming language?"*
6. It will now retrieve your new chunk and answer

**What you learn:** RAG is updatable in real-time (unlike fine-tuning).

---

### Experiment 2: Change the LLM

**Goal:** See that switching LLMs requires only a config change.

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. **Option A: Use Claude (requires API key)**
   ```bash
   # Get key from https://console.anthropic.com
   echo "LLM_PROVIDER=anthropic" >> .env
   echo "ANTHROPIC_API_KEY=sk-ant-YOUR_KEY_HERE" >> .env
   ```

3. **Option B: Use Ollama (free, local)**
   ```bash
   # First, install Ollama: https://ollama.com
   # Then:
   ollama pull llama3.2
   ollama serve
   
   # In another terminal:
   echo "LLM_PROVIDER=ollama" >> .env
   ```

4. Restart the app
5. Ask the chatbot a question
6. Notice the different quality/style of answers

**What you learn:** LLM abstraction lets you swap providers without code changes.

---

### Experiment 3: Inspect Embeddings

**Goal:** See what embeddings actually look like.

1. Open `http://localhost:8000/lab` (AI Lab)
2. Click "Embedding Demo"
3. Type: *"Pragna codes"*
4. See the 384-dimensional vector produced

**What you learn:** Text is converted to numbers, and similar text → similar numbers.

---

### Experiment 4: Debug Retrieval

**Goal:** See what context was retrieved for your question.

1. Open browser DevTools (F12)
2. Go to Console tab
3. Ask the chatbot a question
4. In DevTools, you'll see SSE events:
   ```javascript
   {
     event: "citations",
     data: {
       citations: [
         { title: "Who I am, in my own voice", score: 0.92 },
         { title: "...", score: 0.87 }
       ]
     }
   }
   ```

**What you learn:** You can see exactly what chunks were retrieved and their similarity scores.

---

### Experiment 5: Modify the System Prompt

**Goal:** Change the chatbot's behavior.

1. Open `app/rag.py`
2. Find the `build_prompt()` function
3. Change the system prompt:
   ```python
   return f"""You are a sarcastic assistant answering questions about Pragna.

Context passages:
{context}

Question: {question}

Answer:"""
   ```
4. Restart the app
5. Ask the chatbot: *"Tell me about Pragna"*
6. Notice the sarcastic tone

**What you learn:** System prompts shape LLM behavior without changing the underlying RAG.

---

## Deploy to Production

### Overview: Three Deployment Options

| Option | Difficulty | Cost | Time | Best For |
|--------|-----------|------|------|----------|
| **Railway** | ⭐ (easiest) | Free tier available | 5 min | Quick MVP, portfolio demo |
| **Render** | ⭐ (easiest) | Free tier available | 5 min | Quick MVP, portfolio demo |
| **AWS/DigitalOcean** | ⭐⭐⭐ (harder) | $5-30/mo | 30 min | Production, scaling |

**I recommend starting with Railway or Render** — they handle Docker deployment automatically, and your app is already Dockerized.

---

### 🎯 FREE & EFFICIENT: Best Option for You

**TL;DR:** Use **Render's free tier** for permanent hosting (always-on), or **Railway** if you want a bit more generosity.

#### Why Render's Free Tier?
- ✅ **Always-on** (never sleeps, unlike some platforms)
- ✅ **Permanent** (truly free, no credit card needed)
- ✅ **Simple** (5 minutes to deploy)
- ✅ **Good for portfolios** (your chatbot runs 24/7)

#### Render Free Tier Limits
- 750 compute hours/month (basically unlimited for one always-on app)
- 100GB bandwidth/month (plenty for a chatbot)
- PostgreSQL available (up to 1GB storage free)

#### Why Railway Second?
- More generous free tier ($5/month credit)
- But free tier can spin down if unused
- Great for testing multiple deployments

#### Why NOT the Others?
- **Heroku** — Removed free tier (now $7+/month minimum)
- **AWS** — Confusing free tier, easy to accidentally get charged
- **DigitalOcean** — $5/month minimum (not free)
- **Vercel** — Designed for frontend, not backend APIs

---

### FASTEST FREE OPTION: Render + SQLite (5 minutes)

This is the absolute fastest way to get live. No database setup, just push and deploy.

#### Step 1: Prepare Code for Render
```bash
cd /Users/peggu/Portfolio/Portfolio

# Make sure everything is committed
git add .
git commit -m "Ready for free deployment"

# If you haven't pushed to GitHub yet:
# Create a repo on GitHub
# Then:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Portfolio.git
git push -u origin main
```

#### Step 2: Create Render Account (1 minute)
1. Go to https://render.com
2. Click **"Get Started"** (top right)
3. Sign up with GitHub (1-click)
4. Authorize Render to access your repos

#### Step 3: Deploy (2 minutes)
1. Click the **Render logo** (top-left)
2. Click **"New +"** → **"Web Service"**
3. Select your Portfolio repo (if it doesn't appear, click "Connect account" again)
4. Fill in:
   - **Name:** `portfolio` (or any name)
   - **Environment:** `Docker`
   - **Region:** Choose closest to you (e.g., Ohio, Singapore)
   - **Plan:** `Free` (toggle to free tier)
5. Click **"Create Web Service"**
6. Wait 3-5 minutes while it builds

#### Step 4: Add Environment Variables (30 seconds)
1. Your service is now visible in dashboard
2. Click on it
3. Go to **"Environment"** (left sidebar)
4. Click **"Add Environment Variable"**
5. Add these two:
   ```
   ENVIRONMENT = production
   LLM_PROVIDER = echo
   ```
6. Click **"Save"**
7. It auto-redeploys (watch the build log)

#### Step 5: Your Live URL ✅
1. Go back to the service
2. Look at top-right — you'll see: `https://portfolio-xyz.onrender.com`
3. Click it and test
4. **Share this URL with anyone!**

**Total time: ~5 minutes. Zero cost forever.**

---

### ALTERNATIVE: Railway Free Tier (Also Good — 5 minutes)

Railway gives you $5/month in free credits, which is usually enough for one app.

#### Steps (Same as Render)
1. Go to https://railway.app
2. Sign up with GitHub
3. Click **"New Project"** → **"Deploy from GitHub"**
4. Select your repo
5. Set environment variables (same two as above)
6. Get your URL

**Key difference:** Railway can spin down after 15 minutes of inactivity. For a portfolio, this is sometimes annoying (first visitor waits for cold start). Render doesn't do this.

---

### STOP: Before You Deploy

Make sure this file exists and is up to date:

```bash
# Check that the knowledge base exists
ls -la data/knowledge_base.md

# It should be ~270 lines
wc -l data/knowledge_base.md
```

If it's empty or missing, add some content:

```bash
cat > data/knowledge_base.md << 'EOF'
# Pragna's Knowledge Base

## CHUNK: Who I am
I'm a software engineer building AI products.

---

## CHUNK: My experience
I have 4.5 years of production engineering experience.

---
EOF
```

---

### DEPLOY NOW (Copy-Paste These Commands)

```bash
# 1. Make sure you're in the right directory
cd /Users/peggu/Portfolio/Portfolio

# 2. Commit everything
git add .
git commit -m "Deploy: free tier setup"

# 3. Push to GitHub (if not already there)
git push origin main

# 4. Then go to https://render.com and follow the 4 steps above
```

**After you hit deploy, check your email** — you'll get a notification when it's done.

---

### Test Your Live App

Once Render says "Live":

1. Open your URL in a browser
2. Try the chat widget: *"Tell me about yourself"*
3. You should see the echo response (it retrieves context from knowledge_base.md)
4. If you see errors, check the **"Logs"** tab in Render dashboard

---

### Upgrade to Claude (Optional — Makes it Better)

Your deployed app currently uses `LLM_PROVIDER=echo` (just echoes context, no real LLM).

To get real answers from Claude:

1. Go to https://console.anthropic.com/dashboard
2. Get an **API key** (click "Create key")
3. Back in Render dashboard, go to **Environment**
4. Add:
   ```
   LLM_PROVIDER = anthropic
   ANTHROPIC_API_KEY = sk-ant-YOUR_KEY_HERE
   ```
5. Redeploy (click the redeploy button)
6. Refresh your browser

**Cost:** ~$0.003 per chat message. Budget $1-5/month to test.

---

### Make It Truly Permanent (Optional — Database)

By default, chat history is stored locally and lost on redeploy.

To keep chat forever:

1. In Render dashboard, click **"New +"** → **"PostgreSQL"**
2. Create a database (free tier has 1GB, plenty for a chatbot)
3. Render auto-creates `DATABASE_URL` environment variable
4. Go back to your service, go to **Environment**
5. Verify `DATABASE_URL` is there (it should be auto-added)
6. Redeploy

Now chat history persists across restarts. ✅

---

### If Deployment Fails: Debugging

| Error | Fix |
|-------|-----|
| **Build failed: "requirements.txt not found"** | Your requirements.txt is missing. Run `pip freeze > requirements.txt` |
| **Build failed: "Dockerfile syntax error"** | Dockerfile exists and should be fine. Try `docker build .` locally to test |
| **App crashes on startup** | Check logs. Usually missing environment variable. Re-read step 4 above |
| **"ModuleNotFoundError: No module named 'app'"** | Your folder structure is wrong. It should be `Portfolio/app/main.py` |
| **Live URL gives 404** | Wrong port. Check that Dockerfile exposes port 8000 |

**To debug locally:**
```bash
docker build -t portfolio .
docker run -p 8000:8000 -e LLM_PROVIDER=echo portfolio
# Open http://localhost:8000
```

---

## Option 3: Deploy Locally with a Tunnel (Quick Test — 2 minutes)

Railway automatically detects your Docker setup and deploys.

#### Step 1: Prepare Your Code

```bash
# Make sure you're in the project directory
cd /Users/peggu/Portfolio/Portfolio

# Commit your changes to git (Railway needs a git repo)
git add .
git commit -m "Ready for deployment"

# If you haven't initialized git yet:
# git init
# git add .
# git commit -m "Initial commit"
```

#### Step 2: Create a Railway Account

1. Go to https://railway.app
2. Click **"Start Now"** (top right)
3. Sign up with GitHub (easiest option)
4. Authorize Railway to access your GitHub

#### Step 3: Deploy Your App

1. Click **"New Project"** → **"Deploy from GitHub repo"**
2. Select your Portfolio repo (you'll need to push it to GitHub first)
3. Railway auto-detects the Dockerfile
4. Click **"Deploy"**
5. Wait 2-3 minutes while it builds and deploys

#### Step 4: Set Environment Variables

In the Railway dashboard:

1. Click your deployed service
2. Go to **"Variables"** tab
3. Add these (required):
   ```
   ENVIRONMENT=production
   LLM_PROVIDER=echo
   ```

4. (Optional) Add if you want Claude:
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-YOUR_KEY
   ```

#### Step 5: Get Your Live URL

1. Go to **"Settings"** tab
2. Scroll to **"Domains"**
3. Copy the auto-generated URL: `https://your-app-xyz.railway.app`
4. Open it in your browser ✅

**That's it! Your app is live.**

**Important:** Railway's free tier gives you ~$5/month credit. To keep it running for free:
- Disable the PostgreSQL database (use SQLite instead) — edit docker-compose.yml to just the API service
- Or upgrade to a paid plan ($5+/month)

---

### Option 2: Deploy to Render (Easiest Alternative — 5 minutes)

Render is similar to Railway but sometimes has a longer free tier.

#### Step 1: Push to GitHub

```bash
# If not already done:
git init
git add .
git commit -m "Initial commit"

# Create a repo on GitHub and push
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/Portfolio.git
git push -u origin main
```

#### Step 2: Create a Render Account

1. Go to https://render.com
2. Sign up with GitHub
3. Authorize Render

#### Step 3: Deploy

1. Click **"New +"** → **"Web Service"**
2. Select your Portfolio repo
3. Fill in:
   - **Name:** portfolio
   - **Runtime:** Docker
   - **Region:** Choose closest to you
   - **Plan:** Free (or Starter if you want guaranteed uptime)

4. Click **"Create Web Service"**
5. Wait 3-5 minutes while it builds

#### Step 4: Set Environment Variables

1. Go to **"Environment"** in the left sidebar
2. Add:
   ```
   ENVIRONMENT=production
   LLM_PROVIDER=echo
   ```

3. Redeploy (click the redeploy button)

#### Step 5: Get Your Live URL

Your app is at: `https://portfolio-xyz.onrender.com` (see in the dashboard)

---

### Option 3: Deploy Locally with a Tunnel (Quick Test — 2 minutes)

**Goal:** Get a live URL without setting up a cloud account (great for testing).

#### Using ngrok (Free)

1. Install ngrok: https://ngrok.com/download
2. Sign up for a free account
3. Run your app locally:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. In another terminal:
   ```bash
   ngrok http 8000
   ```

5. Copy the URL (like `https://abc123.ngrok.io`)
6. Share it — your local app is now publicly accessible

**Downside:** Stops when you close the tunnel. Great for demos, not for permanent deployment.

---

### Option 4: Deploy to AWS (Production-Grade — 30 minutes)

For serious, scalable deployment.

#### Using AWS App Runner (Simplest AWS Option)

1. **Create an AWS account** at https://aws.amazon.com
2. **Push your code to GitHub** (same as above)
3. **Go to AWS App Runner** (search in AWS console)
4. Click **"Create service"**
5. Choose **"Source code repository"** → select your GitHub repo
6. Configure:
   - Runtime: Docker
   - Port: 8000
   - Environment variables (same as Railway)
7. Click **"Create & deploy"**
8. Wait 5 minutes
9. Get your auto-generated URL

**Cost:** ~$0.065/hour running (about $47/month if always on). Scales automatically.

---

### Option 5: Deploy to DigitalOcean (Balanced — 20 minutes)

Good middle ground between easy and powerful.

#### Using DigitalOcean App Platform

1. Create an account at https://www.digitalocean.com
2. Go to **"App Platform"**
3. Click **"Create Apps"** → **"GitHub"**
4. Select your Portfolio repo
5. Configure:
   - Component type: Web Service
   - Source: Dockerfile
   - Port: 8000
6. Set environment variables
7. Choose a plan ($5-12/month)
8. Deploy

Your app gets a `.ondigitalocean.app` domain.

---

### After Deployment: What's Next?

#### 1. Custom Domain (Optional)

Want `pragna.ai` instead of `portfolio-xyz.railway.app`?

1. Buy a domain (Namecheap, GoDaddy, etc.)
2. Go to your deployment platform's domain settings
3. Point the domain's nameservers to the platform
4. Enable SSL (automatic on all platforms)

#### 2. Enable a Real LLM

Your live app runs with `LLM_PROVIDER=echo` (no API key needed).

To upgrade to Claude (better answers):

1. Get an API key from https://console.anthropic.com
2. In your deployment platform's environment variables:
   ```
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-YOUR_KEY
   ```
3. Redeploy
4. Your chatbot now answers with Claude instead of echoing context

**Cost:** ~$0.003 per chat message with Claude. Budget $1-5/month for testing.

#### 3. Enable Persistent Storage (PostgreSQL)

By default, the app uses SQLite (stores chat history locally on each deploy, lost on restart).

To keep chat history forever:

**On Railway:** Add PostgreSQL service (free tier available)
**On Render:** Add PostgreSQL database (free tier available)

Then set in environment variables:
```
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/portfolio
```

---

### Troubleshooting Deployment

| Issue | Solution |
|-------|----------|
| **"Build fails"** | Check `docker build .` works locally first. Run in terminal. |
| **"App crashes on start"** | Check logs in platform dashboard. Usually missing env var or port issue. |
| **"Embedding model won't download"** | Set `EMBEDDING_PROVIDER=hash` (uses hash fallback, no download needed) |
| **"Can't connect to database"** | Database URL format is wrong. Copy exact format from platform docs. |
| **"Live URL is slow"** | Cold start (first load after deployment). Subsequent requests are faster. |

---

### Quick Deployment Checklist

- [ ] Code is in git repo (committed)
- [ ] `.env` file is not committed (add to `.gitignore`)
- [ ] Dockerfile exists and builds locally (`docker build .`)
- [ ] Choose platform (Railway or Render recommended)
- [ ] Connect GitHub account
- [ ] Select repo and deploy
- [ ] Set environment variables (at least `ENVIRONMENT=production`)
- [ ] Test the live URL
- [ ] (Optional) Add custom domain
- [ ] (Optional) Add Claude API key for real LLM

---

## Glossary

| Term | Simple Explanation |
|------|-------------------|
| **RAG** | Retrieve context from a knowledge base, then ask an LLM to answer based on it |
| **LLM** | A neural network trained on billions of words; good at predicting next words |
| **Embedding** | A list of numbers that represents meaning; similar texts have similar embeddings |
| **Vector DB** | A database optimized for storing and searching by similarity (not exact match) |
| **FastAPI** | A Python web framework for building APIs quickly |
| **Async** | Code that can do other work while waiting for slow operations (I/O) |
| **SSE** | Server-Sent Events; a way to stream data from server to browser |
| **Chunking** | Splitting a knowledge base into smaller, meaningful units |
| **Retrieval** | Finding the most relevant chunks for a query (using embedding similarity) |
| **Augmentation** | Adding retrieved context to the LLM prompt |
| **Generation** | Asking the LLM to produce an answer |
| **Pydantic** | A Python library for validating data types |
| **SQLAlchemy** | A Python library for interacting with databases without writing SQL |
| **Hallucination** | When an LLM invents false information instead of saying "I don't know" |
| **Fine-tuning** | Training an LLM on your specific data (expensive, slow to update) |
| **Cosine Similarity** | A way to measure how similar two vectors are (0 = different, 1 = identical) |
| **Streaming** | Sending data in chunks as it's produced (like ChatGPT) instead of waiting for the whole thing |
| **Tokenization** | Splitting text into tokens (roughly words, sometimes subwords) |
| **Sentiment Analysis** | Determining if text is positive, negative, or neutral |
| **Prompt Engineering** | Crafting the text you send to an LLM to get better answers |

---

## Next Steps

1. **Run the app locally** (you've done this)
2. **Read `app/rag.py` end-to-end** (takes ~10 minutes; it's the core)
3. **Try Experiment 1** (add a chunk, see it work)
4. **Try Experiment 5** (modify the system prompt)
5. **Explore the frontend** (open index.html in editor, see how the chat widget works)
6. **Read the API docs** (http://localhost:8000/docs)
7. **Deploy to production** (docker-compose up — but that's advanced)

---

## Questions?

- **How does embedding similarity work?** → Read the `cosine_similarity` function in `app/rag.py`
- **How do I add authentication?** → Look at Pydantic's `Field(max_length=...)` for input validation
- **How do I scale this?** → Swap SQLite for PostgreSQL + pgvector (docker-compose.yml already has this)
- **Can I use this as a starting template?** → Yes! It's MIT licensed

---

**You now have the mental model. Go build!** 🚀
