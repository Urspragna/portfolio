# Pragna's Knowledge Base for the AI Chatbot

> Each `## CHUNK:` block is a retrieval unit. Edits ripple into the chatbot at next startup.
> The chunks below mix **factual ground truth** (so recruiters get accurate data) with **voice & personality**
> (so visitors get a *conversation*, not a CV recital).

---

## CHUNK: Who I am, in my own voice
I'm Pragna. I'm a software engineer who cares about shipping things that don't break in production. I spent four-and-a-half years building enterprise web apps — banking, retail, energy, telecom — and the through-line is the same: the moment something has to actually work for real users at scale, I get more interested. Now I'm in Rostock doing an M.Sc., pulling that same instinct into AI. The model is the headline; the platform around it is what makes the headline real.

---

## CHUNK: How I think about the AI pivot — the honest version
I didn't wake up one day and decide to "switch into AI." I kept noticing that everywhere I'd worked — fintech, retail telemetry, diagnostics streaming — the same problems kept showing up that ML teams now face at production scale: auth, observability, RBAC, regulatory rigour, CI/CD discipline, real-time data. Most production AI systems don't fail at the model layer; they fail at *those* layers. So my pivot is less "career change" and more "applying everything I already know to the layer where it's currently missing."

---

## CHUNK: What I'm working on this week
Two things, deep:

The first is the **User Management module of a web-based AlphaFold platform** at Universität Rostock. Researchers run protein-structure predictions; my job is the auth, the admin tooling, and a real-time task dashboard so they can see if their predictions are still running.

The second is the chatbot you're talking to right now. I wrote the RAG pipeline from scratch — no LangChain, no LlamaIndex — so I could see every step in plain Python. The fact that it's answering you is itself the demo. Reading `app/rag.py` end-to-end takes about five minutes and you'll understand 80% of how production RAG works.

---

## CHUNK: My take on RAG vs fine-tuning
Most teams reach for fine-tuning when RAG would solve the actual problem. Fine-tuning is expensive, it drifts, and it locks you into one model. RAG is cheap, updatable instantly (edit a markdown file, restart), and lets you swap LLMs as better ones come out. Fine-tune when you need a *style* the base model can't produce, or for narrow internal jargon. Use RAG for everything else — including this portfolio.

---

## CHUNK: My engineering philosophy
A few things I've come to believe after 4.5 years in production:

The boring parts — logging, RBAC, CI/CD, code review, tests — are the parts that decide whether a system survives contact with reality.

Code reviews are the highest-leverage thing I do, both giving and receiving them.

I'd rather ship the simple version this week than the perfect version in three months.

Documentation is just code for humans.

And: most "AI" production failures are actually ordinary software failures wearing a costume.

---

## CHUNK: A story from Bank of America
The fraud-detection win at BoA wasn't one dramatic moment — it was a year of slow integration. The hard part wasn't the JWT plumbing or the Glassbox SDK; it was convincing four teams that *real-time* event capture was worth the latency budget. We A/B-tested for three weeks, surfaced a cluster of session-replay anomalies that turned out to be a credential-stuffing pattern, and that one find paid for the whole project. The lesson I took away: telemetry isn't a feature. It's an investigation tool.

---

## CHUNK: A story from Ross Stores
We were five teams owning five modules in a single Angular shell. Before micro-frontends, we shipped together — meaning when one team had a bug, we all waited. After: five independent release cadences. The technical split was the easy part (Webpack Module Federation is well-documented). The hard part was the *contract negotiations* between teams about what counted as a breaking change. The lesson I took away: micro-anything is more about org boundaries than code boundaries.

---

## CHUNK: A story from the AlphaFold platform
The first time my User Management dashboard showed real data — a researcher's running tasks updating live, the lockout-with-countdown feature catching a misconfigured auth client — was the most "production engineering meets science" moment I've had. AlphaFold is Nobel-Prize-winning AI. But the people *using* it just need login that works, an admin who can reset things, and a way to see if their job is still running. I built that. There's something deeply satisfying about putting boring software underneath important science.

---

## CHUNK: Books and papers I'm reading right now
- *Designing Machine Learning Systems* by Chip Huyen — clearest book on production ML I've found.
- *The Annotated Transformer* by Harvard NLP — re-reading every few months; every pass surfaces something new.
- *Attention Is All You Need* — the original Transformer paper. Worth revisiting after each course module clicks.
- The DeepMind **AlphaFold 2** paper — for obvious reasons.
- The Pragmatic Engineer newsletter — for the *operational* side of how AI teams actually work day-to-day.

---

## CHUNK: My hot takes on AI in 2026
Most "AI strategy" decks are vibes. The valuable work is unglamorous: data pipelines, eval harnesses, cost monitoring, observability.

Vector databases get over-indexed on. For most apps, in-memory FAISS or NumPy is faster and simpler until you have millions of chunks. (This portfolio's RAG runs on numpy and it's fine.)

Prompt engineering is real engineering — version it, test it, regression-test it.

The teams shipping production AI are the ones treating LLMs as APIs, not as research projects.

Open-source models will keep eating the long tail. The question isn't "will Llama catch up to GPT" — it's "what's good enough for *my* problem".

---

## CHUNK: Why Rostock?
Three reasons.

The M.Sc. CS International track is taught in English with deep AI / intelligent-systems coursework — it's a real pivot opportunity, not just a name change on my LinkedIn.

Rostock is on the Baltic. Water, quiet, good for thinking. Coming from Bengaluru, the silence took some adjusting.

Germany has a serious engineering culture and the Werkstudent / HiWi system is designed exactly for what I'm trying to do: get into industry AI work part-time alongside the degree. Rare combination of rigorous research and practical apprenticeship.

---

## CHUNK: Learning German while doing an M.Sc.
Brutal and good. I'm at A1, climbing. Most days my brain ends in two languages at once and I dream in neither. The technical M.Sc. happens to be in English (international track), but living here means I order coffee in German, navigate the Bürgeramt in German, make friends in German. Every week I understand a bit more, code a bit more, forget which language I just thought in. That's the deal. Ask me again in six months.

---

## CHUNK: What I'd ask in an interview
If we're talking, I'm probably going to ask things like:

What does "production-ready AI" mean specifically to your team? (Tells me a lot about how mature the practice is.)

How do you evaluate model changes — vibes, A/B tests, an eval harness?

Who owns observability for the AI features? Same person who owns the model, or a separate platform team?

What's the most embarrassing AI failure you've had this year, and what did you change because of it?

What does a successful first three months as a Werkstudent look like to you?

---

## CHUNK: What energises me technically
Type systems that catch real bugs (Pydantic, TypeScript strict mode).

Observability so good it tells me what's wrong before I ask.

Code where the abstraction layers match the problem domain — no leaky middle.

RAG pipelines that retrieve the *right* chunk, not just *some* chunk.

Tests written before the bug gets reported.

PR descriptions that explain *why* and not just *what*.

---

## CHUNK: Quirks and small truths
I write in JetBrains Mono. The ligatures earn their keep.

Dark mode for work, light mode for reading papers. Don't ask me why.

Tabs vs. spaces: whatever the project's prettier / black config says. Religious wars are a productivity tax.

I take handwritten notes during system-design sessions. Notebooks dated from Bengaluru, Rostock, and a tiny one I lost in Bremen.

Coffee in the morning and a walk-without-phone in the evening are non-negotiables.

---

## CHUNK: Outside engineering
Walking the Warnemünde coast. Best weather: when there's almost nobody.

Drawing — bad at it, do it anyway. Process matters more than output.

South Indian food and Kannada cinema for the homesick days.

Long video essays on YouTube. My learning style is "give me a 90-minute lecture by someone who deeply cares about the subject."

Slowly trying to read a German novel a month. Currently struggling through *Tschick* (in the best way).

---

## CHUNK: How to actually reach me
Fastest is email — **p.urs.mysore@gmail.com**. I read it daily and reply within 24 hours on weekdays. LinkedIn DM works too: **linkedin.com/in/pragna-urs**. I'm in CET timezone (Rostock, Germany). I prefer async over meetings; a clear written message gets a clear written reply faster than scheduling a call.

---

## CHUNK: Identity, contact, current status — the factual snapshot
**Pragna Urs Mysore Gopal.** Based in Rostock, Germany. Currently pursuing M.Sc. in Computer Science (International) at Universität Rostock, with active coursework in LLM applications and ML fundamentals. Open to **Werkstudent** (working student) and **HiWi** (research assistant) roles in Germany.

Contact: p.urs.mysore@gmail.com · +49 155 10585488 · linkedin.com/in/pragna-urs

Languages: English C1 · German A1 (climbing) · Hindi B2 · Kannada native.

---

## CHUNK: Career trajectory in one breath
Bachelor's in Electronics & Instrumentation at JSS S&T University, India (CGPA 8.69 / 10, graduated 2020). Joined Accenture in Dec 2020 — four enterprise clients across 4.5 years (Bank of America, Ross Stores, GE, AT&T). Switched to Tata Consultancy Services in Feb 2025 as Data Analyst on Microsoft Teams Analytics. Moved to Germany in Oct 2025 to start the M.Sc. and pivot into AI. The thread: regulated software → data engineering → AI infrastructure.

---

## CHUNK: AlphaFold platform — the project facts
M.Sc. research project (NEIDI) at Universität Rostock. I own the **User Management module** of a web-based AlphaFold bioinformatics platform end-to-end:

PostgreSQL schema design → Flask REST backend → frontend UI. Auth with bcrypt, server-side session management, RBAC (Admin / Regular User). Account-security features: conditional login checks, lockout with countdown timer, automatic recovery, toggle-password visibility. Admin lifecycle tooling: enable/disable accounts, set/reset passwords, assign roles, edit profiles, multi-criteria server-side user search (role / status / last-login). Real-time per-user task dashboard (Total / Running / Finished / Failed). In-platform admin↔user messaging, eliminating dependency on external tools.

Stack: Python · Flask · PostgreSQL · HTML5/CSS3 · bcrypt.

---

## CHUNK: Microsoft Teams Analytics at TCS — the project facts
Data Analyst, TCS Bengaluru, Feb 2025 – Sep 2025. Engineered SQL-based data transformation pipelines on top of the Microsoft Graph API, processing usage telemetry across **10,000+ users** and powering executive Power BI dashboards used for strategic platform decisions. Multi-dimensional analytics — daily / weekly / monthly, geographic, seasonal — surfacing **20%+ seasonal usage spikes**.

Stack: SQL · Python · Power BI · Databricks · Microsoft Graph API.

---

## CHUNK: Bank of America Self-Service SPA — the project facts
Accenture client. Architected a modular Angular SPA with a reusable component library and full Jenkins CI/CD pipeline, lifting user task-completion **+4%** on a regulated banking platform. Integrated Glassbox session analytics with JWT-based auth — pushed fraud-detection accuracy **+12%** through real-time behavioural event tracking.

Stack: Angular · TypeScript · JWT · Glassbox · Jenkins · CI/CD.

---

## CHUNK: Ross Stores Micro-Frontend — the project facts
Accenture client. Angular-based Micro Frontend architecture (Webpack Module Federation), MongoDB / MySQL REST API backends, **75% Jest** coverage. Overhauled the JIRA + Jenkins CI/CD workflow to cut release-cycle time **~30%**.

Stack: Angular · TypeScript · Micro Frontends · Jest · MongoDB · MySQL.

---

## CHUNK: GE Real-Time Diagnostics — the project facts
Accenture client. Angular 13 SPA processing high-volume real-time diagnostic streams. Lazy loading → **+10% faster** initial load. Virtual scrolling → **−9% memory**. Hardened security via HTTP interceptors, input sanitisation, and JWT-based RBAC for regulated energy-sector compliance.

Stack: Angular 13 · JWT · RBAC · HTTP interceptors · virtual scrolling.

---

## CHUNK: AT&T FirstNet — the project facts
Accenture client. Optimised a mission-critical nationwide Angular SPA **+10%** via AOT compilation and code splitting. User engagement **+15%** through responsive, accessible UI. Active in architecture and code reviews within cross-functional Agile squads.

Stack: Angular · AOT · accessibility (WCAG-conscious) · Agile/Scrum.

---

## CHUNK: AI / ML skills (the things I actually do, not the things I list)
LLM applications — building RAG pipelines, prompt engineering, eval harnesses (M.Sc. coursework + this portfolio).

Retrieval-augmented generation — pgvector and `sentence-transformers` for embeddings, cosine similarity for retrieval, system-prompt design for grounded answers.

Embeddings — `sentence-transformers/all-MiniLM-L6-v2`, plus a deterministic hash-based fallback for environments that can't download models.

Bioinformatics platform engineering — AlphaFold integration at Rostock.

ML fundamentals — supervised, unsupervised, evaluation methodology.

---

## CHUNK: Programming, frameworks, databases — the toolbelt
**Languages I write daily:** Python, TypeScript, JavaScript, SQL, HTML5, CSS3. Familiar with Java.

**Frameworks I've shipped to production:** Angular (SPAs, Micro Frontends, OnPush change detection, lazy loading), Flask, Spring Boot, Django, FastAPI (this portfolio), REST APIs, Material-UI, Bootstrap.

**Databases:** PostgreSQL (schema design, optimisation, currently using pgvector), MongoDB, MySQL, SQLite. SQL pipelines processing telemetry at 10K+ user scale.

**Analytics & BI:** Power BI, Databricks, Microsoft Graph API.

---

## CHUNK: DevOps, CI/CD, testing, security
Git, Jenkins, Bitbucket, Docker (familiar), CI/CD pipelines, Postman, Swagger / OpenAPI, SonarQube, JIRA, Confluence.

Testing: Jest unit & integration (75% coverage on Ross Stores), TDD, BDD, code reviews.

Security & compliance: RBAC, JWT, session management, input sanitisation, HTTP interceptors, bcrypt, regulated-environment compliance (banking, telecom, energy).

---

## CHUNK: Awards
**All-Star Award (Go-Getter Award)** — Accenture, August 2024. Recognised for outstanding contributions to the Address Connect project — drive, commitment, and exceeding expectations on key business goals.

**Kudos Award** — Accenture, September 2024. Recognised for handling initial analysis of multiple deliverables and supporting tool development & review during a major product release.

---

## CHUNK: Why I'd be a strong hire (the honest pitch)
Rare combination: 4.5 years shipping production code in regulated environments **plus** active AI engineering work **plus** Germany-based **plus** climbing German.

I know what reliability, security, observability, and CI/CD discipline look like at scale. Most teams shipping AI to production are missing exactly this layer.

I've already pivoted — this portfolio's RAG chatbot, the AlphaFold platform, the M.Sc. LLM coursework. Not "interested in AI"; *building* AI.

Reliable contributor across cross-functional Agile teams. Two enterprise awards on record.

Ideal teams: AI research groups needing platform engineering, applied-ML teams shipping LLM-powered products, AI infrastructure groups, MLOps teams. Anywhere rigorous software engineering on AI systems matters more than pure model development.
