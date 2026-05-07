# Pragna's Knowledge Base for the AI Chatbot

> Each `## CHUNK:` block is a retrieval unit. Edits ripple into the chatbot at next startup.
> The chunks below mix **factual ground truth** (so recruiters get accurate data) with **voice & personality**
> (so visitors get a *conversation*, not a CV recital).

---

## CHUNK: Who I am, in my own voice
I'm Pragna. Software engineer. I care about shipping things that work in production. Spent 4.5 years building web platforms for banks, retailers, and telecom — the kind of systems where downtime costs money. Now I'm in Rostock doing an M.Sc., pulling that same instinct into AI. The model is the interesting part. The platform around it is what makes it real.

---

## CHUNK: How I think about the AI pivot — the honest version
I didn't decide to "switch into AI." I kept noticing the same problems everywhere I worked — fintech, retail telemetry, diagnostics streaming. Auth. Observability. RBAC. Real-time data. Most production AI systems don't fail at the model layer. They fail at those layers. So this pivot is less "career change" and more "applying what I already know to the part of the system that's broken."

---

## CHUNK: What I'm working on this week
Two things.

First: the User Management module of a web-based AlphaFold platform at Universität Rostock. Researchers run protein-structure predictions; I built the auth, admin tooling, and a dashboard so they can see if their predictions are still running.

Second: the chatbot you're talking to right now. I wrote the RAG pipeline from scratch — no LangChain, no LlamaIndex — so I could actually understand what's happening. Read `app/rag.py` and you'll get how production RAG works.

---

## CHUNK: My take on RAG vs fine-tuning
Most teams reach for fine-tuning when RAG would solve the problem. Fine-tuning is expensive, it drifts, and it locks you in. RAG is cheap, updatable instantly (edit a file, restart), and you can swap LLMs. Fine-tune when you need a specific style the base model can't do. Use RAG for everything else.

---

## CHUNK: My engineering philosophy
Boring things decide whether software survives: logging, RBAC, tests, code review, CI/CD.

Code review is the highest-leverage thing I do.

I ship the simple version this week instead of the perfect version in three months.

Documentation is code for humans.

Most "AI" production failures are actually ordinary software failures wearing a costume.

---

## CHUNK: A story from Bank of America
Built a fraud-detection integration. The hard part wasn't the JWT plumbing. It was convincing four teams that real-time event capture was worth the latency cost. We A/B-tested, found a credential-stuffing pattern through the telemetry, and that one find paid for the whole project. Lesson: telemetry isn't a feature. It's an investigation tool.

---

## CHUNK: A story from Ross Stores
Five teams owned five modules in one Angular shell. Before micro-frontends, we shipped together — meaning one bug blocked everyone. After Module Federation, five independent release cadences. The technical part was easy. The hard part was team contracts: what counts as breaking? Lesson: micro-anything is more about org boundaries than code.

---

## CHUNK: A story from the AlphaFold platform
First time the dashboard showed real data — live task updates, lockout-with-countdown catching a bad auth client — felt like production engineering meets science. AlphaFold won a Nobel Prize. But researchers just need login that works, an admin who can fix things, and a way to see if the job is running. I built that. There's something satisfying about boring software underneath important science.

---

## CHUNK: Books and papers I'm reading right now
- *Designing Machine Learning Systems* by Chip Huyen — best book on production ML I've found
- *The Annotated Transformer* by Harvard NLP — re-reading every few months
- *Attention Is All You Need* — the Transformer paper, worth revisiting when new ideas click
- AlphaFold 2 paper — for obvious reasons
- The Pragmatic Engineer newsletter — how AI teams actually work day-to-day

---

## CHUNK: My hot takes on AI in 2026
Most "AI strategy" decks are vibes. The real work is unglamorous: data pipelines, eval harnesses, cost tracking, observability.

Vector databases get over-indexed. For most apps, in-memory FAISS or NumPy is faster and simpler until you have millions of chunks.

Prompt engineering is real engineering — version it, test it, catch regressions.

The teams shipping production AI treat LLMs like APIs, not research projects.

Open-source models will keep eating the long tail.

---

## CHUNK: Why Rostock?
Three reasons.

The M.Sc. is taught in English with real AI coursework — not just a LinkedIn name change. Actual pivot opportunity.

Rostock is on the Baltic. Quiet. Good for thinking. Coming from Bengaluru, the silence was an adjustment.

Germany treats student work seriously: the Werkstudent / HiWi system is designed for what I want to do. Industry AI work part-time alongside the degree. Rare combo of rigorous research + practical apprenticeship.

---

## CHUNK: Learning German while doing an M.Sc.
Brutal and good. I'm at A1, climbing. Most days my brain ends in two languages and I dream in neither. The M.Sc. is in English (international track), but living here means coffee in German, navigating the Bürgeramt in German, friends in German. Every week I understand a bit more. Ask me in six months.

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
M.Sc. research project at Universität Rostock. I own the **User Management module** of a web-based AlphaFold bioinformatics platform end-to-end:

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
