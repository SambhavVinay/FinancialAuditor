# FinancialAuditor — AI claim engine

Full-stack demo for **motor insurance claim triage**: a **FastAPI** backend runs a **CrewAI** multi-agent workflow (vision, compliance, payout, customer messaging), with deterministic Python guards on top of the LLM outputs. A **Next.js** frontend collects claim details, optionally uploads a damage photo, and displays the JSON decision.

The API is titled **Innovitus AI Claim Engine** in code (`main.py`).

## Repository layout

| Path | Role |
|------|------|
| `main.py` | FastAPI app, NVIDIA multimodal vision call, CrewAI crew, payout/status/confidence enforcement |
| `pyproject.toml` / `uv.lock` | Python dependencies (managed with [uv](https://docs.astral.sh/uv/)) |
| `frontend/` | Next.js 16 + React 19 UI (Tailwind 4), posts to the backend |

## How it works (high level)

1. **Optional image**: If `image_base64` is sent, the backend calls NVIDIA **Gemma 4 31B** (multimodal) for damage level and validity. Non-vehicle images are discarded; a separate **image–description match** score can force rejection if below the configured threshold.
2. **CrewAI agents** (sequential): Forensic Vision Inspector → Policy Compliance Auditor → Payout Actuary → Customer Relations Lead (JSON synthesis). Models are configured against the NVIDIA OpenAI-compatible API in `main.py`.
3. **Python guards**: Damage bands, deductible (₹3,000), third-party / fraud rules, payout clamping, deterministic **Status** and **Confidence Score**, and mismatch handling override LLM drift where needed.

This is a **prototype / demo**; actuarial rules are simplified and not production advice.

## Prerequisites

- **Python 3.12** (see `.python-version`)
- **Node.js** 20+ (for the frontend)
- Valid **NVIDIA API keys** with access to the configured models (see configuration note below)

## Backend setup and run

From the repository root:

```bash
uv sync
uv run python main.py
```

Or, with `uvicorn` explicitly:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

The server listens on **http://localhost:8000**. CORS is open (`*`) for local development.

### API

**`POST /process-claim`**

JSON body (see `ClaimInput` in `main.py`):

- `claim_id` (string)
- `description` (string)
- `policy_type` (string, e.g. `Comprehensive`, `Third-Party`)
- `claim_amount` (integer, rupees)
- `past_claims` (integer)
- `documents` (string, comma-separated list is used for confidence scoring)
- Optional: `image_url`, `image_base64`, `image_mime_type` (defaults to `image/jpeg`)

Response keys: `Claim ID`, `Status`, `Estimated Payout`, `Confidence Score`, `Reason`, `Customer Message`.

**`POST /process-claim/stream`**

Same JSON body as above. Returns **`text/event-stream`** (SSE): one `data: {json}\n\n` frame per event. The final claim object matches `/process-claim` after all Python guards.

| Event `type` | Fields (typical) |
|--------------|------------------|
| `step_started` | `stepId`, `agentRole`, `phase` (`precall` \| `crew`) |
| `step_progress` | `stepId`, `message` |
| `step_finished` | `stepId`, `agentRole`, `phase`, `taskDescription`, `finalOutput`, `rawLog` (PowerShell-style block) |
| `result` | `payload` — final claim JSON |
| `error` | `message`, optional `detail` |

When an image is uploaded, a **`vision_precall`** step runs first (direct Gemma multimodal + image–description match). Then four **crew** steps mirror the CrewAI tasks. The UI consumes this stream with `fetch` and a readable stream (not `EventSource`, because the body must be POSTed).

### Configuration and security

API URLs and keys are currently **hardcoded** in `main.py`. Before any shared or production use, move secrets to **environment variables** (or a secrets manager), rotate any keys that were committed, and restrict CORS to known origins.

## Frontend setup and run

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. The UI calls **`POST /process-claim/stream`** on the API base URL (default **`http://localhost:8000`**). Override with **`NEXT_PUBLIC_API_BASE`** (e.g. in `frontend/.env.local`). Keep the backend running. Claim history is stored in the browser (`localStorage`, last 50 entries). While a claim runs, the **Agent pipeline** shows a vertical n8n-style graph (zigzag edges); click **Show more** on a step to open its execution details in a side popover, and use **Previous / Next / Follow live** to navigate.

For production builds: `npm run build` then `npm run start`.

## Tech stack

- **Python**: FastAPI, Pydantic, CrewAI, Uvicorn; direct `requests` to NVIDIA chat completions for vision and match scoring
- **Frontend**: Next.js 16, React 19, TypeScript, Tailwind CSS 4, @xyflow/react (n8n-style pipeline canvas)

## License

Not specified in-repo; add a `LICENSE` file if you intend to distribute this project.
