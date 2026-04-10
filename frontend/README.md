# FinancialAuditor — frontend

Next.js **claim submission UI** for the FinancialAuditor stack. It streams claim processing from **`POST /process-claim/stream`** on the configured API base (see below).

## Features

- Form: claim ID, policy type, description, amount, past claims, documents list
- Optional **damage photo** (drag-and-drop or file picker): resized client-side to JPEG and sent as base64 with `image_mime_type`
- Optional **image URL** field when no file is attached (passed through to the backend)
- **Claim history** sidebar: persisted in `localStorage` (`innovitus_claim_history`), last 50 submissions
- **Agent pipeline** (n8n-style): vertical zigzag graph with @xyflow/react; **Previous / Next / Follow live** navigation; always-visible **activity** panel for the focused step’s log; **Show more** on each node

## Run locally

1. Start the backend from the repo root (see root [README.md](../README.md)).
2. In this directory:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Scripts

| Command | Purpose |
|---------|---------|
| `npm run dev` | Development server |
| `npm run build` | Production build |
| `npm run start` | Serve production build |
| `npm run lint` | ESLint |

## API base URL

Set **`NEXT_PUBLIC_API_BASE`** (e.g. in `.env.local`) to your backend origin, without a trailing slash. Default is `http://localhost:8000`. The app calls `${NEXT_PUBLIC_API_BASE}/process-claim/stream`.

## Stack

Next.js 16, React 19, TypeScript, Tailwind CSS 4, [@xyflow/react](https://reactflow.dev/) for the n8n-style agent pipeline canvas (see `package.json`).
