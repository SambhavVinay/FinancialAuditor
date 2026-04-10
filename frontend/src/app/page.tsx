"use client";

import { useState, useEffect, useCallback, useRef } from "react";

// ─── Types ────────────────────────────────────────────────────────────────────

interface ClaimInput {
  claim_id: string;
  description: string;
  policy_type: string;
  claim_amount: number;
  past_claims: number;
  documents: string;
  image_url?: string;
  image_base64?: string;
  image_mime_type?: string;
}

interface ClaimResult {
  "Claim ID": string;
  Status: string;
  "Estimated Payout": string | number;
  "Confidence Score": string | number;
  Reason: string;
  "Customer Message": string;
}

interface HistoryEntry {
  id: string; // unique key
  submittedAt: string; // ISO timestamp
  input: ClaimInput;
  result: ClaimResult | null;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const LS_KEY = "innovitus_claim_history";

function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]) {
  localStorage.setItem(LS_KEY, JSON.stringify(entries.slice(0, 50))); // keep last 50
}

const STATUS_COLORS: Record<string, { dot: string; chip: string }> = {
  Approved: {
    dot: "#4ade80",
    chip: "color:#4ade80;border-color:rgba(74,222,128,0.3);background:rgba(74,222,128,0.08)",
  },
  Rejected: {
    dot: "#f87171",
    chip: "color:#f87171;border-color:rgba(248,113,113,0.3);background:rgba(248,113,113,0.08)",
  },
  "Partially Approved": {
    dot: "#fbbf24",
    chip: "color:#fbbf24;border-color:rgba(251,191,36,0.3);background:rgba(251,191,36,0.08)",
  },
  "Under Review": {
    dot: "#38bdf8",
    chip: "color:#38bdf8;border-color:rgba(56,189,248,0.3);background:rgba(56,189,248,0.08)",
  },
};

function getChipStyle(status: string): string {
  for (const key of Object.keys(STATUS_COLORS)) {
    if (status?.toLowerCase().includes(key.toLowerCase()))
      return STATUS_COLORS[key].chip;
  }
  return "color:#a78bfa;border-color:rgba(167,139,250,0.3);background:rgba(167,139,250,0.08)";
}

function getDotColor(status: string): string {
  for (const key of Object.keys(STATUS_COLORS)) {
    if (status?.toLowerCase().includes(key.toLowerCase()))
      return STATUS_COLORS[key].dot;
  }
  return "#a78bfa";
}

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return (
    d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" }) +
    " · " +
    d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })
  );
}

const EMPTY_FORM: ClaimInput = {
  claim_id: "",
  description: "",
  policy_type: "Comprehensive",
  claim_amount: 0,
  past_claims: 0,
  documents: "",
  image_url: "",
};

// Resize image to max dimension and return base64 string
function resizeImage(
  file: File,
  maxDim = 1024,
): Promise<{ base64: string; mimeType: string }> {
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      let { width, height } = img;
      if (width > maxDim || height > maxDim) {
        if (width > height) {
          height = Math.round((height * maxDim) / width);
          width = maxDim;
        } else {
          width = Math.round((width * maxDim) / height);
          height = maxDim;
        }
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d")!.drawImage(img, 0, 0, width, height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.82);
      resolve({ base64: dataUrl.split(",")[1], mimeType: "image/jpeg" });
    };
    img.src = url;
  });
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  const [form, setForm] = useState<ClaimInput>(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClaimResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Image upload state
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imageMime, setImageMime] = useState<string>("image/jpeg");
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load history from localStorage on mount
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) return;
    const { base64, mimeType } = await resizeImage(file);
    setImageBase64(base64);
    setImageMime(mimeType);
    setImagePreview(`data:${mimeType};base64,${base64}`);
  }, []);

  const handleFileInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) await handleFileSelect(file);
    },
    [handleFileSelect],
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) await handleFileSelect(file);
    },
    [handleFileSelect],
  );

  const clearImage = () => {
    setImageBase64(null);
    setImagePreview(null);
    setImageMime("image/jpeg");
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleChange = useCallback(
    (
      e: React.ChangeEvent<
        HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement
      >,
    ) => {
      const { name, value } = e.target;
      setForm((prev) => ({
        ...prev,
        [name]:
          name === "claim_amount" || name === "past_claims"
            ? Number(value)
            : value,
      }));
    },
    [],
  );

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);
    setError(null);

    const entryId = `${Date.now()}`;
    const submittedAt = new Date().toISOString();

    try {
      const body: ClaimInput = {
        ...form,
        image_url: form.image_url || undefined,
        image_base64: imageBase64 || undefined,
        image_mime_type: imageBase64 ? imageMime : undefined,
      };

      const res = await fetch("http://localhost:8000/process-claim", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data: ClaimResult = await res.json();
      setResult(data);

      const entry: HistoryEntry = {
        id: entryId,
        submittedAt,
        input: { ...form },
        result: data,
      };
      const updated = [entry, ...history];
      setHistory(updated);
      saveHistory(updated);
      setActiveId(entryId);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Unknown error";
      setError(msg);
      const entry: HistoryEntry = {
        id: entryId,
        submittedAt,
        input: { ...form },
        result: null,
      };
      const updated = [entry, ...history];
      setHistory(updated);
      saveHistory(updated);
      setActiveId(entryId);
    } finally {
      setLoading(false);
    }
  };

  const loadFromHistory = (entry: HistoryEntry) => {
    setForm({ ...entry.input });
    setResult(entry.result);
    setError(null);
    setActiveId(entry.id);
  };

  const deleteEntry = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const updated = history.filter((h) => h.id !== id);
    setHistory(updated);
    saveHistory(updated);
    if (activeId === id) {
      setActiveId(null);
      setResult(null);
      setForm(EMPTY_FORM);
    }
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem(LS_KEY);
    setActiveId(null);
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Inter', sans-serif; background: #050508; color: #e2e8f0; min-height: 100vh; }

        .bg-grid {
          background-image:
            linear-gradient(rgba(99,102,241,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(99,102,241,0.04) 1px, transparent 1px);
          background-size: 48px 48px;
        }
        .glow-orb { position: fixed; border-radius: 50%; filter: blur(120px); pointer-events: none; z-index: 0; }

        /* ── App shell ─────────────────────────────────────────────────── */
        .app-shell {
          display: flex;
          min-height: 100vh;
          position: relative;
          z-index: 1;
        }

        /* ── Sidebar ───────────────────────────────────────────────────── */
        .sidebar {
          width: 268px;
          min-width: 268px;
          background: rgba(10, 10, 18, 0.85);
          border-right: 1px solid rgba(255,255,255,0.05);
          display: flex;
          flex-direction: column;
          height: 100vh;
          position: sticky;
          top: 0;
          overflow: hidden;
          transition: width 0.3s cubic-bezier(0.16,1,0.3,1), min-width 0.3s cubic-bezier(0.16,1,0.3,1);
          backdrop-filter: blur(24px);
          -webkit-backdrop-filter: blur(24px);
        }
        .sidebar.collapsed { width: 0; min-width: 0; border-right: none; }

        .sidebar-header {
          padding: 1.25rem 1rem 0.85rem;
          border-bottom: 1px solid rgba(255,255,255,0.05);
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-shrink: 0;
        }
        .sidebar-title {
          font-size: 0.7rem;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: rgba(148,163,184,0.5);
          white-space: nowrap;
        }

        .history-list {
          flex: 1;
          overflow-y: auto;
          padding: 0.5rem;
          scrollbar-width: thin;
          scrollbar-color: rgba(139,92,246,0.2) transparent;
        }
        .history-list::-webkit-scrollbar { width: 4px; }
        .history-list::-webkit-scrollbar-track { background: transparent; }
        .history-list::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.25); border-radius: 4px; }

        .history-item {
          border-radius: 10px;
          padding: 0.7rem 0.85rem;
          cursor: pointer;
          transition: background 0.15s, border-color 0.15s;
          border: 1px solid transparent;
          margin-bottom: 0.3rem;
          position: relative;
          white-space: nowrap;
          overflow: hidden;
        }
        .history-item:hover { background: rgba(255,255,255,0.04); border-color: rgba(255,255,255,0.06); }
        .history-item.active { background: rgba(139,92,246,0.1); border-color: rgba(139,92,246,0.25); }

        .history-item-id {
          font-size: 0.82rem;
          font-weight: 600;
          color: #e2e8f0;
          overflow: hidden;
          text-overflow: ellipsis;
          margin-bottom: 0.2rem;
        }
        .history-item-meta {
          font-size: 0.68rem;
          color: rgba(148,163,184,0.45);
          overflow: hidden;
          text-overflow: ellipsis;
        }
        .history-dot {
          display: inline-block;
          width: 6px; height: 6px;
          border-radius: 50%;
          margin-right: 5px;
          flex-shrink: 0;
          vertical-align: middle;
          position: relative;
          top: -1px;
        }
        .delete-btn {
          position: absolute;
          top: 50%;
          right: 0.6rem;
          transform: translateY(-50%);
          background: none;
          border: none;
          color: rgba(148,163,184,0.3);
          cursor: pointer;
          font-size: 0.9rem;
          padding: 0.15rem 0.3rem;
          border-radius: 4px;
          opacity: 0;
          transition: opacity 0.15s, color 0.15s, background 0.15s;
        }
        .history-item:hover .delete-btn { opacity: 1; }
        .delete-btn:hover { color: #f87171; background: rgba(248,113,113,0.1); }

        .sidebar-empty {
          padding: 2rem 1rem;
          text-align: center;
          color: rgba(148,163,184,0.3);
          font-size: 0.8rem;
          line-height: 1.6;
        }

        .sb-clear-btn {
          margin: 0.5rem;
          padding: 0.5rem;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.05);
          background: transparent;
          color: rgba(148,163,184,0.4);
          font-size: 0.7rem;
          cursor: pointer;
          text-align: center;
          transition: background 0.15s, color 0.15s;
          font-family: 'Inter', sans-serif;
        }
        .sb-clear-btn:hover { background: rgba(248,113,113,0.08); color: #f87171; border-color: rgba(248,113,113,0.2); }

        /* ── Toggle btn ────────────────────────────────────────────────── */
        .toggle-btn {
          position: fixed;
          left: 1rem;
          top: 1rem;
          z-index: 100;
          background: rgba(15,15,25,0.85);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 8px;
          width: 34px; height: 34px;
          display: flex; align-items: center; justify-content: center;
          cursor: pointer;
          color: rgba(148,163,184,0.7);
          font-size: 1rem;
          transition: background 0.15s, border-color 0.15s, color 0.15s;
          backdrop-filter: blur(12px);
        }
        .toggle-btn:hover { background: rgba(139,92,246,0.15); border-color: rgba(139,92,246,0.3); color: #a78bfa; }

        /* ── Main content area ─────────────────────────────────────────── */
        .main-content {
          flex: 1;
          min-width: 0;
          padding: 2rem 1.25rem 2rem;
          overflow-y: auto;
        }

        /* ── Existing classes (unchanged) ──────────────────────────────── */
        .glass {
          background: rgba(15, 15, 25, 0.65);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(255,255,255,0.06);
        }
        .glass-input {
          background: rgba(255,255,255,0.04);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 10px;
          color: #e2e8f0;
          padding: 0.65rem 1rem;
          font-size: 0.875rem;
          width: 100%;
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s;
          font-family: 'Inter', sans-serif;
        }
        .glass-input:focus { border-color: rgba(139,92,246,0.5); box-shadow: 0 0 0 3px rgba(139,92,246,0.1); }
        .glass-input::placeholder { color: rgba(148,163,184,0.4); }
        select.glass-input option { background: #0f0f1a; color: #e2e8f0; }
        label {
          display: block;
          font-size: 0.75rem;
          font-weight: 500;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: rgba(148,163,184,0.7);
          margin-bottom: 0.4rem;
        }
        .submit-btn {
          width: 100%;
          padding: 0.85rem;
          border-radius: 12px;
          border: none;
          background: linear-gradient(135deg, #7c3aed, #4f46e5);
          color: #fff;
          font-size: 0.95rem;
          font-weight: 600;
          cursor: pointer;
          letter-spacing: 0.02em;
          transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
          box-shadow: 0 4px 20px rgba(124,58,237,0.35);
          font-family: 'Inter', sans-serif;
        }
        .submit-btn:hover:not(:disabled) { opacity: 0.92; transform: translateY(-1px); box-shadow: 0 8px 28px rgba(124,58,237,0.45); }
        .submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-card { border-radius: 16px; padding: 1.75rem; animation: fadeSlideIn 0.5s cubic-bezier(0.16,1,0.3,1); }
        @keyframes fadeSlideIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
        .stat-chip { border-radius: 9999px; padding: 0.3rem 1rem; font-size: 0.8rem; font-weight: 600; border: 1px solid; display: inline-block; }
        .divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 1.25rem 0; }
        .spinner { display: inline-block; width: 18px; height: 18px; border: 2px solid rgba(255,255,255,0.15); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error-box { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); border-radius: 12px; padding: 1rem 1.25rem; color: #fca5a5; font-size: 0.875rem; animation: fadeSlideIn 0.3s ease; }
        .badge { display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; padding: 0.2rem 0.6rem; border-radius: 6px; border: 1px solid rgba(139,92,246,0.3); color: #a78bfa; background: rgba(139,92,246,0.08); }
        .pulse-dot { width: 7px; height: 7px; border-radius: 50%; background: #a78bfa; animation: pulseDot 1.5s ease-in-out infinite; }
        @keyframes pulseDot { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }
        .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 0.6rem 0; }
        .metric-label { color: rgba(148,163,184,0.65); font-size: 0.82rem; }
        .metric-value { font-size: 0.9rem; font-weight: 600; color: #e2e8f0; }

        /* ── New claim btn ─────────────────────────────────────────────── */
        .new-claim-btn {
          margin: 0.5rem;
          padding: 0.55rem 0.85rem;
          border-radius: 8px;
          border: 1px solid rgba(139,92,246,0.25);
          background: rgba(139,92,246,0.08);
          color: #a78bfa;
          font-size: 0.75rem;
          font-weight: 600;
          cursor: pointer;
          text-align: center;
          transition: background 0.15s, border-color 0.15s;
          font-family: 'Inter', sans-serif;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 0.4rem;
          white-space: nowrap;
        }
        .new-claim-btn:hover { background: rgba(139,92,246,0.15); border-color: rgba(139,92,246,0.4); }

        @media (max-width: 768px) {
          .sidebar { position: fixed; left: 0; top: 0; z-index: 50; height: 100vh; }
          .sidebar.collapsed { width: 0; }
        }
      `}</style>

      {/* Ambient background */}
      <div
        className="bg-grid"
        style={{ position: "fixed", inset: 0, zIndex: 0 }}
      />
      <div
        className="glow-orb"
        style={{
          width: 480,
          height: 480,
          top: -120,
          left: -80,
          background: "rgba(99,50,200,0.18)",
        }}
      />
      <div
        className="glow-orb"
        style={{
          width: 380,
          height: 380,
          bottom: -80,
          right: -60,
          background: "rgba(59,130,246,0.12)",
        }}
      />

      {/* Sidebar toggle button */}
      <button
        className="toggle-btn"
        onClick={() => setSidebarOpen((o) => !o)}
        title={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        style={{
          left: sidebarOpen ? "calc(268px + 0.75rem)" : "1rem",
          transition: "left 0.3s cubic-bezier(0.16,1,0.3,1)",
        }}
      >
        {sidebarOpen ? "◀" : "☰"}
      </button>

      <div className="app-shell">
        {/* ── LEFT SIDEBAR ──────────────────────────────────────────────── */}
        <aside className={`sidebar${sidebarOpen ? "" : " collapsed"}`}>
          <div className="sidebar-header">
            <span className="sidebar-title">Claim History</span>
            <span
              style={{
                fontSize: "0.7rem",
                color: "rgba(148,163,184,0.35)",
                whiteSpace: "nowrap",
              }}
            >
              {history.length} saved
            </span>
          </div>

          {/* New claim button */}
          <button
            className="new-claim-btn"
            onClick={() => {
              setForm(EMPTY_FORM);
              setResult(null);
              setError(null);
              setActiveId(null);
            }}
          >
            <span>＋</span> New Claim
          </button>

          {/* History list */}
          <div className="history-list">
            {history.length === 0 ? (
              <div className="sidebar-empty">
                <div style={{ fontSize: "1.5rem", marginBottom: "0.5rem" }}>
                  🗂️
                </div>
                <div>No claims yet.</div>
                <div>Submitted claims will appear here.</div>
              </div>
            ) : (
              history.map((entry) => (
                <div
                  key={entry.id}
                  className={`history-item${activeId === entry.id ? " active" : ""}`}
                  onClick={() => loadFromHistory(entry)}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.3rem",
                      marginBottom: "0.2rem",
                    }}
                  >
                    {entry.result && (
                      <span
                        className="history-dot"
                        style={{ background: getDotColor(entry.result.Status) }}
                      />
                    )}
                    {!entry.result && (
                      <span
                        className="history-dot"
                        style={{ background: "#64748b" }}
                      />
                    )}
                    <span className="history-item-id">
                      {entry.input.claim_id || "Unnamed Claim"}
                    </span>
                  </div>
                  <div className="history-item-meta">
                    {fmtDate(entry.submittedAt)} · ₹
                    {entry.input.claim_amount.toLocaleString("en-IN")}
                  </div>
                  {entry.result && (
                    <div style={{ marginTop: "0.35rem" }}>
                      <span
                        className="stat-chip"
                        style={{
                          fontSize: "0.62rem",
                          padding: "0.15rem 0.5rem",
                          ...Object.fromEntries(
                            getChipStyle(entry.result.Status)
                              .split(";")
                              .filter(Boolean)
                              .map((s) => {
                                const [k, v] = s.split(":");
                                return [
                                  k
                                    .trim()
                                    .replace(/-([a-z])/g, (_, c) =>
                                      c.toUpperCase(),
                                    ),
                                  v?.trim(),
                                ];
                              }),
                          ),
                        }}
                      >
                        {entry.result.Status}
                      </span>
                    </div>
                  )}
                  <button
                    className="delete-btn"
                    onClick={(e) => deleteEntry(entry.id, e)}
                    title="Remove"
                  >
                    ✕
                  </button>
                </div>
              ))
            )}
          </div>

          {history.length > 0 && (
            <button className="sb-clear-btn" onClick={clearHistory}>
              🗑 Clear all history
            </button>
          )}
        </aside>

        {/* ── MAIN CONTENT ──────────────────────────────────────────────── */}
        <div className="main-content">
          {/* Header */}

          {/* Two-column: form + results */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "1.25rem",
              maxWidth: 1000,
              margin: "0 auto",
              alignItems: "start",
            }}
          >
            {/* ── FORM ─────────────────────────────────────────────────── */}
            <div className="glass result-card" style={{ borderRadius: 20 }}>
              <div style={{ marginBottom: "1.25rem" }}>
                <h2
                  style={{
                    fontSize: "1.05rem",
                    fontWeight: 600,
                    color: "#e2e8f0",
                  }}
                >
                  Submit New Claim
                </h2>
                <p
                  style={{
                    fontSize: "0.78rem",
                    color: "rgba(148,163,184,0.45)",
                    marginTop: 3,
                  }}
                >
                  All fields required unless marked optional
                </p>
              </div>

              <form
                onSubmit={handleSubmit}
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "0.9rem",
                }}
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "0.75rem",
                  }}
                >
                  <div>
                    <label htmlFor="claim_id">Claim ID</label>
                    <input
                      id="claim_id"
                      name="claim_id"
                      className="glass-input"
                      placeholder="CLM-2024-001"
                      value={form.claim_id}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="policy_type">Policy Type</label>
                    <select
                      id="policy_type"
                      name="policy_type"
                      className="glass-input"
                      value={form.policy_type}
                      onChange={handleChange}
                    >
                      <option value="Comprehensive">Comprehensive</option>
                      <option value="Third-Party">Third-Party</option>
                      <option value="Fire & Theft">Fire &amp; Theft</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label htmlFor="description">Claim Description</label>
                  <textarea
                    id="description"
                    name="description"
                    className="glass-input"
                    placeholder="Describe the incident in detail..."
                    rows={3}
                    value={form.description}
                    onChange={handleChange}
                    required
                    style={{ resize: "vertical" }}
                  />
                </div>

                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr",
                    gap: "0.75rem",
                  }}
                >
                  <div>
                    <label htmlFor="claim_amount">Claim Amount (₹)</label>
                    <input
                      id="claim_amount"
                      name="claim_amount"
                      type="number"
                      className="glass-input"
                      placeholder="50000"
                      min={0}
                      value={form.claim_amount || ""}
                      onChange={handleChange}
                      required
                    />
                  </div>
                  <div>
                    <label htmlFor="past_claims">Past Claims</label>
                    <input
                      id="past_claims"
                      name="past_claims"
                      type="number"
                      className="glass-input"
                      placeholder="0"
                      min={0}
                      value={form.past_claims || ""}
                      onChange={handleChange}
                      required
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="documents">Documents Submitted</label>
                  <input
                    id="documents"
                    name="documents"
                    className="glass-input"
                    placeholder="FIR report, repair estimate, photos..."
                    value={form.documents}
                    onChange={handleChange}
                    required
                  />
                </div>

                {/* Image upload */}
                <div>
                  <label>
                    Damage Photo{" "}
                    <span
                      style={{
                        color: "rgba(148,163,184,0.3)",
                        fontSize: "0.68rem",
                        textTransform: "none",
                        letterSpacing: 0,
                      }}
                    >
                      (optional — AI will visually analyse)
                    </span>
                  </label>

                  {/* Upload zone */}
                  <div
                    onClick={() =>
                      !imagePreview && fileInputRef.current?.click()
                    }
                    onDragOver={(e) => {
                      e.preventDefault();
                      setIsDragOver(true);
                    }}
                    onDragLeave={() => setIsDragOver(false)}
                    onDrop={handleDrop}
                    style={{
                      border: `1.5px dashed ${isDragOver ? "rgba(139,92,246,0.6)" : imagePreview ? "rgba(74,222,128,0.35)" : "rgba(255,255,255,0.1)"}`,
                      borderRadius: 12,
                      background: isDragOver
                        ? "rgba(139,92,246,0.07)"
                        : imagePreview
                          ? "rgba(74,222,128,0.04)"
                          : "rgba(255,255,255,0.02)",
                      transition: "all 0.2s",
                      overflow: "hidden",
                      cursor: imagePreview ? "default" : "pointer",
                      minHeight: imagePreview ? "auto" : 90,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      position: "relative",
                    }}
                  >
                    {imagePreview ? (
                      /* Preview */
                      <div style={{ width: "100%", position: "relative" }}>
                        <img
                          src={imagePreview}
                          alt="Damage preview"
                          style={{
                            width: "100%",
                            maxHeight: 200,
                            objectFit: "cover",
                            display: "block",
                            borderRadius: 10,
                          }}
                        />
                        {/* Vision badge */}
                        <div
                          style={{
                            position: "absolute",
                            top: 8,
                            left: 8,
                            background: "rgba(10,10,18,0.85)",
                            border: "1px solid rgba(139,92,246,0.4)",
                            borderRadius: 6,
                            padding: "0.2rem 0.55rem",
                            display: "flex",
                            alignItems: "center",
                            gap: "0.3rem",
                          }}
                        >
                          <div
                            className="pulse-dot"
                            style={{ background: "#4ade80" }}
                          />
                          <span
                            style={{
                              fontSize: "0.65rem",
                              fontWeight: 600,
                              color: "#4ade80",
                              letterSpacing: "0.04em",
                            }}
                          >
                            Vision AI Ready
                          </span>
                        </div>
                        {/* Remove btn */}
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            clearImage();
                          }}
                          style={{
                            position: "absolute",
                            top: 8,
                            right: 8,
                            background: "rgba(10,10,18,0.85)",
                            border: "1px solid rgba(248,113,113,0.3)",
                            borderRadius: 6,
                            color: "#f87171",
                            fontSize: "0.75rem",
                            padding: "0.2rem 0.55rem",
                            cursor: "pointer",
                            fontFamily: "Inter, sans-serif",
                          }}
                        >
                          ✕ Remove
                        </button>
                      </div>
                    ) : (
                      /* Empty state */
                      <div
                        style={{ textAlign: "center", padding: "1.25rem 1rem" }}
                      >
                        <div
                          style={{ fontSize: "1.5rem", marginBottom: "0.4rem" }}
                        >
                          📷
                        </div>
                        <p
                          style={{
                            fontSize: "0.78rem",
                            color: "rgba(148,163,184,0.55)",
                            marginBottom: "0.2rem",
                          }}
                        >
                          Drag & drop or{" "}
                          <span
                            style={{
                              color: "#a78bfa",
                              textDecoration: "underline",
                            }}
                          >
                            browse
                          </span>
                        </p>
                        <p
                          style={{
                            fontSize: "0.65rem",
                            color: "rgba(148,163,184,0.3)",
                          }}
                        >
                          JPG, PNG, WEBP · max 10 MB · auto-resized
                        </p>
                      </div>
                    )}
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    style={{ display: "none" }}
                    onChange={handleFileInputChange}
                  />

                  {/* URL fallback */}
                  {!imagePreview && (
                    <div style={{ marginTop: "0.5rem" }}>
                      <input
                        id="image_url"
                        name="image_url"
                        className="glass-input"
                        placeholder="…or paste an image URL"
                        value={form.image_url}
                        onChange={handleChange}
                        style={{ fontSize: "0.8rem" }}
                      />
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  className="submit-btn"
                  disabled={loading}
                  style={{ marginTop: "0.35rem" }}
                >
                  {loading ? (
                    <span
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        gap: "0.6rem",
                      }}
                    >
                      <span className="spinner" /> Processing Claim…
                    </span>
                  ) : (
                    "⚡ Process Claim"
                  )}
                </button>
              </form>
            </div>

            {/* ── RIGHT PANEL ──────────────────────────────────────────── */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "1.1rem",
              }}
            >
              {error && (
                <div className="error-box">
                  <strong>⚠ Error:</strong> {error}
                </div>
              )}

              {!result && !loading && !error && (
                <div
                  className="glass result-card"
                  style={{
                    borderRadius: 20,
                    textAlign: "center",
                    padding: "3rem 2rem",
                  }}
                >
                  <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>
                    🛡️
                  </div>
                  <p
                    style={{
                      color: "rgba(148,163,184,0.45)",
                      fontSize: "0.88rem",
                    }}
                  >
                    Your claim analysis will appear here once you submit the
                    form.
                  </p>
                </div>
              )}

              {loading && (
                <div className="glass result-card" style={{ borderRadius: 20 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "0.75rem",
                      marginBottom: "1.5rem",
                    }}
                  >
                    <div className="spinner" />
                    <span
                      style={{
                        color: "rgba(148,163,184,0.65)",
                        fontSize: "0.875rem",
                      }}
                    >
                      Running AI agents — this may take 30–60 seconds…
                    </span>
                  </div>
                  {[80, 55, 65, 40, 90].map((w, i) => (
                    <div
                      key={i}
                      style={{
                        height: 11,
                        borderRadius: 6,
                        background: "rgba(255,255,255,0.05)",
                        marginBottom: 11,
                        width: `${w}%`,
                        animation: "pulseDot 1.5s ease-in-out infinite",
                        animationDelay: `${i * 0.15}s`,
                      }}
                    />
                  ))}
                </div>
              )}

              {result && (
                <div className="glass result-card" style={{ borderRadius: 20 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      gap: "1rem",
                      marginBottom: "0.25rem",
                    }}
                  >
                    <div>
                      <p
                        style={{
                          fontSize: "0.7rem",
                          color: "rgba(148,163,184,0.45)",
                          fontWeight: 500,
                          textTransform: "uppercase",
                          letterSpacing: "0.07em",
                          marginBottom: 3,
                        }}
                      >
                        Claim ID
                      </p>
                      <p
                        style={{
                          fontSize: "1rem",
                          fontWeight: 700,
                          color: "#e2e8f0",
                        }}
                      >
                        {result["Claim ID"]}
                      </p>
                    </div>
                    <span
                      className="stat-chip"
                      style={Object.fromEntries(
                        getChipStyle(result.Status)
                          .split(";")
                          .filter(Boolean)
                          .map((s) => {
                            const [k, v] = s.split(":");
                            return [
                              k
                                .trim()
                                .replace(/-([a-z])/g, (_, c) =>
                                  c.toUpperCase(),
                                ),
                              v?.trim(),
                            ];
                          }),
                      )}
                    >
                      {result.Status}
                    </span>
                  </div>

                  <hr className="divider" />

                  <div style={{ marginBottom: "0.5rem" }}>
                    <div className="metric-row">
                      <span className="metric-label">Estimated Payout</span>
                      <span
                        className="metric-value"
                        style={{ color: "#4ade80", fontSize: "1.05rem" }}
                      >
                        {typeof result["Estimated Payout"] === "number"
                          ? `₹${result["Estimated Payout"].toLocaleString("en-IN")}`
                          : result["Estimated Payout"]}
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">Confidence Score</span>
                      <span className="metric-value">
                        {result["Confidence Score"]}
                      </span>
                    </div>
                  </div>

                  <hr className="divider" />

                  <div style={{ marginBottom: "1.25rem" }}>
                    <p
                      style={{
                        fontSize: "0.7rem",
                        color: "rgba(148,163,184,0.45)",
                        fontWeight: 500,
                        textTransform: "uppercase",
                        letterSpacing: "0.07em",
                        marginBottom: "0.5rem",
                      }}
                    >
                      Assessment Reason
                    </p>
                    <p
                      style={{
                        fontSize: "0.875rem",
                        color: "rgba(226,232,240,0.8)",
                        lineHeight: 1.65,
                      }}
                    >
                      {result.Reason}
                    </p>
                  </div>

                  <div
                    style={{
                      background: "rgba(139,92,246,0.08)",
                      border: "1px solid rgba(139,92,246,0.18)",
                      borderRadius: 12,
                      padding: "1rem 1.25rem",
                    }}
                  >
                    <p
                      style={{
                        fontSize: "0.7rem",
                        color: "#a78bfa",
                        fontWeight: 500,
                        textTransform: "uppercase",
                        letterSpacing: "0.07em",
                        marginBottom: "0.5rem",
                      }}
                    >
                      💬 Customer Message
                    </p>
                    <p
                      style={{
                        fontSize: "0.875rem",
                        color: "rgba(226,232,240,0.8)",
                        lineHeight: 1.65,
                      }}
                    >
                      {result["Customer Message"]}
                    </p>
                  </div>
                </div>
              )}

              {/* AI Pipeline info */}
              <div
                className="glass"
                style={{ borderRadius: 16, padding: "1.1rem 1.4rem" }}
              >
                <p
                  style={{
                    fontSize: "0.68rem",
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.09em",
                    color: "rgba(148,163,184,0.35)",
                    marginBottom: "0.75rem",
                  }}
                >
                  Processing Pipeline
                </p>
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.55rem",
                  }}
                >
                  {[
                    {
                      icon: "🔍",
                      label: "Forensic Vision Inspector",
                      model: "Gemma 4 31B",
                    },
                    {
                      icon: "📋",
                      label: "Policy Compliance Auditor",
                      model: "GPT-OSS 120B",
                    },
                    {
                      icon: "🧮",
                      label: "Payout Actuary",
                      model: "GPT-OSS 120B",
                    },
                    {
                      icon: "💼",
                      label: "Customer Relations Lead",
                      model: "GPT-OSS 120B",
                    },
                  ].map((agent) => (
                    <div
                      key={agent.label}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.65rem",
                      }}
                    >
                      <span style={{ fontSize: "0.95rem" }}>{agent.icon}</span>
                      <div style={{ flex: 1 }}>
                        <p
                          style={{
                            fontSize: "0.78rem",
                            fontWeight: 500,
                            color: "#e2e8f0",
                          }}
                        >
                          {agent.label}
                        </p>
                        <p
                          style={{
                            fontSize: "0.67rem",
                            color: "rgba(148,163,184,0.4)",
                          }}
                        >
                          {agent.model}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
