import { useState, useRef, useEffect, useCallback } from "react";

const API = "http://localhost:8000";

const genSessionId = () => Math.random().toString(36).slice(2);

// ─── tiny helpers ────────────────────────────────────────────────────────────

function fmt(ts) {
  return new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function initials(title) {
  return title.split(" ").slice(0, 2).map(w => w[0]).join("").toUpperCase();
}

// Maps "[src:N]" tokens in answer text to clickable chips
function parseAnswer(text, sources) {
  if (!sources?.length) return [{ type: "text", value: text }];
  const parts = [];
  const regex = /\[src:(\d+)\]/g;
  let last = 0, m;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > last) parts.push({ type: "text", value: text.slice(last, m.index) });
    const idx = parseInt(m[1], 10);
    parts.push({ type: "cite", idx, source: sources[idx] });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ type: "text", value: text.slice(last) });
  return parts;
}

// ─── Source detail modal ──────────────────────────────────────────────────────

function SourceModal({ source, onClose }) {
  if (!source) return null;
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 9999, padding: "2rem",
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderRadius: "var(--border-radius-lg)",
          width: "100%", maxWidth: 640, maxHeight: "80vh",
          display: "flex", flexDirection: "column", overflow: "hidden",
        }}
      >
        {/* header */}
        <div style={{
          padding: "1rem 1.25rem",
          borderBottom: "0.5px solid var(--color-border-tertiary)",
          display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12,
        }}>
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 4 }}>
              <TypeBadge type={source.type} />
              {source.scope === "direct" && <span style={badge("info")}>direct lookup</span>}
            </div>
            <p style={{ margin: 0, fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>
              {source.paper}
            </p>
            <p style={{ margin: "2px 0 0", fontSize: 12, color: "var(--color-text-secondary)" }}>
              {source.section} · page {source.page ?? "—"}
            </p>
          </div>
          <button onClick={onClose} style={iconBtn()}>✕</button>
        </div>

        {/* figure image */}
        {source.figure_b64 && (
          <div style={{ padding: "1rem 1.25rem 0", textAlign: "center" }}>
            <img
              src={`data:image/png;base64,${source.figure_b64}`}
              alt="figure"
              style={{ maxWidth: "100%", maxHeight: 280, objectFit: "contain", borderRadius: 6 }}
            />
          </div>
        )}

        {/* node text */}
        <div style={{ flex: 1, overflowY: "auto", padding: "1rem 1.25rem" }}>
          <p style={{
            margin: 0, fontSize: 13, lineHeight: 1.7,
            color: "var(--color-text-primary)",
            whiteSpace: "pre-wrap", fontFamily: "var(--font-mono)",
          }}>
            {source.node_text || "No text preview available."}
          </p>
        </div>

        {/* score */}
        {source.score !== undefined && source.score !== "direct" && (
          <div style={{
            padding: "0.75rem 1.25rem",
            borderTop: "0.5px solid var(--color-border-tertiary)",
            display: "flex", gap: 12,
          }}>
            <Meter label="relevance score" value={source.score} />
            {source.rerank_score !== undefined && (
              <Meter label="rerank score" value={Math.min(1, (source.rerank_score + 10) / 20)} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Meter({ label, value }) {
  const pct = Math.round(value * 100);
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: "var(--color-text-secondary)" }}>{label}</span>
        <span style={{ fontSize: 11, fontWeight: 500 }}>{pct}%</span>
      </div>
      <div style={{ height: 4, background: "var(--color-background-secondary)", borderRadius: 2 }}>
        <div style={{
          height: "100%", width: `${pct}%`, borderRadius: 2,
          background: pct > 70 ? "var(--color-text-success)" : pct > 40 ? "var(--color-text-warning)" : "var(--color-text-danger)",
          transition: "width 0.4s ease",
        }} />
      </div>
    </div>
  );
}

// ─── small style helpers ─────────────────────────────────────────────────────

function badge(variant = "gray") {
  const map = {
    info: { bg: "var(--color-background-info)", color: "var(--color-text-info)" },
    success: { bg: "var(--color-background-success)", color: "var(--color-text-success)" },
    warning: { bg: "var(--color-background-warning)", color: "var(--color-text-warning)" },
    danger: { bg: "var(--color-background-danger)", color: "var(--color-text-danger)" },
    gray: { bg: "var(--color-background-secondary)", color: "var(--color-text-secondary)" },
  };
  const { bg, color } = map[variant] || map.gray;
  return {
    display: "inline-block", fontSize: 11, padding: "2px 7px",
    borderRadius: "var(--border-radius-md)", background: bg, color,
    fontWeight: 500, whiteSpace: "nowrap",
  };
}

function TypeBadge({ type }) {
  const v = type === "figure" ? "info" : type === "table" ? "success" : type === "formula" ? "warning" : "gray";
  return <span style={badge(v)}>{type}</span>;
}

function iconBtn(extra = {}) {
  return {
    background: "none", border: "none", cursor: "pointer",
    color: "var(--color-text-secondary)", fontSize: 14, padding: "2px 6px",
    borderRadius: "var(--border-radius-md)", lineHeight: 1,
    ...extra,
  };
}

// ─── PDF pill in sidebar ──────────────────────────────────────────────────────

function PdfPill({ pdf, focused, onToggleFocus, onDelete }) {
  const [hover, setHover] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => { setHover(false); setConfirmDelete(false); }}
      style={{
        display: "flex", alignItems: "center", gap: 10, padding: "10px 12px",
        borderRadius: "var(--border-radius-md)",
        border: focused
          ? "0.5px solid var(--color-border-info)"
          : "0.5px solid var(--color-border-tertiary)",
        background: focused
          ? "var(--color-background-info)"
          : hover ? "var(--color-background-secondary)" : "var(--color-background-primary)",
        cursor: "pointer", transition: "all 0.15s ease",
        position: "relative",
      }}
      onClick={() => !confirmDelete && onToggleFocus()}
    >
      {/* initials avatar */}
      <div style={{
        width: 32, height: 32, borderRadius: 6, flexShrink: 0,
        background: focused ? "var(--color-background-primary)" : "var(--color-background-secondary)",
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 11, fontWeight: 500, color: "var(--color-text-secondary)",
        border: "0.5px solid var(--color-border-tertiary)",
      }}>
        {initials(pdf.paper_title)}
      </div>

      {/* text */}
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{
          margin: 0, fontSize: 12, fontWeight: 500,
          color: focused ? "var(--color-text-info)" : "var(--color-text-primary)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {pdf.paper_title}
        </p>
        <p style={{
          margin: 0, fontSize: 11, color: "var(--color-text-secondary)",
          overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        }}>
          {pdf.node_count} chunks · {fmt(pdf.ingested_at)}
        </p>
      </div>

      {/* delete / confirm */}
      {hover && (
        confirmDelete ? (
          <div style={{ display: "flex", gap: 4, flexShrink: 0 }} onClick={e => e.stopPropagation()}>
            <button
              onClick={() => onDelete()}
              style={{ ...iconBtn(), color: "var(--color-text-danger)", fontSize: 11, padding: "3px 8px",
                border: "0.5px solid var(--color-border-danger)", borderRadius: "var(--border-radius-md)" }}
            >
              confirm
            </button>
            <button onClick={() => setConfirmDelete(false)} style={iconBtn()}>✕</button>
          </div>
        ) : (
          <button
            onClick={e => { e.stopPropagation(); setConfirmDelete(true); }}
            style={iconBtn({ flexShrink: 0 })}
            title="Remove from library"
          >
            🗑
          </button>
        )
      )}

      {focused && (
        <div style={{
          position: "absolute", top: 6, right: 6, width: 6, height: 6,
          borderRadius: "50%", background: "var(--color-text-info)",
        }} />
      )}
    </div>
  );
}

// ─── Drop zone ────────────────────────────────────────────────────────────────

function DropZone({ onFilesDropped, uploading }) {
  const [over, setOver] = useState(false);
  const inputRef = useRef();

  const handle = files => {
    const pdfs = [...files].filter(f => f.name.endsWith(".pdf"));
    if (pdfs.length) onFilesDropped(pdfs);
  };

  return (
    <div
      onDragOver={e => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
      onClick={() => !uploading && inputRef.current.click()}
      style={{
        border: `0.5px dashed ${over ? "var(--color-border-info)" : "var(--color-border-secondary)"}`,
        borderRadius: "var(--border-radius-md)",
        padding: "16px 12px",
        textAlign: "center",
        background: over ? "var(--color-background-info)" : "transparent",
        cursor: uploading ? "wait" : "pointer",
        transition: "all 0.15s ease",
      }}
    >
      <input
        ref={inputRef} type="file" accept=".pdf" multiple hidden
        onChange={e => handle(e.target.files)}
      />
      <p style={{ margin: 0, fontSize: 12, color: "var(--color-text-secondary)" }}>
        {uploading ? "Ingesting…" : "Drop PDFs here or click to browse"}
      </p>
    </div>
  );
}

// ─── Ingestion toast ──────────────────────────────────────────────────────────

function Toast({ items, onDismiss }) {
  if (!items.length) return null;
  return (
    <div style={{
      position: "fixed", bottom: 24, right: 24, display: "flex", flexDirection: "column",
      gap: 8, zIndex: 999,
    }}>
      {items.map(item => (
        <div key={item.id} style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 14px",
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-secondary)",
          borderRadius: "var(--border-radius-md)",
          minWidth: 260,
        }}>
          <span style={{ fontSize: 13, color: item.error ? "var(--color-text-danger)" : "var(--color-text-primary)", flex: 1 }}>
            {item.message}
          </span>
          <button onClick={() => onDismiss(item.id)} style={iconBtn()}>✕</button>
        </div>
      ))}
    </div>
  );
}

// ─── Chat bubble ─────────────────────────────────────────────────────────────

function Bubble({ msg, onCiteClick }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
        <div style={{
          maxWidth: "80%", padding: "10px 14px",
          background: "var(--color-background-secondary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
          fontSize: 14, lineHeight: 1.65, color: "var(--color-text-primary)",
        }}>
          {msg.content}
        </div>
      </div>
    );
  }

  const parts = parseAnswer(msg.content, msg.sources);

  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{
        fontSize: 14, lineHeight: 1.75, color: "var(--color-text-primary)",
        whiteSpace: "pre-wrap",
      }}>
        {parts.map((part, i) =>
          part.type === "text" ? (
            <span key={i}>{part.value}</span>
          ) : (
            <button
              key={i}
              onClick={() => onCiteClick(part.source)}
              title={`${part.source?.type} · ${part.source?.paper} · p.${part.source?.page}`}
              style={{
                display: "inline-flex", alignItems: "center", gap: 4,
                padding: "1px 7px", margin: "0 2px",
                borderRadius: "var(--border-radius-md)",
                border: "0.5px solid var(--color-border-info)",
                background: "var(--color-background-info)",
                color: "var(--color-text-info)",
                fontSize: 11, fontWeight: 500, cursor: "pointer",
                lineHeight: 1.6,
              }}
            >
              ↗ {part.source?.type ?? "src"} p.{part.source?.page}
            </button>
          )
        )}
      </div>

      {/* source chips row */}
      {msg.sources?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
          {msg.sources.map((s, i) => (
            <button
              key={i}
              onClick={() => onCiteClick(s)}
              style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "3px 9px",
                borderRadius: "var(--border-radius-md)",
                border: "0.5px solid var(--color-border-tertiary)",
                background: "var(--color-background-secondary)",
                color: "var(--color-text-secondary)",
                fontSize: 11, cursor: "pointer",
              }}
            >
              <TypeBadge type={s.type} />
              <span>p.{s.page}</span>
              {s.score && s.score !== "direct" && (
                <span style={{ color: "var(--color-text-tertiary)" }}>{Math.round(s.score * 100)}%</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* quality scores */}
      {msg.scores && (
        <div style={{ display: "flex", gap: 12, marginTop: 10 }}>
          <Meter label="groundedness" value={msg.scores.groundedness} />
          <Meter label="relevance" value={msg.scores.relevance} />
        </div>
      )}
    </div>
  );
}

// ─── Main app ─────────────────────────────────────────────────────────────────

export default function App() {
  const [pdfs, setPdfs] = useState([]);
  const [focused, setFocused] = useState(new Set()); // set of paper_titles
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [modalSource, setModalSource] = useState(null);
  const [sessionId] = useState(genSessionId);
  const scrollRef = useRef();

  // ── fetch PDF list ──────────────────────────────────────────────────────────
  const refreshPdfs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/pdfs`);
      const data = await res.json();
      setPdfs(data);
    } catch (e) {
      pushToast("Could not reach the API server.", true);
    }
  }, []);

  useEffect(() => { refreshPdfs(); }, [refreshPdfs]);

  // auto-scroll
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // ── toasts ──────────────────────────────────────────────────────────────────
  const pushToast = (message, error = false) => {
    const id = Math.random().toString(36).slice(2);
    setToasts(t => [...t, { id, message, error }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 6000);
  };

  // ── upload ──────────────────────────────────────────────────────────────────
  const handleUpload = async (files) => {
    setUploading(true);
    for (const file of files) {
      pushToast(`Ingesting "${file.name}"…`);
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await fetch(`${API}/pdfs/upload`, { method: "POST", body: fd });
        const data = await res.json();
        if (data.status === "already_ingested") {
          pushToast(`"${file.name}" already in library.`);
        } else {
          pushToast(`Done: "${data.paper_title}" (${data.node_count} chunks)`);
        }
      } catch (e) {
        pushToast(`Failed to ingest "${file.name}": ${e.message}`, true);
      }
    }
    setUploading(false);
    refreshPdfs();
  };

  // ── delete ──────────────────────────────────────────────────────────────────
  const handleDelete = async (pdf) => {
    try {
      await fetch(`${API}/pdfs/${encodeURIComponent(pdf.filename)}`, { method: "DELETE" });
      setFocused(f => { const n = new Set(f); n.delete(pdf.paper_title); return n; });
      pushToast(`Removed "${pdf.paper_title}"`);
      refreshPdfs();
    } catch (e) {
      pushToast(`Delete failed: ${e.message}`, true);
    }
  };

  // ── toggle focus ────────────────────────────────────────────────────────────
  const toggleFocus = (title) => {
    setFocused(f => {
      const n = new Set(f);
      n.has(title) ? n.delete(title) : n.add(title);
      return n;
    });
  };

  // ── send chat ───────────────────────────────────────────────────────────────
  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setMessages(m => [...m, { role: "user", content: q }]);
    setLoading(true);

    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          query: q,
          paper_titles: [...focused],
        }),
      });
      const data = await res.json();
      setMessages(m => [...m, {
        role: "assistant",
        content: data.answer,
        sources: data.sources,
        scores: data.scores,
      }]);
    } catch (e) {
      setMessages(m => [...m, {
        role: "assistant",
        content: `Error: ${e.message}`,
        sources: [],
      }]);
    }
    setLoading(false);
  };

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  // ─── render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{
      display: "flex", height: "100vh", width: "100%",
      fontFamily: "var(--font-sans)",
      background: "var(--color-background-tertiary)",
      overflow: "hidden",
    }}>

      {/* ── LEFT PANEL ─────────────────────────────────────────────────────── */}
      <aside style={{
        width: 280, flexShrink: 0,
        display: "flex", flexDirection: "column",
        borderRight: "0.5px solid var(--color-border-tertiary)",
        background: "var(--color-background-primary)",
        overflow: "hidden",
      }}>
        <div style={{
          padding: "1rem 1rem 0.75rem",
          borderBottom: "0.5px solid var(--color-border-tertiary)",
        }}>
          <p style={{ margin: "0 0 2px", fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>
            Library
          </p>
          <p style={{ margin: 0, fontSize: 11, color: "var(--color-text-secondary)" }}>
            {pdfs.length} paper{pdfs.length !== 1 ? "s" : ""} ·{" "}
            {focused.size > 0 ? `${focused.size} focused` : "all in scope"}
          </p>
        </div>

        {/* drop zone */}
        <div style={{ padding: "0.75rem 1rem", borderBottom: "0.5px solid var(--color-border-tertiary)" }}>
          <DropZone onFilesDropped={handleUpload} uploading={uploading} />
        </div>

        {/* focus hint */}
        {focused.size > 0 && (
          <div style={{
            margin: "0.5rem 1rem 0",
            padding: "6px 10px",
            borderRadius: "var(--border-radius-md)",
            background: "var(--color-background-info)",
            border: "0.5px solid var(--color-border-info)",
            display: "flex", alignItems: "center", justifyContent: "space-between",
          }}>
            <span style={{ fontSize: 11, color: "var(--color-text-info)" }}>
              Querying {focused.size} paper{focused.size > 1 ? "s" : ""}
            </span>
            <button
              onClick={() => setFocused(new Set())}
              style={{ ...iconBtn(), fontSize: 11, color: "var(--color-text-info)" }}
            >
              clear
            </button>
          </div>
        )}

        {/* PDF list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "0.5rem 1rem 1rem", display: "flex", flexDirection: "column", gap: 6 }}>
          {pdfs.length === 0 && (
            <p style={{ fontSize: 12, color: "var(--color-text-tertiary)", textAlign: "center", marginTop: 24 }}>
              No papers yet. Drop a PDF above.
            </p>
          )}
          {pdfs.map(pdf => (
            <PdfPill
              key={pdf.filename}
              pdf={pdf}
              focused={focused.has(pdf.paper_title)}
              onToggleFocus={() => toggleFocus(pdf.paper_title)}
              onDelete={() => handleDelete(pdf)}
            />
          ))}
        </div>
      </aside>

      {/* ── RIGHT PANEL: CHAT ────────────────────────────────────────────────── */}
      <main style={{
        flex: 1, display: "flex", flexDirection: "column",
        minWidth: 0, background: "var(--color-background-primary)",
      }}>
        {/* header */}
        <div style={{
          padding: "0.875rem 1.5rem",
          borderBottom: "0.5px solid var(--color-border-tertiary)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          flexShrink: 0,
        }}>
          <div>
            <p style={{ margin: 0, fontSize: 13, fontWeight: 500, color: "var(--color-text-primary)" }}>
              Research assistant
            </p>
            <p style={{ margin: 0, fontSize: 11, color: "var(--color-text-secondary)" }}>
              {focused.size > 0
                ? `Focused: ${[...focused].join(", ").slice(0, 80)}${[...focused].join(", ").length > 80 ? "…" : ""}`
                : "All papers in scope — click papers on the left to narrow focus"}
            </p>
          </div>
          <button
            onClick={async () => {
              await fetch(`${API}/sessions/${sessionId}`, { method: "DELETE" });
              setMessages([]);
            }}
            style={{
              padding: "5px 12px", fontSize: 12,
              border: "0.5px solid var(--color-border-secondary)",
              borderRadius: "var(--border-radius-md)",
              background: "none", cursor: "pointer",
              color: "var(--color-text-secondary)",
            }}
          >
            Reset conversation
          </button>
        </div>

        {/* messages */}
        <div
          ref={scrollRef}
          style={{ flex: 1, overflowY: "auto", padding: "1.5rem" }}
        >
          {messages.length === 0 && (
            <div style={{ textAlign: "center", marginTop: 64 }}>
              <p style={{ fontSize: 22, fontWeight: 500, color: "var(--color-text-primary)", margin: "0 0 8px" }}>
                Ask about your papers
              </p>
              <p style={{ fontSize: 14, color: "var(--color-text-secondary)", margin: 0 }}>
                Focus specific papers on the left, or search across your whole library.
              </p>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", marginTop: 24 }}>
                {[
                  "What methods were used?",
                  "Summarise the key findings",
                  "Compare across papers",
                  "Show me Figure 2",
                ].map(hint => (
                  <button
                    key={hint}
                    onClick={() => { setInput(hint); }}
                    style={{
                      padding: "7px 14px", fontSize: 13,
                      border: "0.5px solid var(--color-border-secondary)",
                      borderRadius: "var(--border-radius-md)",
                      background: "var(--color-background-secondary)",
                      color: "var(--color-text-secondary)",
                      cursor: "pointer",
                    }}
                  >
                    {hint}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <Bubble key={i} msg={msg} onCiteClick={setModalSource} />
          ))}

          {loading && (
            <div style={{ display: "flex", gap: 5, alignItems: "center", color: "var(--color-text-tertiary)", fontSize: 13, marginBottom: 16 }}>
              <Spinner />
              Searching and generating…
            </div>
          )}
        </div>

        {/* input bar */}
        <div style={{
          padding: "1rem 1.5rem",
          borderTop: "0.5px solid var(--color-border-tertiary)",
          flexShrink: 0,
        }}>
          {/* focused paper chips */}
          {focused.size > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
              {[...focused].map(t => (
                <div key={t} style={{
                  display: "inline-flex", alignItems: "center", gap: 6,
                  padding: "3px 10px",
                  borderRadius: "var(--border-radius-md)",
                  border: "0.5px solid var(--color-border-info)",
                  background: "var(--color-background-info)",
                  fontSize: 11, color: "var(--color-text-info)",
                }}>
                  {t.length > 40 ? t.slice(0, 40) + "…" : t}
                  <button onClick={() => toggleFocus(t)} style={{ ...iconBtn(), fontSize: 10, color: "var(--color-text-info)", padding: "0 2px" }}>✕</button>
                </div>
              ))}
            </div>
          )}

          <div style={{ display: "flex", gap: 10, alignItems: "flex-end" }}>
            <textarea
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask a question about your papers… (Enter to send, Shift+Enter for new line)"
              rows={2}
              style={{
                flex: 1, resize: "none",
                padding: "10px 12px", fontSize: 14,
                border: "0.5px solid var(--color-border-secondary)",
                borderRadius: "var(--border-radius-md)",
                background: "var(--color-background-secondary)",
                color: "var(--color-text-primary)",
                fontFamily: "var(--font-sans)",
                lineHeight: 1.5,
                outline: "none",
              }}
            />
            <button
              onClick={send}
              disabled={loading || !input.trim()}
              style={{
                padding: "10px 20px", fontSize: 13, fontWeight: 500,
                borderRadius: "var(--border-radius-md)",
                border: "0.5px solid var(--color-border-secondary)",
                background: loading || !input.trim() ? "var(--color-background-secondary)" : "var(--color-background-primary)",
                color: loading || !input.trim() ? "var(--color-text-tertiary)" : "var(--color-text-primary)",
                cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                whiteSpace: "nowrap", height: 56,
              }}
            >
              Send ↵
            </button>
          </div>
        </div>
      </main>

      {/* ── modals & toasts ─────────────────────────────────────────────────── */}
      {modalSource && <SourceModal source={modalSource} onClose={() => setModalSource(null)} />}
      <Toast items={toasts} onDismiss={id => setToasts(t => t.filter(x => x.id !== id))} />
    </div>
  );
}

function Spinner() {
  return (
    <div style={{
      width: 14, height: 14, border: "2px solid var(--color-border-secondary)",
      borderTopColor: "var(--color-text-secondary)",
      borderRadius: "50%",
      animation: "spin 0.8s linear infinite",
    }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}