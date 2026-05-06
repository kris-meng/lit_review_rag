import { useState, useRef, useEffect, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

const API = "http://localhost:8000";
const genSessionId = () => Math.random().toString(36).slice(2);
pdfjs.GlobalWorkerOptions.workerSrc = `https://unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
// ─── helpers ─────────────────────────────────────────────────────────────────

function fmt(ts) {
  return new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

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

// ─── styles ───────────────────────────────────────────────────────────────────

const colors = {
  bg: "#2d2535",
  panel: "#3a2f45",
  panelLight: "#4a3d58",
  chat: "#5a4f68",
  inputBg: "#ede8f5",
  accent: "#c084fc",
  accentDim: "rgba(192,132,252,0.15)",
  accentBorder: "rgba(192,132,252,0.4)",
  text: "#f0eaf8",
  textMuted: "rgba(240,234,248,0.6)",
  textDim: "rgba(240,234,248,0.35)",
  userBubble: "#6b5f7a",
  border: "rgba(255,255,255,0.08)",
  highlight: "rgba(192,132,252,0.25)",
};

const radius = { sm: 8, md: 12, lg: 20, xl: 24 };

// ─── PDF Thumbnail ────────────────────────────────────────────────────────────

function PdfThumb({ pdf, focused, onToggle, onDelete }) {
  const [hover, setHover] = useState(false);
  const [confirmDel, setConfirmDel] = useState(false);
  console.log("pdf object:", pdf);
  const pdfUrl = `${API}/pdfs/view/${pdf.filename.split("/").map(encodeURIComponent).join("/")}`;
  console.log("pdfUrl:", pdfUrl);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => { setHover(false); setConfirmDel(false); }}
      onClick={() => !confirmDel && onToggle()}
      style={{
        display: "flex", flexDirection: "column", alignItems: "center",
        gap: 8, cursor: "pointer", position: "relative", width: 100,
      }}
    >
      <div style={{
        width: 80, height: 104,
        borderRadius: radius.md, overflow: "hidden",
        border: focused ? `2px solid ${colors.accent}` : "2px solid transparent",
        position: "relative",
      }}>
        <Document
          file={pdfUrl}
          loading={<div style={{ width: 80, height: 104, background: "rgba(255,255,255,0.05)" }} />}
        >
          <Page
            pageNumber={1}
            width={80}
            height={104}
            renderTextLayer={false}
            renderAnnotationLayer={false}
          />
        </Document>

        {focused && (
          <div style={{
            position: "absolute", top: 6, right: 6,
            width: 8, height: 8, borderRadius: "50%",
            background: colors.accent,
          }} />
        )}

        {hover && !confirmDel && (
          <button
            onClick={e => { e.stopPropagation(); setConfirmDel(true); }}
            style={{
              position: "absolute", top: 4, left: 4,
              background: "rgba(0,0,0,0.6)", border: "none",
              borderRadius: 6, color: "#f87171", fontSize: 11,
              padding: "2px 5px", cursor: "pointer",
            }}
          >🗑</button>
        )}

        {confirmDel && (
          <div
            onClick={e => e.stopPropagation()}
            style={{
              position: "absolute", inset: 0, background: "rgba(0,0,0,0.8)",
              display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 6,
              borderRadius: radius.md,
            }}
          >
            <span style={{ fontSize: 11, color: "#f87171" }}>Remove?</span>
            <div style={{ display: "flex", gap: 4 }}>
              <button onClick={() => onDelete()} style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 4,
                background: "#f87171", border: "none", color: "white", cursor: "pointer",
              }}>Yes</button>
              <button onClick={() => setConfirmDel(false)} style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 4,
                background: "transparent", border: "1px solid rgba(255,255,255,0.3)",
                color: "white", cursor: "pointer",
              }}>No</button>
            </div>
          </div>
        )}
      </div>

      <p style={{
        margin: 0, fontSize: 11, color: focused ? colors.accent : colors.textMuted,
        textAlign: "center", lineHeight: 1.3,
        overflow: "hidden", display: "-webkit-box",
        WebkitLineClamp: 2, WebkitBoxOrient: "vertical",
        width: "100%",
      }}>
        {pdf.paper_title}
      </p>
    </div>
  );
}

// ─── Drop zone ────────────────────────────────────────────────────────────────

function DropZone({ onFiles, uploading }) {
  const [over, setOver] = useState(false);
  const ref = useRef();

  const handle = files => {
    const pdfs = [...files].filter(f => f.name.endsWith(".pdf"));
    if (pdfs.length) onFiles(pdfs);
  };

  return (
    <div
      onDragOver={e => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={e => { e.preventDefault(); setOver(false); handle(e.dataTransfer.files); }}
      onClick={() => !uploading && ref.current.click()}
      style={{
        border: `1.5px dashed ${over ? colors.accent : "rgba(255,255,255,0.2)"}`,
        borderRadius: radius.md, padding: "10px 8px", textAlign: "center",
        cursor: uploading ? "wait" : "pointer",
        background: over ? colors.accentDim : "transparent",
        transition: "all 0.15s",
      }}
    >
      <input ref={ref} type="file" accept=".pdf" multiple hidden onChange={e => handle(e.target.files)} />
      <p style={{ margin: 0, fontSize: 11, color: colors.textMuted }}>
        {uploading ? "Ingesting…" : "+ Drop PDFs"}
      </p>
    </div>
  );
}

// ─── Source citation chip ─────────────────────────────────────────────────────

function CiteBubble({ source, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "inline-flex", alignItems: "center", gap: 4,
        padding: "2px 8px", margin: "0 2px",
        borderRadius: 20,
        background: colors.accentDim,
        border: `1px solid ${colors.accentBorder}`,
        color: colors.accent,
        fontSize: 11, fontWeight: 500, cursor: "pointer",
        fontFamily: "inherit",
      }}
    >
      {source?.paper?.split(" ").slice(0, 2).join(" ")} · p.{source?.page}
    </button>
  );
}

// ─── Chat bubble ─────────────────────────────────────────────────────────────

function Bubble({ msg, onSourceClick }) {
  const isUser = msg.role === "user";

  if (isUser) {
    return (
      <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 16 }}>
        <div style={{
          maxWidth: "70%", padding: "12px 16px",
          background: colors.userBubble,
          borderRadius: `${radius.lg}px ${radius.lg}px 4px ${radius.lg}px`,
          fontSize: 14, lineHeight: 1.6, color: colors.text,
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
        fontSize: 14, lineHeight: 1.75, color: colors.text,
        whiteSpace: "pre-wrap",
      }}>
        {parts.map((part, i) =>
          part.type === "text" ? (
            <span key={i}>{part.value}</span>
          ) : (
            <CiteBubble key={i} source={part.source} onClick={() => onSourceClick(part.source)} />
          )
        )}
      </div>

      {/* source chips */}
      {msg.sources?.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}>
          {msg.sources.map((s, i) => (
            <button
              key={i}
              onClick={() => onSourceClick(s)}
              style={{
                display: "inline-flex", alignItems: "center", gap: 5,
                padding: "4px 10px",
                borderRadius: 20,
                border: `1px solid rgba(255,255,255,0.15)`,
                background: "rgba(255,255,255,0.06)",
                color: colors.textMuted,
                fontSize: 11, cursor: "pointer", fontFamily: "inherit",
              }}
            >
              <span style={{
                padding: "1px 6px", borderRadius: 10, fontSize: 10, fontWeight: 600,
                background: s.type === "figure" ? "rgba(96,165,250,0.2)" :
                             s.type === "table" ? "rgba(52,211,153,0.2)" :
                             s.type === "formula" ? "rgba(251,191,36,0.2)" : "rgba(255,255,255,0.1)",
                color: s.type === "figure" ? "#60a5fa" :
                       s.type === "table" ? "#34d399" :
                       s.type === "formula" ? "#fbbf24" : colors.textMuted,
              }}>
                {s.type}
              </span>
              p.{s.page}
              {s.score && s.score !== "direct" &&
                <span style={{ color: colors.textDim }}>{Math.round(s.score * 100)}%</span>
              }
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── PDF Source Panel ─────────────────────────────────────────────────────────

function SourcePanel({ source, onClose }) {
  if (!source) return null;

  const filename = source.filename;
  const pdfUrl = filename ? `${API}/pdfs/view/${encodeURIComponent(filename)}` : null;
  const pageNum = source.page ?? 1;
  const [pageWidth, setPageWidth] = useState(0);
  const [pageHeight, setPageHeight] = useState(0);
  const pageRef = useRef();

  const onPageLoad = (page) => {
    setPageWidth(page.width);
    setPageHeight(page.height);
  };

  // docling bbox: l, b, r, t in PDF units (origin bottom-left)
  // react-pdf renders with origin top-left, so we flip y
  const bbox = source.bbox;
  const hasBox = bbox && bbox.l != null && pageWidth > 0;

  const containerWidth = 380;
  const scale = containerWidth / pageWidth;

  const highlightStyle = hasBox ? {
    position: "absolute",
    left: bbox.l * scale,
    top: (pageHeight - bbox.t) * scale,
    width: (bbox.r - bbox.l) * scale,
    height: (bbox.t - bbox.b) * scale,
    background: "rgba(192,132,252,0.3)",
    border: "2px solid rgba(192,132,252,0.8)",
    borderRadius: 2,
    pointerEvents: "none",
  } : null;

  return (
    <div style={{
      width: 420, flexShrink: 0,
      background: colors.panel,
      borderLeft: `1px solid ${colors.border}`,
      display: "flex", flexDirection: "column",
      borderRadius: `0 ${radius.xl}px ${radius.xl}px 0`,
      overflow: "hidden",
    }}>
      {/* header */}
      <div style={{
        padding: "12px 16px", flexShrink: 0,
        borderBottom: `1px solid ${colors.border}`,
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <p style={{ margin: 0, fontSize: 12, fontWeight: 600, color: colors.text }}>
            {source.paper}
          </p>
          <p style={{ margin: 0, fontSize: 11, color: colors.textMuted }}>
            {source.section} · page {source.page} · {source.type}
            {source.score && source.score !== "direct" && ` · ${Math.round(source.score * 100)}% match`}
          </p>
        </div>
        <button onClick={onClose} style={{
          background: "rgba(255,255,255,0.1)", border: "none",
          borderRadius: "50%", width: 28, height: 28,
          color: colors.text, cursor: "pointer", fontSize: 14,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>✕</button>
      </div>

      {/* figure image */}
      {source.figure_b64 && (
        <div style={{ padding: "12px 16px 0", flexShrink: 0 }}>
          <img
            src={`data:image/png;base64,${source.figure_b64}`}
            alt="figure"
            style={{ width: "100%", borderRadius: 8, border: `1px solid ${colors.border}` }}
          />
        </div>
      )}

      {/* PDF page with highlight */}
      <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px" }}>
        {pdfUrl ? (
          <div style={{ position: "relative", display: "inline-block", width: "100%" }}>
            <Document file={pdfUrl}>
              <Page
                pageNumber={pageNum}
                width={containerWidth}
                onLoadSuccess={onPageLoad}
                inputRef={pageRef}
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            </Document>
            {highlightStyle && <div style={highlightStyle} />}
          </div>
        ) : (
          <div style={{ color: colors.textDim, fontSize: 12, textAlign: "center", marginTop: 40 }}>
            PDF not available
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Toast ────────────────────────────────────────────────────────────────────

function Toast({ items, onDismiss }) {
  if (!items.length) return null;
  return (
    <div style={{ position: "fixed", bottom: 24, right: 24, display: "flex", flexDirection: "column", gap: 8, zIndex: 999 }}>
      {items.map(item => (
        <div key={item.id} style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 14px", minWidth: 260,
          background: colors.panelLight,
          border: `1px solid ${colors.border}`,
          borderRadius: radius.md,
        }}>
          <span style={{ fontSize: 13, color: item.error ? "#f87171" : colors.text, flex: 1 }}>{item.message}</span>
          <button onClick={() => onDismiss(item.id)} style={{ background: "none", border: "none", color: colors.textMuted, cursor: "pointer", fontSize: 14 }}>✕</button>
        </div>
      ))}
    </div>
  );
}

// ─── Spinner ──────────────────────────────────────────────────────────────────

function Spinner() {
  return (
    <div style={{
      width: 14, height: 14,
      border: "2px solid rgba(255,255,255,0.2)",
      borderTopColor: colors.accent,
      borderRadius: "50%",
      animation: "spin 0.8s linear infinite",
    }}>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [pdfs, setPdfs] = useState([]);
  const [focused, setFocused] = useState(new Set());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  const [sessionId] = useState(genSessionId);
  const scrollRef = useRef();

  const refreshPdfs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/pdfs`);
      setPdfs(await res.json());
    } catch { pushToast("Could not reach the API server.", true); }
  }, []);

  useEffect(() => { refreshPdfs(); }, [refreshPdfs]);
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  const pushToast = (message, error = false) => {
    const id = Math.random().toString(36).slice(2);
    setToasts(t => [...t, { id, message, error }]);
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), 5000);
  };

  const handleUpload = async (files) => {
    setUploading(true);
    for (const file of files) {
      pushToast(`Ingesting "${file.name}"…`);
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await fetch(`${API}/pdfs/upload`, { method: "POST", body: fd });
        const data = await res.json();
        pushToast(data.status === "already_ingested"
          ? `"${file.name}" already in library.`
          : `Done: "${data.paper_title}" (${data.node_count} chunks)`);
      } catch (e) {
        pushToast(`Failed: ${e.message}`, true);
      }
    }
    setUploading(false);
    refreshPdfs();
  };

  const handleDelete = async (pdf) => {
    try {
      await fetch(`${API}/pdfs/${encodeURIComponent(pdf.filename)}`, { method: "DELETE" });
      setFocused(f => { const n = new Set(f); n.delete(pdf.paper_title); return n; });
      pushToast(`Removed "${pdf.paper_title}"`);
      refreshPdfs();
    } catch (e) { pushToast(`Delete failed: ${e.message}`, true); }
  };

  const toggleFocus = (title) => {
    setFocused(f => {
      const n = new Set(f);
      n.has(title) ? n.delete(title) : n.add(title);
      return n;
    });
  };

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    setActiveSource(null);
    setMessages(m => [...m, { role: "user", content: q }]);
    setLoading(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, query: q, paper_titles: [...focused] }),
      });
      const data = await res.json();
      setMessages(m => [...m, {
        role: "assistant", content: data.answer,
        sources: data.sources, scores: data.scores,
      }]);
    } catch (e) {
      setMessages(m => [...m, { role: "assistant", content: `Error: ${e.message}`, sources: [] }]);
    }
    setLoading(false);
  };

  const handleKey = e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } };

  const hints = ["Summarize this paper", "Compare Papers", "Find Research Gap", "Show me Figure 2"];

  return (
    <div style={{
      width: "100vw", height: "100vh",
      background: colors.bg,
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      overflow: "hidden",
    }}>
      {/* outer card */}
      <div style={{
        width: "calc(100vw - 48px)", height: "calc(100vh - 48px)",
        maxWidth: 1400,
        background: colors.panel,
        borderRadius: radius.xl,
        display: "flex", overflow: "hidden",
        boxShadow: "0 24px 80px rgba(0,0,0,0.5)",
        minWidth: 0,
      }}>

        {/* ── LEFT: documents panel ───────────────────────────────────── */}
        <div style={{
          width: 300, flexShrink: 0,
          display: "flex", flexDirection: "column",
          borderRight: `1px solid ${colors.border}`,
          padding: "1.25rem 1rem",
          background: colors.panelLight,
          borderRadius: `${radius.xl}px 0 0 ${radius.xl}px`,
          overflow: "hidden", 
        }}>
          {/* header */}
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
            <span style={{ fontSize: 15, fontWeight: 700, color: colors.text }}>Documents</span>
            <div style={{ display: "flex", gap: 8 }}>
              {focused.size > 0 && (
                <button
                  onClick={() => setFocused(new Set())}
                  title="Clear focus"
                  style={{ background: "none", border: "none", cursor: "pointer", color: colors.accent, fontSize: 16 }}
                >⊘</button>
              )}
            </div>
          </div>

          {/* focused indicator */}
          {focused.size > 0 && (
            <div style={{
              marginBottom: 12, padding: "5px 10px",
              background: colors.accentDim, borderRadius: radius.sm,
              fontSize: 11, color: colors.accent,
            }}>
              {focused.size} paper{focused.size > 1 ? "s" : ""} focused
            </div>
          )}

          {/* thumbnails grid */}
          <div style={{
            flex: 1, overflowY: "auto",
            display: "flex", flexWrap: "wrap",
            gap: 12, alignContent: "flex-start",
            paddingBottom: 8,
          }}>
            {pdfs.length === 0 && (
              <p style={{ fontSize: 12, color: colors.textDim, width: "100%", textAlign: "center", marginTop: 24 }}>
                No papers yet
              </p>
            )}
            {pdfs.map(pdf => (
              <PdfThumb
                key={pdf.filename}
                pdf={pdf}
                focused={focused.has(pdf.paper_title)}
                onToggle={() => toggleFocus(pdf.paper_title)}
                onDelete={() => handleDelete(pdf)}
              />
            ))}
          </div>

          {/* drop zone */}
          <div style={{ marginTop: 12 }}>
            <DropZone onFiles={handleUpload} uploading={uploading} />
          </div>
        </div>

        {/* ── MIDDLE: chat ────────────────────────────────────────────── */}
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          background: colors.chat, minWidth: 0,
          textAlign: "left", 
        }}>
{/* focused paper chips + reset */}
<div style={{
  padding: "8px 1.25rem",
  borderBottom: `1px solid ${colors.border}`,
  display: "flex", alignItems: "center", gap: 6,
  flexShrink: 0,
}}>
  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, flex: 1 }}>
    {focused.size === 0 && (
      <span style={{ fontSize: 11, color: colors.textDim }}>Searching all papers</span>
    )}
    {[...focused].map(t => (
      <div key={t} style={{
        display: "inline-flex", alignItems: "center", gap: 5,
        padding: "3px 10px", borderRadius: 20,
        background: colors.accentDim,
        border: `1px solid ${colors.accentBorder}`,
        fontSize: 11, color: colors.accent,
      }}>
        {t.length > 35 ? t.slice(0, 35) + "…" : t}
        <button onClick={() => toggleFocus(t)} style={{
          background: "none", border: "none", cursor: "pointer",
          color: colors.accent, fontSize: 12, padding: 0, lineHeight: 1,
        }}>✕</button>
      </div>
    ))}
  </div>
  {messages.length > 0 && (
    <button
      onClick={async () => {
        await fetch(`${API}/sessions/${sessionId}`, { method: "DELETE" });
        setMessages([]);
      }}
      style={{
        background: "rgba(255,255,255,0.08)", border: "none",
        borderRadius: 20, padding: "4px 12px", flexShrink: 0,
        color: colors.textMuted, fontSize: 11, cursor: "pointer",
      }}
    >Reset</button>
  )}
</div>
          {/* messages */}
<div ref={scrollRef} style={{ flex: 1, overflowY: "auto", padding: "1.5rem 1.25rem", position: "relative" }}>

  {messages.length === 0 && (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 20, paddingTop: 8 }}>
      <p style={{ fontSize: 20, fontWeight: 700, color: colors.text, margin: 0 }}>
        Ready to dive deep into research?
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "flex-start" }}>
        {hints.map(h => (
          <button key={h} onClick={() => setInput(h)} style={{
            padding: "8px 18px", borderRadius: 20, fontSize: 13,
            background: "rgba(255,255,255,0.1)",
            border: "1px solid rgba(255,255,255,0.15)",
            color: colors.text, cursor: "pointer", fontFamily: "inherit",
          }}>{h}</button>
        ))}
      </div>
    </div>
  )}

  {messages.map((msg, i) => (
    <Bubble key={i} msg={msg} onSourceClick={setActiveSource} />
  ))}

  {loading && (
    <div style={{ display: "flex", gap: 8, alignItems: "center", color: colors.textMuted, fontSize: 13, marginBottom: 16 }}>
      <Spinner /> Searching and generating…
    </div>
  )}
</div>

          {/* input */}
          <div style={{ padding: "1rem 1.25rem" }}>
            <div style={{
              display: "flex", alignItems: "center", gap: 10,
              background: colors.inputBg,
              borderRadius: 30, padding: "8px 8px 8px 18px",
            }}>
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                placeholder="Ask away!"
                rows={1}
                style={{
                  flex: 1, resize: "none", border: "none", outline: "none",
                  background: "transparent", fontSize: 14, color: "#1a1a2e",
                  fontFamily: "inherit", lineHeight: 1.5,
                }}
              />
              <button
                onClick={send}
                disabled={loading || !input.trim()}
                style={{
                  width: 36, height: 36, borderRadius: "50%", border: "none",
                  background: loading || !input.trim() ? "rgba(0,0,0,0.15)" : colors.accent,
                  color: "white", cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 16, flexShrink: 0,
                }}
              >↑</button>
            </div>
          </div>
        </div>

        

        {/* ── RIGHT: source panel ──────────────────────────────────────── */}
        {activeSource ? (
          <SourcePanel source={activeSource} onClose={() => setActiveSource(null)} />
        ) : (
          <div style={{
            width: 0, transition: "width 0.2s ease", overflow: "hidden",
          }} />
        )}

      </div>

      <Toast items={toasts} onDismiss={id => setToasts(t => t.filter(x => x.id !== id))} />
    </div>
  );
}