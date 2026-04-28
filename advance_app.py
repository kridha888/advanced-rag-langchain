import streamlit as st
import requests
import json

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Advanced RAG",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

/* Header */
.rag-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.rag-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(90deg, #f5f5f5 40%, #a0e4b0 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.rag-header p {
    color: #666;
    font-size: 0.95rem;
    font-family: 'JetBrains Mono', monospace;
}

/* Status badge */
.badge-ready   { background:#1a3a25; color:#6fdc8c; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-family:monospace; border:1px solid #2d6a45; }
.badge-offline { background:#3a1a1a; color:#dc6f6f; padding:4px 12px; border-radius:20px; font-size:0.78rem; font-family:monospace; border:1px solid #6a2d2d; }

/* Cards */
.card {
    background: #161616;
    border: 1px solid #262626;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.5rem;
}

/* Answer box */
.answer-box {
    background: #111;
    border-left: 3px solid #6fdc8c;
    border-radius: 0 10px 10px 0;
    padding: 1.2rem 1.4rem;
    font-size: 1rem;
    line-height: 1.75;
    color: #ddd;
    white-space: pre-wrap;
}

/* Technique log */
.log-line {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #888;
    line-height: 1.8;
}
.log-line.highlight { color: #6fdc8c; }

/* Source chip */
.source-chip {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #303030;
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #aaa;
    margin-right: 6px;
}

/* Score bar */
.score-bar-bg { background:#222; border-radius:4px; height:6px; width:100%; margin-top:4px; }
.score-bar-fill { background:linear-gradient(90deg,#6fdc8c,#3ab060); border-radius:4px; height:6px; }

/* Metric pill */
.metric-pill {
    background:#1a1a1a;
    border:1px solid #2a2a2a;
    border-radius:8px;
    padding:0.7rem 1rem;
    text-align:center;
}
.metric-pill .val { font-size:1.4rem; font-weight:700; color:#6fdc8c; }
.metric-pill .lbl { font-size:0.7rem; color:#555; font-family:monospace; text-transform:uppercase; letter-spacing:1px; }

/* Streamlit overrides */
.stButton > button {
    background: #6fdc8c !important;
    color: #0d0d0d !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    padding: 0.55rem 1.4rem !important;
    width: 100%;
}
.stButton > button:hover { background:#5cc97a !important; }
.stTextArea textarea, .stTextInput input {
    background: #161616 !important;
    border: 1px solid #2a2a2a !important;
    color: #e8e8e8 !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
}
div[data-testid="stFileUploader"] {
    background: #161616;
    border: 1px dashed #333;
    border-radius: 10px;
    padding: 0.5rem;
}
.stSpinner > div { border-top-color: #6fdc8c !important; }
.stAlert { border-radius: 10px !important; }
hr { border-color: #222 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_status():
    try:
        r = requests.get(f"{API_BASE}/status", timeout=3)
        return r.json() if r.ok else None
    except Exception:
        return None

def ingest_file(file_bytes, filename):
    try:
        r = requests.post(
            f"{API_BASE}/ingest",
            files={"file": (filename, file_bytes)},
            timeout=180,
        )
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

def ask_question(question):
    try:
        r = requests.post(
            f"{API_BASE}/ask",
            json={"question": question},
            timeout=120,
        )
        return r.json() if r.ok else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}

def get_history():
    try:
        r = requests.get(f"{API_BASE}/history", timeout=5)
        return r.json().get("history", []) if r.ok else []
    except Exception:
        return []


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <h1>Advanced RAG</h1>
    <p>Query Rewriting · HyDE · Hybrid Search · Re-Ranking · Parent-Child · Self-RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Status bar ─────────────────────────────────────────────────────────────────
status = get_status()
if status:
    col_s1, col_s2, col_s3, col_s4 = st.columns([2, 2, 2, 2])
    with col_s1:
        badge = "badge-ready" if status["status"] == "ready" else "badge-offline"
        label = "● INDEX READY" if status["status"] == "ready" else "○ NO INDEX"
        st.markdown(f'<div class="metric-pill"><div class="val"><span class="{badge}">{label}</span></div><div class="lbl">Pipeline</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="metric-pill"><div class="val" style="font-size:0.95rem">{status.get("llm","—")}</div><div class="lbl">LLM</div></div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(f'<div class="metric-pill"><div class="val" style="font-size:0.95rem">{status.get("vectorstore","—")}</div><div class="lbl">Vector Store</div></div>', unsafe_allow_html=True)
    with col_s4:
        st.markdown(f'<div class="metric-pill"><div class="val">{status.get("queries_served", 0)}</div><div class="lbl">Queries Served</div></div>', unsafe_allow_html=True)
else:
    st.error("⚠️ Cannot reach the API at `http://localhost:8000`. Make sure the server is running (`python advanced_server.py`).")

st.markdown("<hr>", unsafe_allow_html=True)

# ── Layout: Left sidebar + Main ───────────────────────────────────────────────
left, right = st.columns([1, 2], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────────
with left:
    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader("PDF or TXT file", type=["pdf", "txt"], label_visibility="collapsed")

    if uploaded:
        if st.button("Ingest Document"):
            with st.spinner("Indexing … this may take a minute"):
                result = ingest_file(uploaded.read(), uploaded.name)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(f"✓ Indexed **{uploaded.name}**")
                st.markdown(f"""
                <div class="card">
                    <div class="card-label">Ingestion Stats</div>
                    <div>📦 Parent chunks: <b>{result.get('parent_chunks', '—')}</b></div>
                    <div>🔹 Child chunks: <b>{result.get('child_chunks', '—')}</b></div>
                    <div>⏱ Time: <b>{result.get('elapsed_ms', '—')} ms</b></div>
                </div>
                """, unsafe_allow_html=True)
                st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Techniques panel
    st.markdown("### ⚙️ Active Techniques")
    techniques = [
        ("T1", "Query Rewriting"),
        ("T2", "HyDE"),
        ("T3", "Hybrid Search"),
        ("T4", "Cross-Encoder Re-Rank"),
        ("T5", "Parent-Child Chunks"),
        ("T6", "Self-RAG Guard"),
    ]
    for tag, name in techniques:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
            <span class="source-chip">{tag}</span>
            <span style="font-size:0.88rem;color:#bbb">{name}</span>
            <span style="margin-left:auto;color:#6fdc8c;font-size:0.8rem">✓</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # History panel
    st.markdown("### 🕒 Recent Queries")
    history = get_history()
    if history:
        for h in history[:5]:
            ok = h.get("hallucination_ok", True)
            icon = "✓" if ok else "⚠"
            color = "#6fdc8c" if ok else "#dc8c6f"
            # FIX: use .get() for h['question'], h['timestamp'], h['latency_ms']
            h_question = h.get('question', '')
            h_timestamp = h.get('timestamp', '')
            h_latency = h.get('latency_ms', '—')
            st.markdown(f"""
            <div class="card" style="padding:0.8rem 1rem">
                <div style="font-size:0.83rem;color:#ccc;margin-bottom:4px">{h_question[:60]}{'…' if len(h_question)>60 else ''}</div>
                <div style="font-family:monospace;font-size:0.7rem;color:#555">
                    {h_timestamp[:16].replace('T',' ')} &nbsp;·&nbsp; {h_latency} ms &nbsp;·&nbsp;
                    <span style="color:{color}">{icon} {'Grounded' if ok else 'Flagged'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#444;font-size:0.85rem">No queries yet.</div>', unsafe_allow_html=True)


# ── RIGHT: Ask ─────────────────────────────────────────────────────────────────
with right:
    st.markdown("### 💬 Ask a Question")

    question = st.text_area(
        "Question",
        placeholder="e.g. What are the main conclusions of the report?",
        height=100,
        label_visibility="collapsed",
    )

    ask_btn = st.button("Ask →", disabled=(not status or status["status"] != "ready"))

    if ask_btn and question.strip():
        with st.spinner("Thinking through 6 RAG techniques …"):
            data = ask_question(question.strip())

        if "error" in data:
            st.error(f"Error: {data['error']}")
        else:
            # ── Answer ──────────────────────────────────
            hall_ok = data.get("hallucination_ok", True)
            hall_color = "#6fdc8c" if hall_ok else "#e07a5f"
            hall_label = "✓ Grounded" if hall_ok else "⚠ Strict Mode Used"

            # FIX: data['answer'] → data.get('answer', '—')
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem">
                    <div class="card-label">Answer</div>
                    <span style="font-family:monospace;font-size:0.75rem;color:{hall_color}">{hall_label}</span>
                </div>
                <div class="answer-box">{data.get('answer', '—')}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Metrics row ──────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1:
                # FIX: data["latency_ms"] → data.get("latency_ms", "—")
                st.markdown(f'<div class="metric-pill"><div class="val">{data.get("latency_ms", "—")} ms</div><div class="lbl">Latency</div></div>', unsafe_allow_html=True)
            with m2:
                # FIX: data["sources"] → data.get("sources", [])
                st.markdown(f'<div class="metric-pill"><div class="val">{len(data.get("sources", []))}</div><div class="lbl">Sources Used</div></div>', unsafe_allow_html=True)
            with m3:
                # FIX: data["scores"] → data.get("scores", [])
                top_score = round(data.get("scores", [])[0], 3) if data.get("scores") else "—"
                st.markdown(f'<div class="metric-pill"><div class="val">{top_score}</div><div class="lbl">Top Rerank Score</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Query transformations ─────────────────────
            with st.expander("🔄 Query Transformations"):
                # FIX: data['question'], data['rewritten_query'], data['hyde_doc'] → .get()
                st.markdown(f"""
                <div class="card-label">Original</div>
                <div style="color:#ccc;margin-bottom:1rem">{data.get('question', '—')}</div>
                <div class="card-label">Rewritten (T1)</div>
                <div style="color:#ccc;margin-bottom:1rem">{data.get('rewritten_query', '—')}</div>
                <div class="card-label">HyDE Document (T2)</div>
                <div style="color:#aaa;font-size:0.88rem;font-style:italic">{data.get('hyde_doc', '—')}</div>
                """, unsafe_allow_html=True)

            # ── Sources ───────────────────────────────────
            # FIX: data['sources'] → data.get('sources', []) in both expander title and loop
            with st.expander(f"📚 Sources ({len(data.get('sources', []))})"):
                for src in data.get("sources", []):
                    # FIX: src['score'] → src.get('score', 0)
                    score_pct = max(0, min(100, int((src.get('score', 0) + 10) / 20 * 100)))
                    st.markdown(f"""
                    <div class="card" style="margin-bottom:0.8rem">
                        <div style="display:flex;justify-content:space-between;align-items:center">
                            <div>
                                <!-- FIX: src['rank'], src['source'], src['page'] → .get() -->
                                <span class="source-chip">#{src.get('rank', '?')}</span>
                                <span class="source-chip">{src.get('source', '—')}</span>
                                <span class="source-chip">p.{src.get('page', '?')}</span>
                            </div>
                            <!-- FIX: src['score'] → src.get('score', '—') -->
                            <span style="font-family:monospace;font-size:0.78rem;color:#6fdc8c">{src.get('score', '—')}</span>
                        </div>
                        <div class="score-bar-bg" style="margin:8px 0 10px">
                            <div class="score-bar-fill" style="width:{score_pct}%"></div>
                        </div>
                        <!-- FIX: src['snippet'] → src.get('snippet', '') -->
                        <div style="font-size:0.83rem;color:#888;font-family:'JetBrains Mono',monospace;line-height:1.6">{src.get('snippet', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Technique log ─────────────────────────────
            with st.expander("🔬 Pipeline Log"):
                log_html = ""
                # FIX: data["technique_log"] → data.get("technique_log", [])
                for line in data.get("technique_log", []):
                    is_hi = any(x in line for x in ["T1:", "T2:", "T3:", "T4:", "T5:", "T6:", "GEN:"])
                    cls = "log-line highlight" if is_hi else "log-line"
                    log_html += f'<div class="{cls}">{line}</div>'
                st.markdown(f'<div class="card">{log_html}</div>', unsafe_allow_html=True)

    elif ask_btn:
        st.warning("Please enter a question.")

    if status and status["status"] != "ready":
        st.info("⬅️ Upload and ingest a document first to enable asking questions.")