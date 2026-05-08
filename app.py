import contextlib
import io
import os
import sys
import time
from pathlib import Path

import streamlit as st

from reid_main import run_reid_multi


class TeeStream(io.TextIOBase):
    """Write to both a StringIO buffer (for UI log) and the real terminal simultaneously."""
    def __init__(self, buffer: io.StringIO, real_stdout):
        self._buf = buffer
        self._real = real_stdout

    def write(self, s):
        self._buf.write(s)
        self._real.write(s)
        self._real.flush()
        return len(s)

    def flush(self):
        self._buf.flush()
        self._real.flush()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
TMP_INPUT_DIR = PROJECT_ROOT / "tmp_inputs"
OUTPUT_ROOT = PROJECT_ROOT / "reid_output"

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Person ReID Studio",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar is unused — we use column layout
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def save_uploaded_file(uploaded_file, target_path: Path) -> str:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(target_path)


def read_binary_file(file_path: str) -> bytes:
    with open(file_path, "rb") as f:
        return f.read()


def validate_inputs(model_type, clip_mode, uploaded_videos, uploaded_image, text_query):
    if not uploaded_videos:
        return "Please upload at least one input video."
    # CLIP-only pipeline
    if clip_mode == "image" and uploaded_image is None:
        return "CLIP image mode requires a query image."
    if clip_mode == "text" and (not text_query or not text_query.strip()):
        return "CLIP text mode requires a text query."
    if clip_mode == "both":
        if uploaded_image is None:
            return "CLIP both mode requires a query image."
        if not text_query or not text_query.strip():
            return "CLIP both mode requires a text query."
    return None


# ── CSS injection ─────────────────────────────────────────────────────────────

def inject_css(is_dark: bool):
    if is_dark:
        # Dark theme tokens
        vars_css = """
            --bg:              #0D1117;
            --bg-secondary:    #161B22;
            --surface:         #1C2333;
            --surface-raised:  #21293B;
            --border:          rgba(255,255,255,0.08);
            --border-accent:   rgba(99,102,241,0.5);
            --text-primary:    #E6EDF3;
            --text-secondary:  #8B949E;
            --text-muted:      #484F58;
            --accent:          #6366F1;
            --accent-light:    #818CF8;
            --accent-dim:      rgba(99,102,241,0.15);
            --success:         #3FB950;
            --warning:         #D29922;
            --error:           #F85149;
            --log-bg:          #0D1117;
            --log-border:      rgba(99,102,241,0.3);
            --nav-bg:          rgba(13,17,23,0.85);
            --card-shadow:     0 4px 24px rgba(0,0,0,0.4);
            --input-bg:        rgba(22,27,34,0.9);
        """
    else:
        # Light theme tokens
        vars_css = """
            --bg:              #F6F8FA;
            --bg-secondary:    #FFFFFF;
            --surface:         #FFFFFF;
            --surface-raised:  #F0F3FF;
            --border:          rgba(0,0,0,0.08);
            --border-accent:   rgba(79,70,229,0.4);
            --text-primary:    #111827;
            --text-secondary:  #6B7280;
            --text-muted:      #9CA3AF;
            --accent:          #4F46E5;
            --accent-light:    #6366F1;
            --accent-dim:      rgba(79,70,229,0.1);
            --success:         #16A34A;
            --warning:         #D97706;
            --error:           #DC2626;
            --log-bg:          #0D1117;
            --log-border:      rgba(79,70,229,0.3);
            --nav-bg:          rgba(246,248,250,0.85);
            --card-shadow:     0 2px 12px rgba(0,0,0,0.08);
            --input-bg:        rgba(255,255,255,0.95);
        """

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{ {vars_css} }}

    /* ── Global reset ── */
    html, body, [class*="css"], .stMarkdown, p, li, span, div {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary);
    }}

    .stApp {{
        background: var(--bg) !important;
    }}

    /* ── Hide default streamlit chrome ── */
    header[data-testid="stHeader"] {{
        background: transparent !important;
        display: none;
    }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}

    /* ── Sidebar: never shown (we use column layout instead) ── */
    [data-testid="stSidebar"]          {{ display: none !important; }}
    [data-testid="collapsedControl"]  {{ display: none !important; }}
    section[data-testid="stSidebar"]  {{ display: none !important; }}

    /* ── Main content padding (custom navbar is 52px tall) ── */
    .block-container {{
        padding-top: 4.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }}

    /* ── Left control panel card ── */
    .panel-card {{
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem 1.1rem;
        box-shadow: var(--card-shadow);
        position: sticky;
        top: 4.5rem;
    }}

    /* ── Sidebar section headers ── */
    .sidebar-section-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid var(--border);
    }}

    /* ── Widget labels ── */
    [data-testid="stWidgetLabel"],
    .stRadio label,
    .stSelectbox label,
    .stSlider label,
    .stTextArea label,
    .stFileUploader label,
    .stCheckbox label {{
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em !important;
    }}

    /* ── Radio group ── */
    div[role="radiogroup"] {{
        display: flex;
        gap: 4px;
        background: var(--bg) !important;
        border: 1px solid var(--border) !important;
        padding: 4px !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
    }}
    div[role="radiogroup"] > label {{
        padding: 5px 12px !important;
        border-radius: 7px !important;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.83rem !important;
        font-weight: 500 !important;
    }}
    div[role="radiogroup"] > label:hover {{
        background: var(--accent-dim) !important;
        color: var(--accent-light) !important;
    }}

    /* ── Selectbox ── */
    div[data-baseweb="select"] > div {{
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        transition: border-color 0.2s;
    }}
    div[data-baseweb="select"] > div:hover {{
        border-color: var(--border-accent) !important;
    }}

    /* ── Text area ── */
    div[data-testid="stTextArea"] textarea {{
        background: var(--input-bg) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
        transition: border-color 0.2s;
    }}
    div[data-testid="stTextArea"] textarea:focus {{
        border-color: var(--border-accent) !important;
        box-shadow: 0 0 0 3px var(--accent-dim) !important;
    }}

    /* ── File uploader ── */
    div[data-testid="stFileUploader"] {{
        background: var(--surface) !important;
        border: 1.5px dashed var(--border-accent) !important;
        border-radius: 12px !important;
        transition: all 0.25s ease;
    }}
    div[data-testid="stFileUploader"]:hover {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 4px var(--accent-dim);
    }}
    div[data-testid="stFileUploader"] section {{
        background: transparent !important;
    }}

    /* ── Primary button ── */
    div[data-testid="stButton"] > button[kind="primary"] {{
        background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.03em !important;
        padding: 0.65rem 1.5rem !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.4) !important;
        transition: all 0.25s ease !important;
        color: #fff !important;
    }}
    div[data-testid="stButton"] > button[kind="primary"]:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.5) !important;
    }}
    div[data-testid="stButton"] > button[kind="primary"]:active {{
        transform: translateY(0px) !important;
    }}

    /* ── Secondary button ── */
    div[data-testid="stButton"] > button[kind="secondary"],
    div[data-testid="stDownloadButton"] > button {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: var(--accent-light) !important;
        transition: all 0.2s ease !important;
        padding: 0.5rem 1rem !important;
    }}
    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        background: var(--accent-dim) !important;
        border-color: var(--border-accent) !important;
        transform: translateY(-1px) !important;
    }}

    /* ── Slider ── */
    div[data-testid="stSlider"] {{
        padding: 4px 0;
    }}

    /* ── Alert / info boxes ── */
    div[data-testid="stAlert"] {{
        border-radius: 12px !important;
        border-left: 4px solid !important;
        font-size: 0.88rem;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-muted);
    }}

    /* ── Content card ── */
    .reid-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--card-shadow);
        margin-bottom: 1.25rem;
        transition: box-shadow 0.2s;
    }}
    .reid-card:hover {{
        box-shadow: 0 6px 32px rgba(0,0,0,0.15);
    }}

    /* ── Metric cards ── */
    .metrics-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}
    .metric-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        box-shadow: var(--card-shadow);
        transition: transform 0.2s, box-shadow 0.2s;
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light));
        border-radius: 14px 14px 0 0;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }}
    .metric-icon {{
        font-size: 1.4rem;
        margin-bottom: 0.4rem;
    }}
    .metric-label {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }}
    .metric-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }}
    .metric-value.accent {{
        color: var(--accent-light);
    }}

    /* ── Log console ── */
    .log-console {{
        background: #0D1117;
        border: 1px solid var(--log-border);
        border-radius: 14px;
        padding: 1rem 1.25rem;
        font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace !important;
        font-size: 0.78rem;
        line-height: 1.6;
        color: #C9D1D9;
        max-height: 360px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.4);
    }}
    .log-console .log-line {{
        display: block;
        padding: 1px 0;
    }}
    .log-console .log-line.success {{ color: #3FB950; }}
    .log-console .log-line.warning {{ color: #E3B341; }}
    .log-console .log-line.error   {{ color: #F85149; }}
    .log-console .log-line.info    {{ color: #58A6FF; }}
    .log-header {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }}
    .log-dot {{
        width: 10px; height: 10px;
        border-radius: 50%;
        display: inline-block;
    }}
    .log-dot.red   {{ background: #FF5F57; }}
    .log-dot.yellow{{ background: #FEBC2E; }}
    .log-dot.green {{ background: #28C840; }}
    .log-title {{
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-muted);
        letter-spacing: 0.05em;
    }}

    /* ── Processing loader ── */
    .loader-card {{
        background: var(--surface);
        border: 1px solid var(--border-accent);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 0 4px var(--accent-dim), var(--card-shadow);
        margin-bottom: 1.25rem;
    }}
    .loader-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }}
    .loader-subtitle {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 1.2rem;
    }}
    .progress-bar-track {{
        background: var(--bg);
        border-radius: 99px;
        height: 6px;
        overflow: hidden;
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, var(--accent), var(--accent-light), #A78BFA);
        background-size: 200% 100%;
        animation: shimmer 1.8s infinite;
    }}
    @keyframes shimmer {{
        0%   {{ background-position: 200% center; }}
        100% {{ background-position: -200% center; }}
    }}
    .pulse-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--accent);
        border-radius: 50%;
        animation: pulse 1.4s ease-in-out infinite;
        margin: 0 3px;
    }}
    .pulse-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .pulse-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    @keyframes pulse {{
        0%, 100% {{ transform: scale(0.8); opacity: 0.5; }}
        50%       {{ transform: scale(1.2); opacity: 1.0; }}
    }}

    /* ── Section header chip ── */
    .section-chip {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: var(--accent-dim);
        color: var(--accent-light);
        border: 1px solid var(--border-accent);
        border-radius: 999px;
        padding: 0.3rem 0.85rem;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }}

    /* ── Video output cards ── */
    .video-card-title {{
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    /* ── Status badge ── */
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    .status-badge.idle {{
        background: rgba(139,148,158,0.12);
        color: var(--text-secondary);
        border: 1px solid rgba(139,148,158,0.2);
    }}
    .status-badge.running {{
        background: rgba(99,102,241,0.15);
        color: var(--accent-light);
        border: 1px solid rgba(99,102,241,0.3);
    }}
    .status-badge.done {{
        background: rgba(63,185,80,0.12);
        color: #3FB950;
        border: 1px solid rgba(63,185,80,0.25);
    }}
    .status-badge.error {{
        background: rgba(248,81,73,0.12);
        color: #F85149;
        border: 1px solid rgba(248,81,73,0.25);
    }}
    .status-dot {{
        width: 7px; height: 7px;
        border-radius: 50%;
        background: currentColor;
    }}
    .status-dot.running {{
        animation: blink 1.2s ease-in-out infinite;
    }}
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.3; }}
    }}

    /* ── Navbar ── */
    .reid-navbar {{
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2rem;
        height: 52px;
        background: var(--nav-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-bottom: 1px solid var(--border);
        box-shadow: 0 1px 12px rgba(0,0,0,0.1);
    }}
    .nav-brand {{
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }}
    .nav-logo {{
        font-size: 1.35rem;
        line-height: 1;
    }}
    .nav-title {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }}
    .nav-version {{
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--accent-light);
        background: var(--accent-dim);
        border: 1px solid var(--border-accent);
        border-radius: 5px;
        padding: 1px 6px;
        letter-spacing: 0.04em;
    }}
    .nav-right {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }}
    .nav-meta {{
        font-size: 0.75rem;
        color: var(--text-muted);
    }}

    /* -- Divider ── */
    .reid-divider {{
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.25rem 0;
    }}

    </style>
    """, unsafe_allow_html=True)


# ── Navbar renderer ───────────────────────────────────────────────────────────

def render_navbar(run_status: str):
    status_map = {
        "idle":     ("idle",    "●", "Idle"),
        "running":  ("running", "●", "Processing"),
        "done":     ("done",    "●", "Completed"),
        "error":    ("error",   "●", "Error"),
    }
    cls, dot, label = status_map.get(run_status, status_map["idle"])

    dot_cls = "running" if run_status == "running" else ""
    st.markdown(f"""
    <div class="reid-navbar">
        <div class="nav-brand">
            <span class="nav-logo">🕵️</span>
            <span class="nav-title">Person ReID Studio</span>
            <span class="nav-version">v2.0</span>
        </div>
        <div class="nav-right">
            <span class="nav-meta">YOLO · DeepSORT · CLIP</span>
            <div class="status-badge {cls}">
                <span class="status-dot {dot_cls}"></span>
                {label}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Log console renderer ──────────────────────────────────────────────────────

def render_log_console(log_text: str):
    if not log_text:
        return

    lines_html = ""
    for raw_line in log_text.splitlines():
        line = raw_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if not line.strip():
            continue
        ll = line.lower()
        if any(k in ll for k in ["error", "failed", "critical", "cannot"]):
            cls = "error"
        elif any(k in ll for k in ["warning", "warn"]):
            cls = "warning"
        elif any(k in ll for k in ["✅", "success", "done", "loaded", "saved", "finished"]):
            cls = "success"
        elif any(k in ll for k in ["===", "---", "device:", "fps:", "processing"]):
            cls = "info"
        else:
            cls = ""
        lines_html += f'<span class="log-line {cls}">{line}</span>\n'

    st.markdown(f"""
    <div class="log-header">
        <span class="log-dot red"></span>
        <span class="log-dot yellow"></span>
        <span class="log-dot green"></span>
        <span class="log-title">Pipeline Output — stdout</span>
    </div>
    <div class="log-console">{lines_html}</div>
    """, unsafe_allow_html=True)


# ── Metric cards renderer ─────────────────────────────────────────────────────

def render_metric_cards(model_type, clip_mode, info, elapsed, alpha=None):
    best_id    = info.get("id",    "N/A")
    best_score = info.get("score", 0.0)
    score_str  = f"{best_score:.4f}" if isinstance(best_score, float) else str(best_score)
    model_str  = model_type.upper()
    if model_type == "clip":
        model_str = f"CLIP ({clip_mode})"

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-icon">🤖</div>
            <div class="metric-label">Model</div>
            <div class="metric-value accent">{model_str}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">Best Track ID</div>
            <div class="metric-value">{best_id}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">Best Score</div>
            <div class="metric-value accent">{score_str}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">⏱️</div>
            <div class="metric-label">Runtime</div>
            <div class="metric-value">{elapsed:.1f}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Loader renderer ───────────────────────────────────────────────────────────

def render_loader():
    st.markdown("""
    <style>
    @keyframes spin-ring {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes fade-in-up {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .loader-wrapper {
        animation: fade-in-up 0.4s ease both;
    }
    .spinner-ring {
        width: 64px;
        height: 64px;
        border: 4px solid var(--accent-dim);
        border-top-color: var(--accent);
        border-right-color: var(--accent-light);
        border-radius: 50%;
        animation: spin-ring 0.9s cubic-bezier(0.68,-0.55,0.27,1.55) infinite;
        margin: 0 auto 1.25rem auto;
    }
    .stage-row {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stage-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: var(--accent-dim);
        border: 1px solid var(--border-accent);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        font-weight: 600;
        color: var(--accent-light);
        letter-spacing: 0.03em;
    }
    </style>
    <div class="loader-card loader-wrapper">
        <div class="spinner-ring"></div>
        <div class="loader-title">Running ReID Pipeline</div>
        <div class="loader-subtitle" style="margin-bottom:1rem">
            Processing your video — this may take a few minutes
        </div>
        <div class="stage-row">
            <span class="stage-badge">🔍 Detecting persons</span>
            <span class="stage-badge">🎯 Tracking &amp; ReID</span>
            <span class="stage-badge">🎬 Rendering video</span>
        </div>
        <div style="margin-top:1.25rem">
            <div class="progress-bar-track">
                <div class="progress-bar-fill" style="width:100%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Session state defaults ────────────────────────────────────────────────────
if "last_run"    not in st.session_state:
    st.session_state["last_run"]    = None
if "run_status"  not in st.session_state:
    st.session_state["run_status"]  = "idle"
if "log_output"  not in st.session_state:
    st.session_state["log_output"]  = ""

# ── Theme selector lives OUTSIDE the sidebar now ─────────────────────────────
# We read it from a hidden component via session state default
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "Dark"

# ── Apply CSS theme BEFORE rendering anything visible ────────────────────────
inject_css(is_dark=(st.session_state["theme_mode"] == "Dark"))

# ── Navbar ────────────────────────────────────────────────────────────────────
render_navbar(st.session_state["run_status"])

# ── Top bar: theme toggle (compact, right-aligned) ───────────────────────────
top_r1, top_r2 = st.columns([6, 1])
with top_r2:
    new_theme = st.radio(
        "Theme", ["Dark", "Light"],
        index=0 if st.session_state["theme_mode"] == "Dark" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="theme_radio",
    )
    if new_theme != st.session_state["theme_mode"]:
        st.session_state["theme_mode"] = new_theme
        st.rerun()

# ── Main two-column layout: left = controls panel, right = output ─────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    # Wrap everything in a styled card so it looks like a sidebar
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    # ── Model selection ───────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">🔍 CLIP Query Mode</div>', unsafe_allow_html=True)
    model_type = "clip"  # CLIP-only pipeline
    clip_mode = st.selectbox(
        "CLIP query mode",
        options=["image", "text", "both"],
        help="image-only · text-only · fused image+text",
        label_visibility="collapsed",
        key="clip_mode_select",
    )
    alpha = 0.5
    if clip_mode == "both":
        st.markdown('<div class="sidebar-section-label">⚖ Alpha — Image Weight</div>', unsafe_allow_html=True)
        alpha = st.slider(
            "Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
            help="Final embedding = α·image + (1-α)·text",
            label_visibility="collapsed",
            key="alpha_slider",
        )

    st.markdown('<hr class="reid-divider">', unsafe_allow_html=True)

    # ── Input video ───────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section-label">📁 Input Videos</div>', unsafe_allow_html=True)
    uploaded_videos = st.file_uploader(
        "Video files", type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed", key="video_uploader",
        accept_multiple_files=True,
        help="Upload one or more videos to search across",
    )
    if uploaded_videos:
        n_vids = len(uploaded_videos)
        st.markdown(
            f'<div style="font-size:.78rem;color:var(--accent-light);margin-top:-.3rem">'
            f'📼 {n_vids} video{"s" if n_vids > 1 else ""} queued</div>',
            unsafe_allow_html=True,
        )

    # ── Conditional query inputs ──────────────────────────────────────────────
    needs_image = (model_type == "clip" and clip_mode in ["image", "both"])
    needs_text  = (model_type == "clip" and clip_mode in ["text", "both"])

    # Normalise — always a list
    if uploaded_videos is None:
        uploaded_videos = []

    uploaded_image = None
    text_query = ""

    if needs_image:
        st.markdown('<div class="sidebar-section-label">🖼 Query Image</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Query image", type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Upload a clear photo of the person to search for",
            key="image_uploader",
        )

    if needs_text:
        st.markdown('<div class="sidebar-section-label">💬 Text Query</div>', unsafe_allow_html=True)
        text_query = st.text_area(
            "Query text",
            placeholder="e.g. A person wearing a red jacket and black pants...",
            help="Describe the target person's appearance",
            label_visibility="collapsed",
            key="text_query_area",
        )

    # Mode hint badge
    hint_map = {
        ("clip",  "image"):  ("🔵 CLIP · Image",  "Requires: video + query image"),
        ("clip",  "text"):   ("🟣 CLIP · Text",   "Requires: video + text description"),
        ("clip",  "both"):   ("🟡 CLIP · Fusion", "Requires: video + image + text"),
    }
    hint_key = (model_type, clip_mode if model_type == "clip" else "image")
    hint_label, hint_desc = hint_map.get(hint_key, ("", ""))
    if hint_label:
        st.markdown(f"""
        <div style="margin-top:.75rem;background:var(--accent-dim);border:1px solid var(--border-accent);
                    border-radius:10px;padding:.6rem .85rem;">
            <div style="font-size:.8rem;font-weight:700;color:var(--accent-light);margin-bottom:.2rem">{hint_label}</div>
            <div style="font-size:.73rem;color:var(--text-secondary)">{hint_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="reid-divider">', unsafe_allow_html=True)
    run_button = st.button("▶  Run ReID Pipeline", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close panel-card

with right_col:

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.markdown("""
        <div class="reid-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:.5rem">📤</div>
            <div style="font-weight:700;font-size:.95rem;margin-bottom:.3rem;">Upload Media</div>
            <div style="font-size:.82rem;color:var(--text-secondary);">
                Upload a video and a query image (or text) in the sidebar to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_info2:
        st.markdown("""
        <div class="reid-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:.5rem">⚙️</div>
            <div style="font-weight:700;font-size:.95rem;margin-bottom:.3rem;">Configure Model</div>
            <div style="font-size:.82rem;color:var(--text-secondary);">
                Choose OSNet for appearance-based ReID or CLIP for image/text/multi-modal queries.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_info3:
        st.markdown("""
        <div class="reid-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:.5rem">🎬</div>
            <div style="font-weight:700;font-size:.95rem;margin-bottom:.3rem;">Review Results</div>
            <div style="font-size:.82rem;color:var(--text-secondary);">
                Annotated and cropped output videos appear here with full pipeline logs.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Run logic ────────────────────────────────────────────────────────────────
# Run immediately when button clicked — no intermediate rerun needed.
# Uploaded files are available in this same render cycle.
if run_button:
    validation_error = validate_inputs(model_type, clip_mode, uploaded_videos, uploaded_image, text_query)
    if validation_error:
        st.error(f"⚠️  {validation_error}")
    else:
        st.session_state["run_status"] = "running"
        st.session_state["log_output"] = ""

        # Show loader immediately in this same render pass
        render_navbar("running")
        render_loader()
        results_placeholder = st.empty()

        TMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

        run_tag = time.strftime("%Y%m%d_%H%M%S")
        run_output_dir = OUTPUT_ROOT / f"run_{run_tag}"

        # Save all uploaded videos
        video_paths = []
        for uv in uploaded_videos:
            vp = save_uploaded_file(uv, TMP_INPUT_DIR / f"video_{run_tag}_{uv.name}")
            video_paths.append(vp)

        image_path = None
        if uploaded_image is not None:
            image_path = save_uploaded_file(
                uploaded_image, TMP_INPUT_DIR / f"image_{run_tag}_{uploaded_image.name}"
            )

        start = time.time()
        pipeline_error = None
        all_results = None
        log_buffer = io.StringIO()  # will be replaced inside try block

        with results_placeholder.container():
            n_vids = len(video_paths)
            with st.spinner(
                f"Running ReID on {n_vids} video{'s' if n_vids > 1 else ''} — "
                "check the terminal for live progress…"
            ):
                try:
                    log_buffer = io.StringIO()
                    tee = TeeStream(log_buffer, sys.__stdout__)
                    with contextlib.redirect_stdout(tee):
                        all_results = run_reid_multi(
                            video_paths=video_paths,
                            INPUT_PERSON_IMAGE=image_path,
                            OUTPUT_DIR=str(run_output_dir),
                            model_type=model_type,
                            clip_query_mode=clip_mode,
                            INPUT_PERSON_TEXT=text_query,
                            alpha=alpha,
                        )
                except Exception as e:
                    pipeline_error = str(e)

        elapsed = time.time() - start
        log_text = log_buffer.getvalue()

        if pipeline_error:
            st.session_state["run_status"] = "error"
            st.session_state["last_run"] = {
                "error": pipeline_error,
                "log":   log_text,
                "elapsed": elapsed,
            }
        else:
            st.session_state["run_status"] = "done"
            st.session_state["last_run"] = {
                "all_results": all_results,
                "model_type":  model_type,
                "clip_mode":   clip_mode,
                "alpha":       alpha,
                "elapsed":     elapsed,
                "log":         log_text,
            }
        st.rerun()



# ── Display completed run results ─────────────────────────────────────────────
last_run = st.session_state.get("last_run")

if last_run:
    run_status = st.session_state["run_status"]

    if run_status == "error":
        st.error(f"**Pipeline failed:** {last_run.get('error', 'Unknown error')}")
        elapsed = last_run.get("elapsed", 0)
        st.caption(f"Runtime before failure: {elapsed:.1f}s")
    else:
        all_results = last_run.get("all_results", [])
        model_type  = last_run["model_type"]
        clip_mode   = last_run["clip_mode"]
        alpha       = last_run["alpha"]
        elapsed     = last_run["elapsed"]

        if not all_results:
            st.warning("⚠️  No output generated. Try another query or a different video clip.")
        else:
            # ── Overall summary metric row ────────────────────────────────
            n_total    = len(all_results)
            n_ok       = sum(1 for r in all_results if r["error"] is None and r["results"])
            n_fail     = n_total - n_ok
            best_score = max(
                (r["info"].get("score", 0.0) for r in all_results if r["info"]),
                default=0.0,
            )
            model_str = model_type.upper() if model_type != "clip" else f"CLIP ({clip_mode})"

            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-icon">🤖</div>
                    <div class="metric-label">Model</div>
                    <div class="metric-value accent">{model_str}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">📼</div>
                    <div class="metric-label">Videos</div>
                    <div class="metric-value">{n_ok} / {n_total} matched</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-label">Best Score</div>
                    <div class="metric-value accent">{best_score:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon">⏱️</div>
                    <div class="metric-label">Total Runtime</div>
                    <div class="metric-value">{elapsed:.1f}s</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Per-video tabs ────────────────────────────────────────────
            tab_labels = [
                f"{'✅' if r['error'] is None and r['results'] else '❌'} "
                f"{r['video_name'][:20]}"
                for r in all_results
            ]
            tabs = st.tabs(tab_labels)

            for tab, vr in zip(tabs, all_results):
                with tab:
                    vname = vr["video_name"]
                    verr  = vr["error"]
                    vinfo = vr["info"]
                    vres  = vr["results"]

                    if verr:
                        st.error(f"**{vname}** — {verr}")
                    else:
                        # Per-video metric row
                        v_score = vinfo.get("score", 0.0)
                        v_id    = vinfo.get("id", "N/A")
                        st.markdown(f"""
                        <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem">
                            <div class="metric-card" style="flex:1;min-width:120px">
                                <div class="metric-icon">🎯</div>
                                <div class="metric-label">Track ID</div>
                                <div class="metric-value">{v_id}</div>
                            </div>
                            <div class="metric-card" style="flex:1;min-width:120px">
                                <div class="metric-icon">📊</div>
                                <div class="metric-label">Score</div>
                                <div class="metric-value accent">{v_score:.4f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        annotated_path = vres.get("annotated") if vres else None
                        crop_path      = vres.get("crop")      if vres else None

                        vid_col1, vid_col2 = st.columns(2, gap="large")

                        with vid_col1:
                            if annotated_path and os.path.exists(annotated_path):
                                st.markdown("""
                                <div class="video-card-title">🎬 Annotated Output</div>
                                """, unsafe_allow_html=True)
                                st.video(annotated_path)
                                with open(annotated_path, "rb") as vf:
                                    st.download_button(
                                        label="⬇ Download Annotated",
                                        data=vf.read(),
                                        file_name=f"{vname}_annotated.mp4",
                                        mime="video/mp4",
                                        use_container_width=True,
                                        key=f"dl_ann_{vname}",
                                    )

                        with vid_col2:
                            if crop_path and os.path.exists(crop_path):
                                st.markdown("""
                                <div class="video-card-title">✂️ Cropped Output</div>
                                """, unsafe_allow_html=True)
                                st.video(crop_path)
                                with open(crop_path, "rb") as vf:
                                    st.download_button(
                                        label="⬇ Download Cropped",
                                        data=vf.read(),
                                        file_name=f"{vname}_crop.mp4",
                                        mime="video/mp4",
                                        use_container_width=True,
                                        key=f"dl_crop_{vname}",
                                    )

    # ── Log console ───────────────────────────────────────────────────────
    log_text = last_run.get("log", "")
    if log_text:
        st.markdown('<hr class="reid-divider">', unsafe_allow_html=True)
        st.markdown('<div class="section-chip">📋 Pipeline Log</div>', unsafe_allow_html=True)
        render_log_console(log_text)
