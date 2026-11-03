"""
AI Study Planner  -  Streamlit + Groq
Run:  streamlit run app.py
"""
import json
import os
import re
from datetime import date, datetime, time, timedelta
from html import escape

import pandas as pd
import streamlit as st

try:
    from groq import Groq
except Exception:  # library missing -> app still works with the built-in planner
    Groq = None

# ----------------------------------------------------------------------------
# Page config & constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="AI Study Planner", page_icon="📚", layout="wide")

MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TYPES = ["Learn", "Practice", "Revise", "Mock test"]
PALETTE = ["#2e7d32", "#1565c0", "#ef6c00", "#6a1b9a", "#00838f", "#c62828", "#827717", "#4527a0"]
MAX_DAYS = 42      # planning window (days) - keeps AI answers fast and reliable
MAX_SUBJECTS = 8
BREAK_MIN = 10     # break between sessions (minutes)

# ----------------------------------------------------------------------------
# Study-friendly styling (large, high-contrast fonts, calm colours)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');
html, body, .stApp { font-family: 'Nunito','Segoe UI',Arial,sans-serif; }
.stApp { background: linear-gradient(180deg,#f7f4ea 0%,#edf5ee 100%); }
.block-container { padding-top: 1.6rem; max-width: 1250px; }

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { font-size: 1.08rem; line-height: 1.65; color: #1f2937; }
[data-testid="stCaptionContainer"] p { color: #3f4b5b !important; font-size: 1rem !important; }
h1, h2, h3, h4 { color: #14532d !important; font-weight: 800 !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.3rem !important; }

label, [data-testid="stWidgetLabel"] p {
    color: #1f2937 !important; font-size: 1.05rem !important; font-weight: 700 !important; }

section[data-testid="stSidebar"] { background: #e3eee0; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#1f2937; }

div[data-baseweb="input"] input, div[data-baseweb="base-input"] input, textarea,
div[data-baseweb="select"] > div {
    background: #ffffff !important; color: #1f2937 !important; font-size: 1.05rem !important; }

.stButton > button, .stDownloadButton > button {
    background: #15803d; color: #ffffff; border: none; border-radius: 12px;
    padding: .65rem 1.3rem; font-size: 1.1rem; font-weight: 800; }
.stButton > button p, .stDownloadButton > button p { color: #ffffff !important; font-size: 1.1rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #166534; color:#fff; }

.stTabs [data-baseweb="tab"] p { font-size: 1.1rem !important; font-weight: 800 !important; color:#1f2937 !important; }
[data-testid="stMetricValue"] { color: #14532d !important; font-weight: 800; }
[data-testid="stMetricLabel"] p { color: #1f2937 !important; font-weight: 700; }
[data-testid="stExpander"] { background: #ffffff; border: 1px solid #cfe3d0; border-radius: 14px; }
[data-testid="stExpander"] summary p { color: #14532d !important; font-weight: 800; font-size: 1.1rem; }

.hero { background: linear-gradient(120deg,#166534,#0f766e); border-radius: 20px;
        padding: 1.6rem 2rem; margin-bottom: 1.2rem; box-shadow: 0 6px 18px rgba(22,101,52,.25); }
.hero h1 { color:#ffffff !important; margin:0; font-size: 2.3rem !important; }
.hero p  { color:#e8fff0 !important; font-size: 1.15rem !important; margin:.4rem 0 0 0; }

.day-card { background:#ffffff; border:1px solid #d6e6d6; border-radius:16px;
            padding: .9rem 1.1rem; margin: .7rem 0; }
.day-head { color:#14532d; font-weight:800; font-size:1.18rem; margin-bottom:.5rem; }
.day-head span { color:#475569; font-weight:600; font-size:1rem; }
.sess { display:flex; flex-wrap:wrap; align-items:center; gap:.6rem; background:#f8fbf7;
        border-left:6px solid #2e7d32; border-radius:10px; padding:.55rem .8rem; margin:.4rem 0; }
.sess.done { opacity:.55; text-decoration: line-through; }
.s-time { font-weight:800; color:#0f172a; min-width:118px; font-size:1.05rem; }
.s-main { color:#1f2937; font-size:1.05rem; flex:1; min-width:200px; }
.chip { background:#e1efe1; color:#14532d; border-radius:999px; padding:.15rem .7rem;
        font-size:.9rem; font-weight:700; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def txt(x) -> str:
    """Safe string from any cell value (handles None / NaN)."""
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def to_time(x) -> time:
    if isinstance(x, datetime):
        return x.time().replace(second=0, microsecond=0)
    if isinstance(x, time):
        return x.replace(second=0, microsecond=0)
    if isinstance(x, str):
        try:
            return datetime.strptime(x.strip()[:5], "%H:%M").time()
        except Exception:
            pass
    return time(9, 0)


def color_for(subject: str) -> str:
    return PALETTE[sum(ord(c) for c in txt(subject).lower()) % len(PALETTE)]


def eligible(sub: dict, d: date, start: date) -> bool:
    """A subject can be studied on day d if d is before its exam (or the exam is today)."""
    return d < sub["exam"] or (d == sub["exam"] == start)


def get_api_key() -> str:
    key = st.session_state.get("api_key_input", "").strip()
    if key:
        return key
    try:
        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
    except Exception:
        key = ""
    return key or os.environ.get("GROQ_API_KEY", "").strip()


# ----------------------------------------------------------------------------
# Input cleaning
# ----------------------------------------------------------------------------
def clean_subjects(df: pd.DataFrame, start: date):
    subs, warns, seen = [], [], set()
    for _, r in df.iterrows():
        name = txt(r.get("Subject"))
        if not name:
            continue
        if name.lower() in seen:
            warns.append(f"Duplicate subject '{name}' ignored.")
            continue
        exam = pd.to_datetime(r.get("Exam Date"), errors="coerce")
        if pd.isna(exam):
            warns.append(f"'{name}' has no valid exam date - skipped.")
            continue
        exam = exam.date()
        if exam < start:
            warns.append(f"'{name}' exam date is before the plan start date - skipped.")
            continue
        conf = pd.to_numeric(r.get("Confidence (1-5)"), errors="coerce")
        conf = 3 if pd.isna(conf) else int(max(1, min(5, round(float(conf)))))
        weak = []
        for t in re.split(r"[,;\n]+", txt(r.get("Weak Topics"))):
            t = t.strip()
            if t and t.lower() not in [w.lower() for w in weak]:
                weak.append(t[:80])
        seen.add(name.lower())
        subs.append({"name": name[:60], "exam": exam, "conf": conf, "weak": weak[:12]})
    if len(subs) > MAX_SUBJECTS:
        warns.append(f"Only the first {MAX_SUBJECTS} subjects are used.")
        subs = subs[:MAX_SUBJECTS]
    return subs, warns


def hours_list(df: pd.DataFrame):
    vals = pd.to_numeric(df["Hours"], errors="coerce").fillna(0).clip(0, 16).tolist()
    vals = (vals + [0] * 7)[:7]
    return [float(v) for v in vals]


def build_capacity(subs, hours, start: date):
    last = max(s["exam"] for s in subs)
    end = min(last, start + timedelta(days=MAX_DAYS - 1))
    cap, d = {}, start
    while d <= end:
        mins = int(hours[d.weekday()] * 60)
        mins -= mins % 5
        if mins >= 20 and any(eligible(s, d, start) for s in subs):
            cap[d] = mins
        d += timedelta(days=1)
    return cap, last > end


# ----------------------------------------------------------------------------
# AI (Groq) planner
# ----------------------------------------------------------------------------
def parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("AI reply did not contain JSON")
        return json.loads(m.group(0))


def call_groq(api_key: str, model: str, subs, capacity, session_len: int, start: date) -> dict:
    client = Groq(api_key=api_key)
    payload = {
        "today": start.isoformat(),
        "subjects": [
            {
                "name": s["name"],
                "exam_date": s["exam"].isoformat(),
                "days_until_exam": (s["exam"] - start).days,
                "confidence_1_to_5": s["conf"],
                "weak_topics": s["weak"],
            }
            for s in subs
        ],
        "days": [
            {"date": d.isoformat(), "weekday": d.strftime("%a"), "available_minutes": m}
            for d, m in capacity.items()
        ],
    }
    system = "You are an expert study coach and timetable planner. Reply with valid JSON only."
    user = f"""Create a study timetable from this data:
{json.dumps(payload)}

Rules:
1. Use ONLY dates listed in "days". Total minutes of sessions on a date must NOT exceed that date's available_minutes. Use about 90-100% of the time.
2. Each session is between 25 and {session_len + 30} minutes (multiples of 5).
3. Use the subject name EXACTLY as given. Never schedule a subject on or after its exam date.
4. Give more time to subjects with low confidence and earlier exams. Cover every weak topic at least twice (first "Learn", then "Practice"/"Revise").
5. In the last 2 days before an exam focus on "Revise" and "Mock test" sessions.
6. Mix subjects across a day for variety. Topics must be specific (e.g. "Integration by parts - 10 practice problems").

Return ONLY this JSON shape:
{{"advice": "2-3 sentence personalised study strategy",
  "sessions": [{{"d": "YYYY-MM-DD", "s": "Subject", "t": "specific topic", "m": 45, "k": "Learn|Practice|Revise|Mock test"}}]}}"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=8000,
        response_format={"type": "json_object"},
    )
    return parse_json(resp.choices[0].message.content or "")


def norm_kind(k: str) -> str:
    k = txt(k).lower()
    for t in TYPES:
        if t.lower()[:4] in k:
            return t
    return "Practice"


def validate_sessions(raw, subs, capacity, start: date):
    """Repair / filter AI output so the timetable can never contain invalid rows."""
    if not isinstance(raw, list):
        return []
    by_name = {s["name"].lower(): s for s in subs}
    left, out = dict(capacity), []
    for it in raw:
        if not isinstance(it, dict):
            continue
        try:
            d = datetime.strptime(txt(it.get("d") or it.get("date"))[:10], "%Y-%m-%d").date()
        except Exception:
            continue
        if d not in left:
            continue
        nm = txt(it.get("s") or it.get("subject")).lower()
        sub = by_name.get(nm) or next(
            (v for k, v in by_name.items() if nm and (nm in k or k in nm)), None)
        if not sub or not eligible(sub, d, start):
            continue
        try:
            m = int(round(float(it.get("m") or it.get("minutes"))))
        except Exception:
            continue
        m = min(max(m, 20), 180, left[d])
        m -= m % 5
        if m < 15:
            continue
        left[d] -= m
        out.append({
            "date": d, "subject": sub["name"],
            "topic": (txt(it.get("t") or it.get("topic")) or "Revision")[:120],
            "minutes": m, "type": norm_kind(it.get("k") or it.get("type")),
        })
    return out


# ----------------------------------------------------------------------------
# Built-in smart scheduler (fallback - guarantees a plan even without AI)
# ----------------------------------------------------------------------------
def local_plan(subs, capacity, session_len: int, start: date):
    assigned = {s["name"]: 0 for s in subs}
    idx = {s["name"]: 0 for s in subs}
    extras = ["Practice problems", "Past-paper questions", "Summary notes & flashcards"]
    out = []
    for d, cap in capacity.items():
        elig = [s for s in subs if eligible(s, d, start)]
        remaining = cap
        while remaining >= 20 and elig:
            def score(s):
                left_days = max((s["exam"] - d).days, 1)
                weight = (6 - s["conf"]) * (1 + 3 / left_days)
                return assigned[s["name"]] / weight
            s = min(elig, key=score)
            m = min(session_len, remaining)
            if remaining - m < 20:
                m = remaining
            m -= m % 5
            seq = s["weak"] + extras
            i = idx[s["name"]]
            topic = seq[i % len(seq)]
            days_left = (s["exam"] - d).days
            if days_left == 1:
                kind, topic = "Revise", f"Final revision: {topic}"
            elif days_left == 2:
                kind = "Mock test"
                topic = f"Timed mock test + review: {topic}"
            else:
                kind = "Learn" if i < len(s["weak"]) else "Practice"
            idx[s["name"]] += 1
            assigned[s["name"]] += m
            remaining -= m
            out.append({"date": d, "subject": s["name"], "topic": topic, "minutes": m, "type": kind})
    return out


def sessions_to_df(sessions, day_start: time) -> pd.DataFrame:
    rows, cur_date, cur = [], None, None
    for s in sorted(sessions, key=lambda x: x["date"]):
        if s["date"] != cur_date:
            cur_date, cur = s["date"], datetime.combine(s["date"], day_start)
        rows.append({
            "Done": False, "Date": s["date"], "Start": cur.time().replace(second=0, microsecond=0),
            "Minutes": int(s["minutes"]), "Subject": s["subject"],
            "Topic": s["topic"], "Type": s["type"],
        })
        cur += timedelta(minutes=s["minutes"] + BREAK_MIN)
    cols = ["Done", "Date", "Start", "Minutes", "Subject", "Topic", "Type"]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
# Plan post-processing / views / export
# ----------------------------------------------------------------------------
def clean_plan(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d["Minutes"] = pd.to_numeric(d["Minutes"], errors="coerce")
    d["Subject"] = d["Subject"].map(txt)
    d = d.dropna(subset=["Date", "Minutes"])
    d = d[(d["Subject"] != "") & (d["Minutes"] > 0)].copy()
    d["Minutes"] = d["Minutes"].astype(int)
    d["Done"] = d["Done"] == True  # noqa: E712  (NaN -> False)
    d["Start"] = d["Start"].map(to_time)
    d["Topic"] = d["Topic"].map(txt)
    d["Type"] = d["Type"].map(lambda x: txt(x) or "Practice")
    d["D"] = d["Date"].dt.date
    d["_s"] = d["Start"].map(lambda t: t.strftime("%H:%M"))
    d = d.sort_values(["D", "_s"], kind="stable").reset_index(drop=True)
    return d


def end_str(day: date, start: time, minutes: int) -> str:
    return (datetime.combine(day, start) + timedelta(minutes=int(minutes))).strftime("%H:%M")


def render_calendar(plan: pd.DataFrame):
    if plan.empty:
        st.info("Your timetable is empty. Add rows in the editor tab.")
        return
    plan = plan.copy()
    plan["W"] = plan["D"].map(lambda x: x - timedelta(days=x.weekday()))
    for n, (wk, wdf) in enumerate(plan.groupby("W")):
        with st.expander(f"📅 Week of {wk:%d %b %Y}", expanded=(n == 0)):
            for day, ddf in wdf.groupby("D"):
                total = int(ddf["Minutes"].sum())
                rows = []
                for _, r in ddf.iterrows():
                    cls = "sess done" if r["Done"] else "sess"
                    rows.append(
                        f'<div class="{cls}" style="border-left-color:{color_for(r["Subject"])}">'
                        f'<div class="s-time">{r["_s"]} – {end_str(day, r["Start"], r["Minutes"])}</div>'
                        f'<div class="s-main"><b>{escape(r["Subject"])}</b> · {escape(r["Topic"])}</div>'
                        f'<span class="chip">{escape(r["Type"])}</span>'
                        f'<span class="chip">{int(r["Minutes"])} min</span></div>'
                    )
                st.markdown(
                    f'<div class="day-card"><div class="day-head">{day:%A, %d %B} '
                    f'<span>· {total // 60}h {total % 60:02d}m planned</span></div>{"".join(rows)}</div>',
                    unsafe_allow_html=True,
                )


def make_ics(plan: pd.DataFrame) -> str:
    def esc(s):
        return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", " ")
    now = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//AI Study Planner//EN", "CALSCALE:GREGORIAN"]
    for i, r in plan.iterrows():
        s = datetime.combine(r["D"], r["Start"])
        e = s + timedelta(minutes=int(r["Minutes"]))
        lines += [
            "BEGIN:VEVENT", f"UID:{i}-{s:%Y%m%dT%H%M}@aistudyplanner", f"DTSTAMP:{now}",
            f"DTSTART:{s:%Y%m%dT%H%M%S}", f"DTEND:{e:%Y%m%dT%H%M%S}",
            f"SUMMARY:{esc(r['Subject'] + ' - ' + r['Type'])}", f"DESCRIPTION:{esc(r['Topic'])}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"


# ----------------------------------------------------------------------------
# Session state defaults
# ----------------------------------------------------------------------------
def init_state():
    t = date.today()
    st.session_state.setdefault("subjects_base", pd.DataFrame({
        "Subject": ["Mathematics", "Physics", "Chemistry"],
        "Exam Date": [t + timedelta(days=14), t + timedelta(days=18), t + timedelta(days=21)],
        "Weak Topics": ["Integration, Probability", "Electromagnetism, Optics",
                        "Organic reactions, Chemical equilibrium"],
        "Confidence (1-5)": [2, 3, 2],
    }))
    st.session_state.setdefault("hours_base", pd.DataFrame({
        "Day": WEEKDAYS, "Hours": [2.0, 2.0, 2.0, 2.0, 1.5, 4.0, 3.0]}))
    st.session_state.setdefault("plan_df", None)
    st.session_state.setdefault("plan_ver", 0)
    st.session_state.setdefault("advice", "")
    st.session_state.setdefault("source", "")
    st.session_state.setdefault("exams", {})


init_state()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.text_input("Groq API key", type="password", key="api_key_input",
                  placeholder="gsk_...", help="Or set GROQ_API_KEY as environment variable / Streamlit secret.")
    use_ai = st.checkbox("Use AI (Groq) to build the plan", value=True)
    model = st.selectbox("AI model", MODELS, index=0)
    start_date = st.date_input("Plan starts on", value=date.today(), min_value=date.today())
    day_start = st.time_input("Daily study start time", value=time(17, 0), step=900)
    session_len = st.slider("Preferred session length (min)", 25, 90, 45, step=5)
    st.markdown("---")
    if st.button("🗑️ Clear timetable"):
        st.session_state.plan_df = None
        st.session_state.advice = ""
    st.caption("💡 Tip: short sessions (25–50 min) with breaks are the most effective way to study.")

# ----------------------------------------------------------------------------
# Header + inputs
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>📚 AI Study Planner</h1>'
    "<p>Tell me your subjects, exam dates, weak topics and free hours – I will build a smart, "
    "editable study timetable for you.</p></div>",
    unsafe_allow_html=True,
)

st.markdown("## 1️⃣ Your subjects")
st.caption("Double-click a cell to edit. Use the empty last row to add a subject. "
           "Separate weak topics with commas. Confidence: 1 = very weak, 5 = very strong.")
subjects_edit = st.data_editor(
    st.session_state.subjects_base,
    key="subjects_editor", num_rows="dynamic", hide_index=True, use_container_width=True,
    column_config={
        "Subject": st.column_config.TextColumn("Subject", required=True, width="medium"),
        "Exam Date": st.column_config.DateColumn("Exam Date", format="YYYY-MM-DD", required=True),
        "Weak Topics": st.column_config.TextColumn("Weak Topics (comma separated)", width="large"),
        "Confidence (1-5)": st.column_config.NumberColumn(
            "Confidence (1-5)", min_value=1, max_value=5, step=1, default=3),
    },
)

st.markdown("## 2️⃣ Hours you can study")
st.caption("How many hours can you study on each weekday?")
hc1, hc2 = st.columns([1, 2])
with hc1:
    hours_edit = st.data_editor(
        st.session_state.hours_base, key="hours_editor", hide_index=True,
        use_container_width=True, disabled=["Day"],
        column_config={"Hours": st.column_config.NumberColumn(
            "Hours available", min_value=0.0, max_value=16.0, step=0.5, format="%.1f")},
    )
with hc2:
    weekly = sum(hours_list(hours_edit))
    st.metric("Total study hours per week", f"{weekly:.1f} h")
    st.caption(f"Plans cover up to {MAX_DAYS} days from the start date.")

st.markdown("## 3️⃣ Generate")
go = st.button("✨ Generate my study timetable", use_container_width=True)

# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
if go:
    subs, warns = clean_subjects(subjects_edit, start_date)
    for w in warns:
        st.warning(w)
    if not subs:
        st.error("Please enter at least one subject with a valid exam date (today or later).")
    else:
        capacity, truncated = build_capacity(subs, hours_list(hours_edit), start_date)
        if not capacity:
            st.error("No study time is available before your exams. Increase your daily hours "
                     "or move the exam dates further away.")
        else:
            if truncated:
                st.info(f"Exams are far away - the timetable covers the next {MAX_DAYS} days. "
                        "Generate again later to plan the following weeks.")
            sessions, advice, source = [], "", "Built-in smart scheduler"
            key = get_api_key()
            if use_ai and Groq is None:
                st.warning("The 'groq' package is not installed - using the built-in scheduler.")
            elif use_ai and not key:
                st.info("No Groq API key found - using the built-in scheduler. "
                        "Paste your key in the sidebar to enable AI planning.")
            elif use_ai:
                total_cap = sum(capacity.values())
                with st.spinner("🤖 The AI is designing your timetable..."):
                    for mdl in dict.fromkeys([model, MODELS[1]]):
                        try:
                            data = call_groq(key, mdl, subs, capacity, session_len, start_date)
                            cand = validate_sessions(data.get("sessions", []), subs, capacity, start_date)
                            covered = {c["subject"] for c in cand}
                            used = sum(c["minutes"] for c in cand)
                            if used >= 0.5 * total_cap and len(covered) == len(subs):
                                sessions, advice, source = cand, txt(data.get("advice")), f"Groq · {mdl}"
                                break
                            st.warning(f"The AI answer from {mdl} was incomplete.")
                        except Exception as e:  # network, auth, rate-limit, bad JSON ...
                            st.warning(f"AI request failed ({type(e).__name__}): {str(e)[:160]}")
                if not sessions:
                    st.info("Switching to the built-in smart scheduler so you still get a plan.")
            if not sessions:
                sessions = local_plan(subs, capacity, session_len, start_date)
                source = "Built-in smart scheduler"
            st.session_state.plan_df = sessions_to_df(sessions, day_start)
            st.session_state.plan_ver += 1
            st.session_state.advice = advice
            st.session_state.source = source
            st.session_state.exams = {s["name"].lower(): s["exam"] for s in subs}
            st.success("✅ Your timetable is ready! Edit it below as you like.")

# ----------------------------------------------------------------------------
# Timetable
# ----------------------------------------------------------------------------
if st.session_state.plan_df is not None:
    st.markdown("---")
    st.markdown("## 📅 Your study timetable")
    st.caption(f"Created by: {st.session_state.source}")
    if st.session_state.advice:
        st.info(f"🧠 **Coach's advice:** {st.session_state.advice}")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📝 Edit timetable", "🗓️ Day-by-day view", "📈 Progress", "⬇️ Export"])

    with tab1:
        st.caption("Double-click any cell to change it. Tick ✔ when a session is done. "
                   "Add a row with the empty bottom row; select a row and press Delete to remove it.")
        edited_plan = st.data_editor(
            st.session_state.plan_df,
            key=f"plan_editor_{st.session_state.plan_ver}",
            num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={
                "Done": st.column_config.CheckboxColumn("✔ Done", default=False),
                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", required=True),
                "Start": st.column_config.TimeColumn("Start", format="HH:mm", step=300),
                "Minutes": st.column_config.NumberColumn(
                    "Minutes", min_value=5, max_value=480, step=5, default=45),
                "Subject": st.column_config.TextColumn("Subject", required=True),
                "Topic": st.column_config.TextColumn("Topic", width="large"),
                "Type": st.column_config.SelectboxColumn("Type", options=TYPES, default="Practice"),
            },
        )

    plan = clean_plan(edited_plan)

    with tab2:
        render_calendar(plan)

    with tab3:
        if plan.empty:
            st.info("Nothing to show yet.")
        else:
            total = int(plan["Minutes"].sum())
            done = int(plan.loc[plan["Done"], "Minutes"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Planned", f"{total / 60:.1f} h")
            c2.metric("Completed", f"{done / 60:.1f} h")
            c3.metric("Sessions done", f"{int(plan['Done'].sum())} / {len(plan)}")
            st.progress(done / total if total else 0.0, text=f"Overall progress: {100 * done / total:.0f}%")
            st.markdown("### By subject")
            for sub, g in plan.groupby("Subject"):
                t_all = int(g["Minutes"].sum())
                t_done = int(g.loc[g["Done"], "Minutes"].sum())
                exam = st.session_state.exams.get(sub.lower())
                extra = f" · exam in {(exam - date.today()).days} days ({exam:%d %b})" if exam else ""
                st.progress(t_done / t_all if t_all else 0.0,
                            text=f"{sub}: {t_done / 60:.1f} / {t_all / 60:.1f} h{extra}")
            st.markdown("### Hours per subject")
            st.bar_chart((plan.groupby("Subject")["Minutes"].sum() / 60.0).rename("Hours"))

    with tab4:
        if plan.empty:
            st.info("Nothing to export yet.")
        else:
            exp = pd.DataFrame({
                "Date": plan["D"].map(lambda x: x.isoformat()),
                "Start": plan["_s"],
                "End": [end_str(r["D"], r["Start"], r["Minutes"]) for _, r in plan.iterrows()],
                "Minutes": plan["Minutes"], "Subject": plan["Subject"],
                "Topic": plan["Topic"], "Type": plan["Type"],
                "Done": plan["Done"].map(lambda b: "Yes" if b else "No"),
            })
            st.write("Download your timetable as a spreadsheet or import it into Google / Apple / Outlook calendar.")
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download CSV", exp.to_csv(index=False).encode("utf-8"),
                               file_name="study_timetable.csv", mime="text/csv",
                               use_container_width=True)
            d2.download_button("📆 Download calendar (.ics)", make_ics(plan).encode("utf-8"),
                               file_name="study_timetable.ics", mime="text/calendar",
                               use_container_width=True)
else:
    st.info("👆 Fill in the details above and press **Generate my study timetable**.")
