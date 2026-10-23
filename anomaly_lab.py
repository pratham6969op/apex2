"""NuclearRad Anomaly Lab: simulate sensor and core anomalies around a reactor with 5 sensors.

Run with: streamlit run anomaly_lab.py

Self-contained: the simulator and the peer-consensus detector live in this file, so it does not
need the `nuclearrad` module. Same visual language as main.py.
"""
import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")
st.set_page_config(page_title="NuclearRad | Anomaly Lab", page_icon="☢", layout="wide")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
NAMES = ["Rad1", "Rad2", "Rad3", "Rad4", "Rad5"]
SENSOR_ANGLES = [90, 18, -54, -126, 162]  # degrees, pentagon around the vessel
N_SAMPLES = 240
WARMUP = 40
LEAK_LEVEL = 112.0      # network median (% of baseline) that counts as a shared rise
ALARM_LEVEL = 1.3       # classic threshold alarm: 130% of baseline
BASE_LEVELS = np.array([100.0, 94.0, 106.0, 98.0, 103.0])
UNCERTAIN_TRUST = 70

MINT, AMBER, RED = "#8ee2bd", "#f3b969", "#ff746c"

# Anomalies that can be injected into a single sensor.
SENSOR_ANOMALIES = {
    "Healthy": dict(
        color="#4f7d6c",
        what="The sensor behaves normally and follows the other four within calibration noise.",
        sees="Reading sits within a few percent of peer consensus.",
        action="NONE",
    ),
    "Spike": dict(
        color="#f3b969",
        what="Short, sharp jumps that fall back to normal. Typical of electrical interference.",
        sees="One sample leaps far from the peers, then the next sample is normal again.",
        action="RECHECK",
    ),
    "Drift": dict(
        color="#ff9f6c",
        what="The reading creeps away from the truth minute by minute as the detector ages.",
        sees="The gap to peer consensus grows steadily while the other sensors agree with each other.",
        action="RECALIBRATE",
    ),
    "Offset": dict(
        color="#e07b9b",
        what="A sudden, permanent step in the reading, like a shifted calibration.",
        sees="The gap to peers jumps once and then stays flat.",
        action="RECALIBRATE",
    ),
    "Stuck": dict(
        color="#7aa7ff",
        what="The sensor freezes on one value and no longer responds to the core.",
        sees="Identical readings sample after sample. Real signals always wobble.",
        action="REPLACE",
    ),
    "Noisy": dict(
        color="#c59bff",
        what="The signal becomes jittery, as with a failing cable or poor shielding.",
        sees="Sample-to-sample jitter is several times larger than during warm-up.",
        action="INSPECT CABLE",
    ),
    "Dropout": dict(
        color="#8a9690",
        what="Readings go missing at random, as if the link keeps cutting out.",
        sees="Samples arrive empty, so the sensor cannot vote in peer consensus.",
        action="RESTORE SIGNAL",
    ),
}
FAULT_COLORS = {k: v["color"] for k, v in SENSOR_ANOMALIES.items()}
FAULT_COLORS["Deviation"] = RED
FAULT_ORDER = list(FAULT_COLORS)
ACTIONS = {k: v["action"] for k, v in SENSOR_ANOMALIES.items()}
ACTIONS["Deviation"] = "INSPECT"

# Anomalies that affect the whole core (all sensors see the change together).
CORE_EVENTS = {
    "None": dict(what="Steady operation.", sees="Nothing unusual."),
    "Coolant leak (real radiation rise)": dict(
        what="Radiation really does rise everywhere, which is a plant problem, not a sensor problem.",
        sees="All five sensors climb together. The AI escalates instead of blaming a sensor.",
    ),
    "Power transient (harmless)": dict(
        what="A short, small power swing that rises and falls on its own.",
        sees="All sensors rise together but stay below the leak level, so no alarm is needed.",
    ),
}


# --------------------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------------------
def generate(sensor_faults, core_event, onset, severity, seed, noise):
    """Return (readings, truth) arrays shaped (samples, sensors)."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_SAMPLES)
    shared = 1 + 0.010 * np.sin(t / 25.0) + 0.005 * np.sin(t / 9.0 + 1.0)

    event = np.ones(N_SAMPLES)
    if core_event.startswith("Coolant leak"):
        ramp = np.where(t >= onset, 1 - np.exp(-(t - onset) / 20.0), 0.0)
        event = 1 + 0.15 * severity * ramp
    elif core_event.startswith("Power transient"):
        event = 1 + 0.08 * (severity / 3.0) * np.exp(-0.5 * ((t - onset - 20) / 10.0) ** 2)

    truth = BASE_LEVELS[None, :] * (shared * event)[:, None]
    x = truth * (1 + rng.normal(0, noise, truth.shape))

    after = t >= onset
    for i, fault in enumerate(sensor_faults):
        if fault == "Spike":
            for s in range(onset, N_SAMPLES, 12):
                x[s, i] += truth[s, i] * 0.25 * severity
        elif fault == "Drift":
            x[after, i] += truth[after, i] * 0.004 * severity * (t[after] - onset)
        elif fault == "Offset":
            x[after, i] += truth[after, i] * 0.06 * severity
        elif fault == "Stuck":
            x[onset:, i] = x[onset - 1, i]
        elif fault == "Noisy":
            x[after, i] += truth[after, i] * rng.normal(0, 0.04 * severity, after.sum())
        elif fault == "Dropout":
            gone = rng.random(N_SAMPLES) < min(0.9, 0.18 * severity)
            gone[:onset] = False
            x[gone, i] = np.nan
    return x, truth


# --------------------------------------------------------------------------------------
# Detector: peer consensus plus simple behavior rules
# --------------------------------------------------------------------------------------
def assess(x):
    n, m = x.shape
    base = np.nanmean(x[:WARMUP], axis=0)
    y = x / base * 100.0                              # every sensor normalised to 100 at baseline
    med = np.nanmedian(y, axis=1)                     # peer consensus
    ratio = y / med[:, None] * 100.0 - 100.0          # % deviation from consensus
    sigma = np.maximum(np.nanstd(ratio[:WARMUP], axis=0), 0.4)
    # Typical sample-to-sample jitter, pooled over the sensors (same detector model), via second differences.
    dmed = np.full(m, max(np.nanmedian(np.abs(np.diff(ratio[:WARMUP], 2, axis=0))), 0.2))

    fault = np.full((n, m), "Healthy", dtype=object)
    trust = np.full((n, m), 100.0)
    target = {"Healthy": 100, "Spike": 55, "Deviation": 30, "Drift": 25, "Offset": 25,
              "Noisy": 35, "Stuck": 15, "Dropout": 5}

    for i in range(m):
        run, tr = 0, 100.0
        for t in range(n):
            label = "Healthy"
            z = ratio[t, i] / sigma[i]
            if np.isnan(x[t, i]):
                label, run = "Dropout", 0
            else:
                if abs(z) > 3.5:
                    run += 1
                elif abs(z) < 2:
                    run = 0
                window = x[max(0, t - 5): t + 1, i]
                if t >= 5 and not np.isnan(window).any() and np.ptp(window) == 0:
                    label = "Stuck"
                elif t >= WARMUP + 14 and _jitter(ratio[t - 14: t + 1, i]) > 2.7 * dmed[i]:
                    label = "Noisy"
                elif run >= 3:
                    label = _classify_deviation(ratio[:t + 1, i], run, sigma[i])
                elif abs(z) > 6:
                    label = "Spike"
            if t < WARMUP:
                label = "Healthy"
            fault[t, i] = label
            tr = 0.6 * tr + 0.4 * target[label]
            trust[t, i] = tr

    shared_rise = med >= LEAK_LEVEL
    leak = shared_rise.copy()
    leak[1:] &= shared_rise[:-1]
    leak[2:] &= shared_rise[:-2]
    leak[:2] = False

    verdict, kind, action, reason = [], [], [], []
    for t in range(n):
        faulty = [(NAMES[i], fault[t, i]) for i in range(m) if fault[t, i] != "Healthy"]
        if leak[t]:
            extra = f" · plus sensor fault on {', '.join(f'{a} ({b})' for a, b in faulty)}" if faulty else ""
            verdict.append("Possible radiation leak" + extra)
            kind.append("escalation")
            action.append("ESCALATE")
            reason.append(f"All sensors rise together: network median is {med[t]:.0f}% of baseline.")
        elif faulty:
            verdict.append("Sensor fault: " + ", ".join(f"{a} ({b})" for a, b in faulty))
            kind.append("attention")
            action.append(ACTIONS[faulty[0][1]])
            reason.append("The flagged reading disagrees with the peer consensus, which the other sensors share.")
        else:
            verdict.append("Normal operation")
            kind.append("ok")
            action.append("NONE")
            reason.append("Shared rise below the leak level, consistent with a harmless power change."
                          if med[t] > 104 else "All five sensors agree with peer consensus.")

    corrected = np.where(fault != "Healthy", med[:, None] * base[None, :] / 100.0, x)
    return dict(y=y, med=med, base=base, fault=fault, trust=trust, corrected=corrected,
                leak=leak, verdict=verdict, kind=kind, action=action, reason=reason)


def _jitter(window):
    d = np.abs(np.diff(window, 2))  # second difference: a steady drift cancels out
    return np.nan if np.isnan(d).all() else np.nanmedian(d)


def _classify_deviation(ratio_hist, run, sigma):
    if run < 15:
        return "Deviation"
    seg = ratio_hist[-15:]
    ok = ~np.isnan(seg)
    if ok.sum() < 10:
        return "Deviation"
    slope = np.polyfit(np.arange(15)[ok], seg[ok], 1)[0]
    return "Drift" if abs(slope) > 0.24 * sigma else "Offset"


# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Manrope:wght@400;500;600;700&display=swap');
:root { --ink:#e8eee9; --muted:#91a19c; --line:#263632; --panel:#101a18; --mint:#8ee2bd; --amber:#f3b969; --red:#ff746c; }
.stApp { background: radial-gradient(ellipse at 62% 0%, #1c302b 0%, #111b19 42%, #0b1211 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
[data-testid="stSidebar"] { background:#101a18; border-right:1px solid var(--line); }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Barlow Condensed',sans-serif !important; letter-spacing:0 !important; }
h1 { font-size:2.8rem !important; font-weight:700 !important; margin-bottom:0 !important; }
.eyebrow { color:var(--mint); font:600 11px 'IBM Plex Mono',monospace; letter-spacing:1.4px; text-transform:uppercase; }
.subtle { color:var(--muted); font-size:13px; }
.side-head { color:#dfe9e3; font:600 19px 'Barlow Condensed',sans-serif; margin:14px 0 2px; }
.status-strip { padding:16px 20px; background:#14211e; border:1px solid #2a4039; border-left:4px solid var(--mint); border-radius:5px; margin:10px 0 18px; }
.sensor-card { background:linear-gradient(145deg,#14201d,#101816); border:1px solid #293a35; border-radius:5px; padding:13px 14px; min-height:104px; }
.sensor-label { color:#aab9b3; font:500 11px 'IBM Plex Mono',monospace; }
.sensor-value { color:#eef5ef; font:600 24px 'IBM Plex Mono',monospace; margin:8px 0 3px; }
.sensor-foot { color:#91a19c; font:500 10px 'IBM Plex Mono',monospace; }
.section-label { color:#dfe9e3; font:600 22px 'Barlow Condensed',sans-serif; margin:6px 0 2px; }
.pill-row { display:flex; flex-wrap:wrap; gap:8px; margin:2px 0 18px; }
.pill { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:99px; background:#13201c;
        border:1px solid #2a3d37; font:500 12px 'IBM Plex Mono',monospace; color:#cbd8d2; }
.pill i { width:9px; height:9px; border-radius:50%; display:inline-block; }
.cat-card { background:linear-gradient(145deg,#14201d,#101816); border:1px solid #293a35; border-radius:5px; padding:14px 16px; height:100%; }
.cat-card b { font:600 20px 'Barlow Condensed',sans-serif; }
.cat-card p { margin:6px 0 0; font-size:13px; color:#b4c2bc; line-height:1.5; }
.cat-card .sig { color:#8ea199; font-size:12px; }
table.report { width:100%; border-collapse:collapse; font-size:13px; }
table.report th { text-align:left; color:#91a19c; font:500 11px 'IBM Plex Mono',monospace; padding:8px 10px; border-bottom:1px solid #2a3d37; }
table.report td { padding:10px; border-bottom:1px solid #1d2c28; color:#dbe6e0; font-family:'IBM Plex Mono',monospace; font-size:12px; }
[data-testid="stMetric"] { background:#14201d; border:1px solid #293a35; border-radius:5px; padding:13px 15px; }
[data-testid="stMetricLabel"] { color:#a8b8b1; }
[data-testid="stMetricValue"] { color:#e8eee9; font-family:'IBM Plex Mono',monospace; }
div[data-baseweb="tab-list"] { gap:8px; }
button[data-baseweb="tab"] { color:#aab9b3; }
@media (max-width:700px) { .block-container { padding:1rem 1rem 2rem; } h1 { font-size:2.1rem !important; } }
</style>
""", unsafe_allow_html=True)


def pill(label, color):
    return f'<span class="pill"><i style="background:{color}"></i>{label}</span>'


def style_fig(fig, height, **kw):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c5d0ca", family="IBM Plex Mono", size=11),
        legend=dict(orientation="h", y=-0.28), hoverlabel=dict(bgcolor="#17231f", bordercolor="#354942"), **kw)
    fig.update_xaxes(gridcolor="#263632", zeroline=False)
    fig.update_yaxes(gridcolor="#263632", zeroline=False)
    return fig


# --------------------------------------------------------------------------------------
# Animated reactor schematic (inline SVG in an iframe)
# --------------------------------------------------------------------------------------
REACTOR_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');
html,body{margin:0;background:transparent;overflow:hidden}
svg{width:100%;height:auto;display:block}
.mono{font-family:'IBM Plex Mono',monospace}
.fuel{fill:__CORE__;fill-opacity:.35;animation:fuel 3.2s ease-in-out infinite}
@keyframes fuel{0%,100%{fill-opacity:.28}50%{fill-opacity:.95}}
.rod{fill:#1a2924;stroke:#d1a766;stroke-width:1.5}
.ripple{fill:none;stroke:__CORE__;stroke-width:2;vector-effect:non-scaling-stroke;transform-box:fill-box;
        transform-origin:center;opacity:0;animation:ripple __RIPPLE__s ease-out infinite}
@keyframes ripple{0%{transform:scale(1);opacity:.55}100%{transform:scale(2.35);opacity:0}}
.pulse{fill:none;stroke-width:2;vector-effect:non-scaling-stroke;transform-box:fill-box;transform-origin:center;
       animation:pulse var(--d) ease-out infinite}
@keyframes pulse{0%{transform:scale(1);opacity:.8}100%{transform:scale(2.1);opacity:0}}
.flow{animation:flow 1.6s linear infinite}
@keyframes flow{to{stroke-dashoffset:-20}}
.spin{transform-origin:__CX__px __CY__px;animation:spin 60s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.blink{animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:.25}}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
"""


def reactor_html(sensors, core_color, leak, median_level, selected):
    cx, cy, R = 380, 320, 265
    ripple_dur = 2.4 if leak else 5.0
    css = (REACTOR_CSS.replace("__CORE__", core_color).replace("__RIPPLE__", str(ripple_dur))
           .replace("__CX__", str(cx)).replace("__CY__", str(cy)))
    p = [f'<svg viewBox="0 0 760 640" xmlns="http://www.w3.org/2000/svg">',
         '<defs><radialGradient id="bg" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#1b2f29"/>'
         '<stop offset="1" stop-color="#0b1211" stop-opacity="0"/></radialGradient>'
         f'<radialGradient id="glow" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="{core_color}" stop-opacity=".45"/>'
         f'<stop offset="1" stop-color="{core_color}" stop-opacity="0"/></radialGradient>'
         '<filter id="soft"><feGaussianBlur stdDeviation="3"/></filter></defs>',
         f'<circle cx="{cx}" cy="{cy}" r="310" fill="url(#bg)"/>',
         f'<circle cx="{cx}" cy="{cy}" r="140" fill="url(#glow)"/>']
    for k in range(3):
        p.append(f'<circle class="ripple" cx="{cx}" cy="{cy}" r="105" style="animation-delay:{k * ripple_dur / 3:.2f}s"/>')
    p.append(f'<circle cx="{cx}" cy="{cy}" r="241" fill="none" stroke="#2a3d37" stroke-width="1" stroke-dasharray="2 7"/>')
    p.append(f'<g class="spin"><circle cx="{cx}" cy="{cy}" r="228" fill="none" stroke="#3b5a50" stroke-width="1.5" stroke-dasharray="14 10"/></g>')
    p.append(f'<circle cx="{cx}" cy="{cy}" r="212" fill="#0e1816" stroke="#41594f" stroke-width="7"/>')
    p.append(f'<circle cx="{cx}" cy="{cy}" r="188" fill="none" stroke="{MINT}" stroke-opacity=".28" stroke-width="2" stroke-dasharray="3 9" class="flow"/>')
    p.append(f'<circle cx="{cx}" cy="{cy}" r="122" fill="#14211e" stroke="#527064" stroke-width="2"/>')

    # hexagonal fuel lattice, every third cell is a control rod
    s = 14.0
    idx = 0
    for q in range(-3, 4):
        for r in range(-3, 4):
            if abs(q + r) > 3:
                continue
            hx = cx + s * np.sqrt(3) * (q + r / 2)
            hy = cy + s * 1.5 * r
            pts = " ".join(f"{hx + (s - 1.5) * np.cos(np.deg2rad(60 * k + 30)):.1f},"
                           f"{hy + (s - 1.5) * np.sin(np.deg2rad(60 * k + 30)):.1f}" for k in range(6))
            if (q - r) % 3 == 0 and (q, r) != (0, 0):
                p.append(f'<polygon class="rod" points="{pts}"/>')
            else:
                delay = (np.hypot(q, r) * 0.35 + idx * 0.03) % 3.2
                p.append(f'<polygon class="fuel" points="{pts}" style="animation-delay:-{delay:.2f}s"/>')
            idx += 1
    p.append(f'<text x="{cx}" y="{cy + 160}" text-anchor="middle" class="mono" font-size="11" fill="#91a19c" letter-spacing="1">'
             f'NETWORK MEDIAN {median_level:.1f}</text>')
    p.append(f'<text x="{cx}" y="{cy - 150}" text-anchor="middle" font-family="Barlow Condensed" font-weight="700" '
             f'font-size="18" fill="#dce8df" letter-spacing="1">REACTOR VESSEL</text>')

    for i, s_ in enumerate(sensors):
        a = np.deg2rad(SENSOR_ANGLES[i])
        ux, uy = np.cos(a), -np.sin(a)
        sx, sy = cx + R * ux, cy + R * uy
        c = s_["color"]
        dash = "" if s_["dropout"] else f'stroke-dasharray="4 6" class="flow"'
        p.append(f'<line x1="{cx + 125 * ux:.1f}" y1="{cy + 125 * uy:.1f}" x2="{sx - 20 * ux:.1f}" y2="{sy - 20 * uy:.1f}" '
                 f'stroke="{c}" stroke-opacity=".55" stroke-width="1.5" {dash}/>')
        dur = "0.9s" if s_["faulty"] else "2.6s"
        p.append(f'<circle class="pulse" cx="{sx:.1f}" cy="{sy:.1f}" r="17" stroke="{c}" style="--d:{dur}"/>')
        p.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="17" fill="#0b1211" stroke="{c}" stroke-width="3"/>')
        if s_["dropout"]:
            p.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="7" fill="none" stroke="{c}" stroke-width="2" stroke-dasharray="3 3"/>')
        else:
            blink = ' class="blink"' if s_["faulty"] else ""
            p.append(f'<circle{blink} cx="{sx:.1f}" cy="{sy:.1f}" r="8" fill="{c}"/>')
        if i == selected:
            p.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="27" fill="none" stroke="#e8eee9" stroke-width="1.2" stroke-dasharray="3 4"/>')
        ax, ay = sx + 42 * ux, sy + 42 * uy
        anchor = "middle" if abs(ux) < 0.3 else "start" if ux > 0 else "end"
        top = ay - 44 if uy < -0.5 else ay + 2 if uy > 0.5 else ay - 18
        p.append(f'<text x="{ax:.1f}" y="{top + 12:.1f}" text-anchor="{anchor}" font-family="Barlow Condensed" '
                 f'font-weight="700" font-size="22" fill="#eef5ef">{s_["name"]}</text>')
        p.append(f'<text x="{ax:.1f}" y="{top + 29:.1f}" text-anchor="{anchor}" class="mono" font-size="14" '
                 f'font-weight="600" fill="#dce8df">{s_["value"]}</text>')
        p.append(f'<text x="{ax:.1f}" y="{top + 44:.1f}" text-anchor="{anchor}" class="mono" font-size="10" '
                 f'fill="{c}" letter-spacing=".8">{s_["tag"]}</text>')
    p.append("</svg>")
    return f"<style>{css}</style>" + "".join(p)


# --------------------------------------------------------------------------------------
# Sidebar: anomaly controls
# --------------------------------------------------------------------------------------
DEFAULT_FAULTS = ["Healthy", "Healthy", "Drift", "Healthy", "Healthy"]
for i in range(5):
    st.session_state.setdefault(f"fault_{i}", DEFAULT_FAULTS[i])
st.session_state.setdefault("core_event", "None")


def reset_all():
    for i in range(5):
        st.session_state[f"fault_{i}"] = "Healthy"
    st.session_state["core_event"] = "None"


st.sidebar.markdown('<div class="eyebrow">Anomaly lab</div>', unsafe_allow_html=True)
st.sidebar.title("NuclearRad AI")
st.sidebar.markdown('<div class="side-head">Core-wide event</div>', unsafe_allow_html=True)
core_event = st.sidebar.selectbox("Event affecting all sensors", list(CORE_EVENTS), key="core_event",
                                  help="Real changes in the core that every sensor sees.")
st.sidebar.markdown('<div class="side-head">Sensor anomalies</div>', unsafe_allow_html=True)
sensor_faults = [
    st.sidebar.selectbox(NAMES[i], list(SENSOR_ANOMALIES), key=f"fault_{i}",
                         help="Fault injected into this sensor only.")
    for i in range(5)
]
st.sidebar.markdown('<div class="side-head">Timing and strength</div>', unsafe_allow_html=True)
onset = st.sidebar.slider("Anomaly starts at sample", 60, 200, 100)
severity = st.sidebar.slider("Severity", 1, 5, 3)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=7)
noise_pct = st.sidebar.slider("Sensor noise (%)", 0.5, 3.0, 1.2, 0.1)
show_truth = st.sidebar.checkbox("Show simulated true value", True)
st.sidebar.button("Reset all to healthy", on_click=reset_all, width="stretch")
st.sidebar.markdown("---")
st.sidebar.caption("Training simulation only. Synthetic normalized readings; not a plant control or safety system.")

x, truth = generate(sensor_faults, core_event, int(onset), severity, int(seed), noise_pct / 100)
res = assess(x)

# --------------------------------------------------------------------------------------
# Header, controls, status
# --------------------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Reactor instrumentation · anomaly simulation</div>', unsafe_allow_html=True)
st.title("Anomaly lab")
st.markdown('<div class="subtle">Pick an anomaly for each of the five sensors Rad1 to Rad5, or for the whole core, '
            'and watch how NuclearRad AI separates a failing sensor from a real radiation rise.</div>',
            unsafe_allow_html=True)

pills = [pill(f"{NAMES[i]} · {sensor_faults[i]}", FAULT_COLORS[sensor_faults[i]]) for i in range(5)]
pills.append(pill(f"Core · {core_event}", RED if core_event.startswith("Coolant") else AMBER if core_event != "None" else "#4f7d6c"))
st.markdown(f'<div class="pill-row">{"".join(pills)}</div>', unsafe_allow_html=True)

default_t = min(int(onset) + 60, N_SAMPLES - 1)
t_now = st.slider("Simulation time · sample (1 sample = 1 minute)", WARMUP, N_SAMPLES - 1, default_t)
focus = st.selectbox("Inspect sensor", NAMES, index=2)
k = NAMES.index(focus)

kind = res["kind"][t_now]
status_color, status_label = {"escalation": (RED, "ESCALATION"), "attention": (AMBER, "ATTENTION"),
                              "ok": (MINT, "SYSTEM NOMINAL")}[kind]
st.markdown(
    f'<div class="status-strip" style="border-left-color:{status_color}">'
    f'<div class="eyebrow" style="color:{status_color}">{status_label} · {core_event.upper()}</div>'
    f'<div style="font-size:18px;font-weight:700;margin:5px 0">{res["verdict"][t_now]}</div>'
    f'<div class="subtle">{res["reason"][t_now]} &nbsp;|&nbsp; Recommended action: <b>{res["action"][t_now]}</b></div></div>',
    unsafe_allow_html=True)

flagged_now = int(np.count_nonzero(res["fault"][t_now] != "Healthy"))
m1, m2, m3, m4 = st.columns(4)
m1.metric("Simulation sample", f"{t_now:03d}", f"minute {t_now}")
m2.metric("Network median", f"{res['med'][t_now]:.1f}%", "of baseline")
m3.metric("Sensors flagged", f"{flagged_now} / 5", "AI fault rules")
m4.metric("Mean sensor trust", f"{np.mean(res['trust'][t_now]):.0f}%", "behavior confidence")
st.markdown("<br>", unsafe_allow_html=True)


def sensor_state(i):
    fault, trust, val = res["fault"][t_now, i], res["trust"][t_now, i], x[t_now, i]
    elevated = res["leak"][t_now] and fault == "Healthy"
    color = (RED if fault != "Healthy" else AMBER if elevated or trust < UNCERTAIN_TRUST else MINT)
    tag = fault.upper() if fault != "Healthy" else "ELEVATED" if elevated else "NOMINAL"
    return fault, trust, val, color, tag


overview_tab, charts_tab, report_tab = st.tabs(["REACTOR VIEW", "SENSOR CHARTS", "DETECTION REPORT"])

# --------------------------------------------------------------------------------------
# Tab 1: reactor view
# --------------------------------------------------------------------------------------
with overview_tab:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">Reactor core · five-sensor ring</div>', unsafe_allow_html=True)
        st.caption("Animated schematic. Rings speed up and turn red on faulty sensors; the whole core glows red on a leak.")
        nodes = []
        for i in range(5):
            fault, trust, val, color, tag = sensor_state(i)
            nodes.append(dict(name=NAMES[i], color=color, tag=tag, faulty=fault != "Healthy",
                              dropout=bool(np.isnan(val)), value="NO SIGNAL" if np.isnan(val) else f"{val:.1f}"))
        core_color = RED if kind == "escalation" else AMBER if kind == "attention" else MINT
        components.html(reactor_html(nodes, core_color, bool(res["leak"][t_now]), res["med"][t_now], k), height=640)
    with right:
        st.markdown('<div class="section-label">Selected channel</div>', unsafe_allow_html=True)
        fault, trust, val, color, tag = sensor_state(k)
        baseline = res["base"][k]
        delta = (val / baseline - 1) * 100 if np.isfinite(val) else np.nan
        st.metric(focus, "Missing" if np.isnan(val) else f"{val:.2f} units",
                  "no signal" if np.isnan(delta) else f"{delta:+.1f}% vs baseline")
        st.metric("AI assessment", fault, f"Trust {trust:.0f}%")
        injected = sensor_faults[k]
        st.markdown(
            f'<div class="cat-card"><div class="eyebrow">Simulated on {focus}</div>'
            f'<b style="color:{FAULT_COLORS[injected]}">{injected}</b>'
            f'<p>{SENSOR_ANOMALIES[injected]["what"]}</p>'
            f'<p class="sig">What the AI looks for: {SENSOR_ANOMALIES[injected]["sees"]}</p></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="section-label">All channels</div>', unsafe_allow_html=True)
        cols = st.columns(5)
        for i, col in enumerate(cols):
            fault, trust, val, color, tag = sensor_state(i)
            col.markdown(
                f'<div class="sensor-card" style="border-top:2px solid {color}">'
                f'<div class="sensor-label">{NAMES[i]}</div>'
                f'<div class="sensor-value">{"--" if np.isnan(val) else f"{val:.0f}"}</div>'
                f'<div class="sensor-foot" style="color:{color}">{tag}<br>{trust:.0f}%</div></div>',
                unsafe_allow_html=True)
        st.caption("Trust measures sensor behavior, not radiation level.")

# --------------------------------------------------------------------------------------
# Tab 2: one chart per sensor
# --------------------------------------------------------------------------------------
with charts_tab:
    st.markdown('<div class="section-label">Individual sensor readings</div>', unsafe_allow_html=True)
    st.caption("Raw reading, AI-corrected estimate, and flagged samples for each sensor up to the selected time.")
    tt = np.arange(t_now + 1)
    grid = [st.columns(2, gap="large") for _ in range(3)]
    for i in range(5):
        col = grid[i // 2][i % 2]
        fault_now, trust_now, _, color, tag = sensor_state(i)
        with col:
            st.markdown(f'<div class="pill-row" style="margin:6px 0 0"><span style="font:700 22px \'Barlow Condensed\'">'
                        f'{NAMES[i]}</span>{pill(sensor_faults[i] + " injected", FAULT_COLORS[sensor_faults[i]])}'
                        f'{pill("AI: " + fault_now + f" · trust {trust_now:.0f}%", color)}</div>',
                        unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tt, y=x[:t_now + 1, i], name="Raw reading", mode="lines",
                                     line=dict(color=RED, width=2)))
            fig.add_trace(go.Scatter(x=tt, y=res["corrected"][:t_now + 1, i], name="AI-corrected estimate",
                                     mode="lines", line=dict(color=MINT, width=2, dash="dash")))
            if show_truth:
                fig.add_trace(go.Scatter(x=tt, y=truth[:t_now + 1, i], name="True value", mode="lines",
                                         line=dict(color="#d5c08e", width=1.5, dash="dot")))
            flags = np.flatnonzero(res["fault"][:t_now + 1, i] != "Healthy")
            flags = flags[~np.isnan(x[flags, i])]
            if flags.size:
                fig.add_trace(go.Scatter(x=flags, y=x[flags, i], mode="markers", name="Flagged by AI",
                                         text=res["fault"][flags, i], marker=dict(color="#fff", size=7, symbol="x"),
                                         hovertemplate="Sample %{x}<br>%{text}<br>Reading %{y:.2f}<extra></extra>"))
            fig.add_hline(y=ALARM_LEVEL * res["base"][i], line_dash="dash", line_color=AMBER,
                          annotation_text="130% alarm", annotation_position="top left")
            fig.add_vline(x=WARMUP, line_dash="dot", line_color="#6b7f77")
            if sensor_faults[i] != "Healthy" or core_event != "None":
                fig.add_vrect(x0=onset, x1=N_SAMPLES, fillcolor=RED, opacity=0.06, line_width=0)
            style_fig(fig, 310, xaxis_title="Sample", yaxis_title="Reading", xaxis_range=[0, N_SAMPLES - 1])
            st.plotly_chart(fig, width="stretch", key=f"sensor_chart_{i}")

# --------------------------------------------------------------------------------------
# Tab 3: detection report
# --------------------------------------------------------------------------------------
with report_tab:
    st.markdown('<div class="section-label">Fault timeline · all sensors</div>', unsafe_allow_html=True)
    st.caption("Each cell is one minute. Dark green means healthy; colors mark what the AI diagnosed.")
    n_codes = len(FAULT_ORDER)
    z = np.array([[FAULT_ORDER.index(f) for f in res["fault"][:t_now + 1, i]] for i in range(5)])
    colorscale = []
    for j, name in enumerate(FAULT_ORDER):
        colorscale += [[j / n_codes, FAULT_COLORS[name]], [(j + 1) / n_codes, FAULT_COLORS[name]]]
    heat = go.Figure(go.Heatmap(
        z=z, x=np.arange(t_now + 1), y=NAMES, zmin=-0.5, zmax=n_codes - 0.5, colorscale=colorscale,
        showscale=False, text=res["fault"][:t_now + 1].T, xgap=0, ygap=3,
        hovertemplate="%{y} · sample %{x}<br>%{text}<extra></extra>"))
    heat.update_yaxes(autorange="reversed")
    heat.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c5d0ca", family="IBM Plex Mono", size=11),
                       xaxis=dict(title="Sample", range=[0, N_SAMPLES - 1], gridcolor="#263632"),
                       hoverlabel=dict(bgcolor="#17231f", bordercolor="#354942"))
    st.plotly_chart(heat, width="stretch", key="timeline")
    st.markdown('<div class="pill-row">' + "".join(pill(n, c) for n, c in FAULT_COLORS.items()) + "</div>",
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">Sensor confidence over time</div>', unsafe_allow_html=True)
    trust_fig = go.Figure()
    for i in range(5):
        trust_fig.add_trace(go.Scatter(x=tt, y=res["trust"][:t_now + 1, i], name=NAMES[i], mode="lines",
                                       line=dict(width=3 if i == k else 1.4)))
    style_fig(trust_fig, 260, xaxis_title="Sample", yaxis_title="Trust (%)", yaxis_range=[0, 100],
              xaxis_range=[0, N_SAMPLES - 1])
    st.plotly_chart(trust_fig, width="stretch", key="trust")

    st.markdown('<div class="section-label">Threshold alarm vs NuclearRad AI</div>', unsafe_allow_html=True)
    rows = []
    for i in range(5):
        alarm_hits = np.flatnonzero(x[WARMUP:, i] > ALARM_LEVEL * res["base"][i])
        alarm = None if alarm_hits.size == 0 else int(alarm_hits[0] + WARMUP)
        ai_hits = np.flatnonzero(res["fault"][WARMUP:, i] != "Healthy")
        ai = None if ai_hits.size == 0 else int(ai_hits[0] + WARMUP)
        if alarm is not None and ai is not None:
            lead = f"AI {alarm - ai:+d} min" if alarm != ai else "same time"
        else:
            lead = "AI only" if ai is not None else "threshold only" if alarm is not None else "-"
        fc = FAULT_COLORS[sensor_faults[i]]
        rows.append(
            f'<tr><td><b>{NAMES[i]}</b></td><td style="color:{fc}">{sensor_faults[i]}</td>'
            f'<td>{"never" if alarm is None else f"sample {alarm}"}</td>'
            f'<td>{"no fault" if ai is None else f"sample {ai}"}</td><td>{lead}</td>'
            f'<td style="color:{sensor_state(i)[3]}">{sensor_state(i)[4]}</td></tr>')
    st.markdown('<table class="report"><tr><th>SENSOR</th><th>INJECTED</th><th>130% ALARM FIRES</th>'
                '<th>AI FIRST FLAGS</th><th>LEAD</th><th>STATUS NOW</th></tr>' + "".join(rows) + "</table>",
                unsafe_allow_html=True)
    leak_hits = np.flatnonzero(res["leak"])
    if leak_hits.size:
        st.info(f"Core-wide leak escalation first raised at sample {int(leak_hits[0])}, "
                f"{int(leak_hits[0]) - int(onset):+d} min from the event start.")

# --------------------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------------------
with st.expander("Anomaly catalog · what each dropdown option simulates"):
    items = [(n, d["color"], d["what"], d["sees"]) for n, d in SENSOR_ANOMALIES.items() if n != "Healthy"]
    items += [(n, RED if n.startswith("Coolant") else AMBER, d["what"], d["sees"])
              for n, d in CORE_EVENTS.items() if n != "None"]
    for row in range(0, len(items), 3):
        cols = st.columns(3)
        for col, (name, color, what, sees) in zip(cols, items[row:row + 3]):
            col.markdown(f'<div class="cat-card"><b style="color:{color}">{name}</b><p>{what}</p>'
                         f'<p class="sig">What the AI looks for: {sees}</p></div>', unsafe_allow_html=True)
            col.markdown("<br>", unsafe_allow_html=True)

with st.expander("Assessment model"):
    st.markdown("""
- Each sensor is calibrated against its own warm-up readings (first 40 samples), then compared with the peer median.
- A shared rise across all sensors is escalated as a possible real leak instead of being blamed on a sensor.
- Rules detect spikes, drift, offsets, stuck readings, excess noise, and dropouts.
- Trust is a 0-100 behavior-confidence score, not a radiation measurement.
- Flagged readings are replaced by a peer-derived estimate for demonstration only.
""")
