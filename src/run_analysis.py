"""
End-to-end LTV analysis for a travel subscription programme (synthetic data).

Steps
  1. Generate synthetic member data
  2. Observed retention by channel (renewal by renewal)
  3. Fit an sBG model per channel and compute 5-year LTV, LTV:CAC and payback
  4. Back-test: fit on data available at 31 Mar 2024, predict FY22 cohort retention
     in years 3 and 4, and compare with a constant-churn model
  5. Uncertainty for a young channel (conversational AI) via bootstrap
  6. Value concentration among currently active members

Run from the project root:  python src/run_analysis.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln

import sbg
from simulate import CHANNELS, OBSERVATION_END, simulate

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, OUT = ROOT / "data", ROOT / "figures", ROOT / "results"
for d in (DATA, FIG, OUT):
    d.mkdir(exist_ok=True)

YEARS, DISCOUNT = 5, 0.10
CUTOFF = pd.Timestamp("2024-03-31")
MATURE = [c for c in CHANNELS if c != "Conversational AI"]

# Chart styling (validated categorical slots 1-2 + neutral ink)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


# ---------------------------------------------------------------- helpers
def observed_retention(df, max_t=4):
    """Share of an acquired cohort still active after t renewals.
    Uses only members who have reached renewal t (handles right-censoring)."""
    surv, s = [1.0], 1.0
    for t in range(1, max_t + 1):
        at_risk = df[(df["renewal_opportunities"] >= t) & (df["renewals"] >= t - 1)]
        if len(at_risk) < 50:
            break
        s *= (at_risk["renewals"] >= t).mean()
        surv.append(s)
    return np.array(surv)


def constant_churn_fit(df):
    """The common shortcut: take the first-renewal churn rate and assume it every year."""
    reached = df[df["renewal_opportunities"] >= 1]
    return (reached["renewals"] == 0).mean()


def as_of(df, cutoff):
    """What the data looked like at `cutoff` (later events hidden)."""
    d = df[df["join_date"] <= cutoff].copy()
    jd = d["join_date"].dt
    passed = (jd.month < cutoff.month) | ((jd.month == cutoff.month) & (jd.day <= cutoff.day))
    opp = cutoff.year - jd.year - (~passed).astype(int)
    d["churned"] = d["churned"] & (d["renewals"] + 1 <= opp)
    d["renewals"] = np.minimum(d["renewals"], opp)
    d["renewal_opportunities"] = opp
    return d


def eur(x):
    return f"€{x:,.0f}"


# ---------------------------------------------------------------- 1. data
members = simulate()
members.to_csv(DATA / "members.csv", index=False)
print(f"{len(members):,} synthetic members, observed to {OBSERVATION_END.date()}")

# ---------------------------------------------------------------- 2-3. retention, sBG, LTV
rows, fits = [], {}
for ch, cfg in CHANNELS.items():
    d = members[members["channel"] == ch]
    a, b = sbg.fit(d["renewals"], d["churned"])
    fits[ch] = (a, b)
    margin = d["margin_per_active_year"].mean()
    s_obs = observed_retention(d)
    ltv5 = sbg.ltv(a, b, margin, YEARS, DISCOUNT)
    c = constant_churn_fit(d)
    ltv_const = sum(margin * (1 - c) ** t / (1 + DISCOUNT) ** t for t in range(YEARS))
    rows.append(dict(
        channel=ch,
        members=len(d),
        first_renewal_rate=s_obs[1] if len(s_obs) > 1 else np.nan,
        retention_y4=s_obs[4] if len(s_obs) > 4 else np.nan,
        sbg_a=a, sbg_b=b,
        avg_margin_per_year=margin,
        ltv_5y=ltv5,
        ltv_5y_constant_churn=ltv_const,
        cac=cfg["cac"],
        ltv_cac=ltv5 / cfg["cac"],
        payback_months=cfg["cac"] / (margin / 12),
    ))
summary = pd.DataFrame(rows)

# ---------------------------------------------------------------- 4. back-test
bt_rows = []
calib = as_of(members, CUTOFF)
for ch in MATURE:
    d_cal = calib[calib["channel"] == ch]
    a, b = sbg.fit(d_cal["renewals"], d_cal["churned"])
    c = constant_churn_fit(d_cal)
    fy22 = members[(members["channel"] == ch) & (members["cohort_fy"] == "FY22")]
    actual = observed_retention(fy22)
    for t in (3, 4):
        bt_rows.append(dict(channel=ch, renewal=t, actual=actual[t],
                            sbg=float(sbg.survival(a, b, t)), constant=(1 - c) ** t))
backtest = pd.DataFrame(bt_rows)
backtest["sbg_error_pp"] = (backtest["sbg"] - backtest["actual"]) * 100
backtest["constant_error_pp"] = (backtest["constant"] - backtest["actual"]) * 100
backtest.round(4).to_csv(OUT / "backtest.csv", index=False)

# ---------------------------------------------------------------- 5. young-channel uncertainty
# Conversational AI members have reached one renewal at most, so the data says how many
# stay after year 1 but nothing about later years. Two sources of uncertainty:
#   (a) sampling noise in the year-1 rate and margin -> bootstrap
#   (b) the unknown shape of later retention -> borrow each mature channel's curve shape
ai = members[members["channel"] == "Conversational AI"].reset_index(drop=True)
ai_margin = ai["margin_per_active_year"].mean()
ai_r1 = observed_retention(ai)[1]

scenarios = {}
for ch in MATURE:
    a, b = fits[ch]
    shape = sbg.survival(a, b, np.arange(YEARS)) / sbg.survival(a, b, 1)
    surv = np.concatenate([[1.0], ai_r1 * shape[1:]])
    scenarios[ch] = float(np.sum(ai_margin * surv / (1 + DISCOUNT) ** np.arange(YEARS)))
ai_low, ai_high = min(scenarios.values()), max(scenarios.values())

rng = np.random.default_rng(7)
boot = []
for _ in range(500):
    s_ = ai.sample(len(ai), replace=True, random_state=rng.integers(1e9))
    r1 = observed_retention(s_)[1]
    boot.append(r1)
ai_r1_ci = np.percentile(boot, [5, 95])

# Replace the unreliable AI point estimate with the scenario midpoint, flagged in the table
mid = (ai_low + ai_high) / 2
m_ai = summary["channel"] == "Conversational AI"
summary.loc[m_ai, "ltv_5y"] = mid
summary.loc[m_ai, "ltv_cac"] = mid / CHANNELS["Conversational AI"]["cac"]
summary.round(3).to_csv(OUT / "ltv_by_channel.csv", index=False)

# ---------------------------------------------------------------- 6. value concentration (active members)
active = members[~members["churned"]].copy()
fut = np.zeros(len(active))
for ch, (a, b) in fits.items():
    m = (active["channel"] == ch).to_numpy()
    n = active.loc[m, "renewals"].to_numpy()
    # members who have already renewed n times are more loyal: Beta(a, b + n)
    for k in range(1, 4):
        s_cond = np.exp(betaln(a, b + n + k) - betaln(a, b + n))
        fut[m] += active.loc[m, "margin_per_active_year"].to_numpy() * s_cond / (1 + DISCOUNT) ** k
active["expected_value_next_3y"] = fut
active = active.sort_values("expected_value_next_3y", ascending=False)
top10 = active["expected_value_next_3y"].head(len(active) // 10).sum() / active["expected_value_next_3y"].sum()
by_tenure = active.groupby("renewals")["expected_value_next_3y"].mean()

# ---------------------------------------------------------------- figures
# Fig 1: retention curves, small multiples
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
t = np.arange(5)
for ax, ch in zip(axes, MATURE):
    d = members[members["channel"] == ch]
    a, b = fits[ch]
    c = constant_churn_fit(d)
    obs = observed_retention(d)
    ax.plot(t, sbg.survival(a, b, t) * 100, color=BLUE, lw=2, label="sBG model")
    ax.plot(t, (1 - c) ** t * 100, color=ORANGE, lw=2, ls="--", label="Constant churn (year-1 rate every year)")
    ax.scatter(np.arange(len(obs)), obs * 100, s=36, color=INK, zorder=3,
               edgecolor=SURFACE, linewidth=1.5, label="Observed")
    ax.set_title(ch, fontsize=10.5, color=INK, loc="left")
    ax.set_xticks(t)
    ax.set_xlabel("Renewals")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_ylim(0, 105)
axes[0].set_ylabel("% of acquired members still active")
axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9.5)
fig.suptitle("Retention by acquisition channel", x=0.01, ha="left", fontsize=12.5, color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(FIG / "01_retention_by_channel.png", dpi=160)
plt.close(fig)

# Fig 2: LTV vs CAC
s2 = summary.sort_values("ltv_cac")
y = np.arange(len(s2))
fig, ax = plt.subplots(figsize=(8.5, 4))
ax.barh(y + 0.19, s2["ltv_5y"], height=0.36, color=BLUE, label="5-year LTV (discounted margin)")
ax.barh(y - 0.19, s2["cac"], height=0.36, color=ORANGE, label="Acquisition cost (CAC)")
for yi, (_, r) in zip(y, s2.iterrows()):
    ax.text(r["ltv_5y"] + 3, yi + 0.19, f"{eur(r['ltv_5y'])}  ·  LTV:CAC {r['ltv_cac']:.1f}",
            va="center", fontsize=9, color=INK)
    ax.text(r["cac"] + 3, yi - 0.19, eur(r["cac"]), va="center", fontsize=9, color=INK2)
ax.set_yticks(y, s2["channel"])
ax.set_xlim(0, s2["ltv_5y"].max() * 1.45)
ax.xaxis.set_major_formatter(lambda v, _: f"€{v:.0f}")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("Value per acquired member vs. cost to acquire", loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "02_ltv_vs_cac.png", dpi=160)
plt.close(fig)

# Fig 3: back-test errors
bt4 = backtest[backtest["renewal"] == 4].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 4))
y = np.arange(len(bt4))
for yi, r in bt4.iterrows():
    ax.plot([r["actual"] * 100, r["constant"] * 100], [yi, yi], color=GRID, lw=2, zorder=1)
ax.scatter(bt4["actual"] * 100, y, s=190, facecolor="none", edgecolor=INK, lw=1.8, zorder=4, label="Actual")
ax.scatter(bt4["sbg"] * 100, y, s=70, color=BLUE, zorder=3, label="sBG forecast", marker="D", edgecolor=SURFACE, lw=1.5)
ax.scatter(bt4["constant"] * 100, y, s=70, color=ORANGE, zorder=3, label="Constant-churn forecast (year-1 rate)", marker="s", edgecolor=SURFACE, lw=1.5)
ax.set_yticks(y, bt4["channel"])
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.set_xlabel("FY22 cohort still active after 4 renewals")
ax.set_xlim(0, 45)
ax.set_ylim(-0.6, len(bt4) - 0.4)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False, fontsize=9)
ax.set_title("Back-test: forecasts made with data to Mar 2024 vs. what happened",
             loc="left", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "03_backtest.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- markdown results
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)

t1 = summary.sort_values("ltv_cac", ascending=False).assign(
    Members=lambda x: x["members"].map("{:,}".format),
    **{"1st renewal": lambda x: x["first_renewal_rate"].map(lambda v: f"{v:.0%}"),
       "Active after 4 renewals": lambda x: x["retention_y4"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.0%}"),
       "Margin / year": lambda x: x["avg_margin_per_year"].map(eur),
       "5-yr LTV": lambda x: [eur(v) + ("*" if c == "Conversational AI" else "") for v, c in zip(x["ltv_5y"], x["channel"])],
       "CAC": lambda x: x["cac"].map(eur),
       "LTV:CAC": lambda x: x["ltv_cac"].map("{:.1f}".format),
       "Payback": lambda x: x["payback_months"].map(lambda v: f"{v:.0f} mo")},
)[["channel", "Members", "1st renewal", "Active after 4 renewals", "Margin / year", "5-yr LTV", "CAC", "LTV:CAC", "Payback"]].rename(columns={"channel": "Channel"})

t2 = backtest.assign(
    Renewal=backtest["renewal"],
    Actual=backtest["actual"].map(lambda v: f"{v:.1%}"),
    sBG=backtest["sbg"].map(lambda v: f"{v:.1%}"),
    **{"Constant churn": backtest["constant"].map(lambda v: f"{v:.1%}"),
       "sBG error": backtest["sbg_error_pp"].map(lambda v: f"{v:+.1f} pp"),
       "Constant error": backtest["constant_error_pp"].map(lambda v: f"{v:+.1f} pp")},
)[["channel", "Renewal", "Actual", "sBG", "Constant churn", "sBG error", "Constant error"]].rename(columns={"channel": "Channel"})

ltv_gap = summary[summary["channel"].isin(MATURE)]
under = (1 - ltv_gap["ltv_5y_constant_churn"] / ltv_gap["ltv_5y"]).mean()
ai_rows = summary["channel"] == "Conversational AI"
ai_row = summary.set_index("channel").loc["Conversational AI"]
scen_txt = ", ".join(f"{k} {eur(v)}" for k, v in sorted(scenarios.items(), key=lambda kv: kv[1]))

results_md = f"""### Data

{len(members):,} synthetic members across {len(CHANNELS)} channels, cohorts April 2021 to March 2025, observed to {OBSERVATION_END:%d %b %Y}.

### LTV by channel (5-year horizon, 10% annual discount rate)

{md_table(t1)}

\\* Midpoint of a scenario range; see "Uncertainty for a young channel".

### Back-test (model fitted on data available at 31 Mar 2024; FY22 cohort)

{md_table(t2)}

Mean absolute error: **sBG {backtest['sbg_error_pp'].abs().mean():.1f} pp** vs. **constant churn {backtest['constant_error_pp'].abs().mean():.1f} pp**. Applying first-year churn to every year understates 5-year LTV by **{under:.0%}** on average across mature channels (the gap grows with longer horizons).

### Uncertainty for a young channel

Conversational AI members ({len(ai):,}) have reached one renewal at most. Their first-renewal rate is **{ai_r1:.0%}** (bootstrapped 90% interval {ai_r1_ci[0]:.0%} to {ai_r1_ci[1]:.0%}), but the data says nothing yet about later years, so the sBG fit for this channel is not reliable on its own. Borrowing the later-year retention shape of each mature channel gives a 5-year LTV between **{eur(ai_low)} and {eur(ai_high)}** ({scen_txt}). Even the low end gives LTV:CAC of **{ai_low / CHANNELS['Conversational AI']['cac']:.1f}**.

### Value concentration among active members

The top 10% of currently active members hold **{top10:.0%}** of expected margin over the next three years. Expected 3-year value rises with tenure: {", ".join(f"{eur(v)} after {int(k)} renewal{'s' if k != 1 else ''}" for k, v in by_tenure.items())}.
"""
(OUT / "results.md").write_text(results_md)
print(results_md)
