"""Regenerate the paper's memorization figures from analysis/out/{recompute,fig5_data}.json.

  fig5_cold_recall_and_recovery.pdf   main-text Fig 5 (a) Real composition by tier x model, (b) Parallel by tier, (c) full-law per model
  fig11_recall_spectrum.pdf           appendix: # of panel models that cold-recall each task (moat ... canon)
  fig12_ladder_bytype.pdf             appendix: cold recall by task structure (single/multi-group) per model
  fig13_domain.pdf                    appendix: cold recall by domain (mean over panel, dot = frontier)
  fig2_feynman_vs_scilaws.pdf         §2.2 side figure: OpenAI ladder on AI-Feynman vs SciLaws-Real (models on BOTH corpora)
Model order everywhere = weak -> strong by mean S_N in the main experiment (as in the paper).
House palette: BLUE #2A6F97 (SciLaws / moat), ORANGE #E8743B (Feynman / canon / above-baseline).
"""
from __future__ import annotations
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "analysis"))
import recompute as RC
OUT = RC.OUT / "figs"; OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((RC.OUT / "recompute.json").read_text()); CUR = D["panels"]["current"]
F5 = json.loads((RC.OUT / "fig5_data.json").read_text())
SHORT = {k: v["short"] for k, v in RC.META.items()}
short = lambda m: SHORT.get(m, m)

BLUE, ORANGE, GREY, GREY_D, NAVY = "#2A6F97", "#E8743B", "#C2C7CC", "#6B7280", "#2A6F97"
def tint(h, f):
    h = h.lstrip("#"); r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return "#{:02X}{:02X}{:02X}".format(*(int(c + (255 - c) * f) for c in (r, g, b)))
ORANGE_LT = tint(ORANGE, .45)
PCT = FuncFormatter(lambda v, _: f"{v:.0f}%")

# ---- shared: scores + order -------------------------------------------------------------
real, struct = defaultdict(dict), defaultdict(dict)
for r in F5["rows"]:
    if r["real"] is not None: real[r["model"]][r["task"]] = r["real"]
    if r["struct"] is not None: struct[r["model"]][r["task"]] = r["struct"]
ORDER = sorted(real, key=lambda m: st.mean(real[m].values()))          # weak -> strong by mean S_N
tasks = sorted({r["task"] for r in F5["rows"]})
MOAT, CANON = set(CUR["moat_members"]), set(CUR["canon_members"])
tier = lambda t: "Moat" if t in MOAT else "Canon" if t in CANON else "Mid"
TIERS = ["Canon", "Mid", "Moat"]; T = 0.02

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight"); fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("  ", name)

def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

# ---- Fig 5 ------------------------------------------------------------------------------
def fig5():
    M = ORDER; LADL = [short(m).replace("-", "-\n", 1) if len(short(m)) > 6 else short(m) for m in M]
    def bands(m, z):
        v = [real[m][t] for t in tasks if tier(t) == z and t in real[m]]; n = len(v) or 1
        return (100*sum(s < 0.5-T for s in v)/n, 100*sum(0.5-T <= s <= 0.5+T for s in v)/n, 100*sum(s > 0.5+T for s in v)/n)
    def pooled(z, thr):
        v = [struct[m][t] for m in M for t in tasks if tier(t) == z and t in struct[m]]
        return 100*sum(s >= thr for s in v)/(len(v) or 1)
    def fullrate(m, moat_only=False):
        v = [struct[m][t] for t in tasks if t in struct[m] and (tier(t) == "Moat" if moat_only else True)]
        return 100*sum(s >= 0.999 for s in v)/(len(v) or 1)
    PUB = [pooled(z, 0.5) for z in TIERS]; FULL = [pooled(z, 0.999) for z in TIERS]
    ALLR = [fullrate(m) for m in M]; MOATR = [fullrate(m, True) for m in M]
    NC = {z: sum(1 for t in tasks if tier(t) == z) for z in TIERS}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 13.5,
                         "axes.labelsize": 13.5, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5, "axes.linewidth": 0.9})
    fig = plt.figure(figsize=(12, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0], width_ratios=[0.9, 1.05], hspace=0.52, wspace=0.30,
                          left=0.06, right=0.985, top=0.905, bottom=0.075)
    ax = fig.add_subplot(gs[0, :]); W, GAP, pos, xs, centers, spans = 0.86, 0.7, 0, [], [], []
    for z in TIERS:
        c0 = pos
        for m in M:
            g, o, b = bands(m, z)
            ax.bar(pos, g, W, color=GREY); ax.bar(pos, o, W, bottom=g, color=ORANGE_LT); ax.bar(pos, b, W, bottom=g+o, color=ORANGE)
            for val, y0, col in [(g, 0, GREY_D), (o, g, GREY_D), (b, g+o, "white")]:
                if val > 6: ax.text(pos, y0 + val/2, f"{val:.0f}", ha="center", va="center", fontsize=8.6, color=col)
            xs.append(pos); pos += 1
        centers.append((c0 + pos - 1)/2); spans.append((c0 - W/2, pos - 1 + W/2)); pos += GAP
    ax.set_xticks(xs); ax.set_xticklabels(LADL * 3, fontsize=9.2); ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("share of tasks"); ax.margins(x=0.01); clean(ax)
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (c, z), (x0, x1) in zip(zip(centers, TIERS), spans):
        ax.plot([x0, x1], [-0.175, -0.175], transform=tr, color=GREY_D, lw=1.1, clip_on=False)
        ax.text(c, -0.202, z, ha="center", va="top", fontsize=13, weight="bold", transform=tr)
        ax.text(c, -0.268, f"n={NC[z]}", ha="center", va="top", fontsize=10.5, color=GREY_D, transform=tr)
    ax.legend(handles=[Patch(color=GREY, label="below baseline"), Patch(color=ORANGE_LT, label="comparable"), Patch(color=ORANGE, label="above baseline")],
              loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=3, frameon=False, handlelength=0.95, columnspacing=1.5, handletextpad=0.4)
    ax.set_title("(a) SciLaws-Real", fontweight="bold", loc="left", pad=9, fontsize=15)
    ax = fig.add_subplot(gs[1, 0])
    for i, (pf, fl) in enumerate(zip(PUB, FULL)):
        ax.plot([i, i], [fl, pf], color=tint(GREY_D, .35), lw=8, solid_capstyle="round", zorder=1)
        ax.text(i + 0.11, (pf + fl)/2, f"{pf-fl:.0f}", va="center", ha="left", fontsize=11, color=GREY_D)
    ax.scatter(range(3), PUB, color=ORANGE, s=115, zorder=3, label="published form"); ax.scatter(range(3), FULL, color=NAVY, s=95, zorder=3, label="full law")
    ax.set_xticks(range(3)); ax.set_xticklabels(TIERS); ax.set_xlim(-0.35, 2.35); ax.set_ylim(0, 100); ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("recovery rate"); ax.set_xlabel("memorization tier"); ax.grid(axis="y", ls=":", alpha=.45); clean(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=1.0, labelspacing=0.3, borderpad=0.2)
    ax.set_title("(b) SciLaws-Parallel · by tier", fontweight="bold", loc="left", pad=9, fontsize=15)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(range(len(M)), ALLR, "-s", color=NAVY, lw=2.6, ms=8, label="all tasks")
    ax.plot(range(len(M)), MOATR, "--D", color=GREY_D, lw=2.2, ms=7, dashes=(4, 2), label="Moat tasks")
    ax.set_xticks(range(len(M))); ax.set_xticklabels(LADL, fontsize=9.2); ax.set_ylim(0, max(25, max(ALLR) + 4)); ax.yaxis.set_major_formatter(PCT)
    ax.set_ylabel("full-law recovery"); ax.margins(x=0.04); ax.grid(axis="y", ls=":", alpha=.45); clean(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=False, handlelength=1.4, labelspacing=0.3, borderpad=0.2)
    ax.set_title("(c) SciLaws-Parallel · full law", fontweight="bold", loc="left", pad=9, fontsize=15)
    save(fig, "fig5_cold_recall_and_recovery")
    return {"published_form_by_tier": dict(zip(TIERS, [round(x/100, 3) for x in PUB])),
            "full_law_by_tier": dict(zip(TIERS, [round(x/100, 3) for x in FULL])),
            "full_law_per_model_pct": {m: round(v, 1) for m, v in zip(M, ALLR)}, "order": M}

# ---- appendix figs ----------------------------------------------------------------------
def appendix():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight", "legend.fontsize": 10})
    order = [m for m in ORDER if m in CUR["models"]] + [m for m in CUR["models"] if m not in ORDER]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(x, [CUR["ladder_by_type"]["II"][m] for m in order], "-^", color=BLUE, lw=2.2, ms=6, label="SciLaws-Real · multi-group")
    ax.plot(x, [CUR["ladder_by_type"]["I"][m] for m in order], "-o", color=tint(BLUE, .45), lw=2.2, ms=6, label="SciLaws-Real · single-group")
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in order], rotation=45, ha="right")
    ax.set_xlabel("Subject model (weak → frontier)"); ax.set_ylabel("Tasks cold-recalled (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.3)
    ax.legend(frameon=False, loc="upper left"); ax.set_title("Cold recall by task structure"); fig.tight_layout(); save(fig, "fig12_ladder_bytype")

    spec = CUR["recall_spectrum"]; n = len(spec) - 1; xs = np.arange(len(spec))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(xs, spec, color=[BLUE] + [tint(BLUE, .6)] * (len(spec) - 2) + [ORANGE], edgecolor="white")
    for xi, v in zip(xs, spec):
        if v > 0: ax.text(xi, v + 0.7, str(v), ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(xs); ax.set_xlabel(f"Number of models (of {n}) that cold-recall the task"); ax.set_ylabel("Number of tasks")
    ax.set_ylim(0, max(spec) + 7); ax.grid(axis="y", alpha=.3)
    ax.legend(handles=[Patch(color=BLUE, label="0 models — discovery moat"), Patch(color=ORANGE, label=f"{n} models — universal canon")], frameon=False, loc="upper center")
    ax.set_title("Distribution of per-task cold recall"); fig.tight_layout(); save(fig, "fig11_recall_spectrum")

    dd = CUR["per_domain"]; y = np.arange(len(dd)); fr = CUR["frontier_model"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4)); ax.grid(axis="x", alpha=.3, zorder=0)
    ax.barh(y, [d["mem_pct"] for d in dd], color=BLUE, edgecolor="white", zorder=2, label=f"mean over {len(CUR['models'])} models")
    ax.scatter([d["front_pct"] for d in dd], y, color=ORANGE, s=48, zorder=3, label=f"frontier ({fr})")
    ax.set_yticks(y); ax.set_yticklabels([f"{d['domain']} ({d['tasks']})" for d in dd], fontsize=9.5)
    ax.set_xlim(0, max(d["front_pct"] for d in dd) + 6); ax.set_xlabel("Tasks cold-recalled (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=9); ax.set_title("Cold recall by domain"); fig.tight_layout(); save(fig, "fig13_domain")

# ---- Fig 2: AI-Feynman vs SciLaws on the OpenAI ladder (models audited on BOTH corpora) ----------
def fig2():
    fe = D["feynman"]; sc = D["scilaws_for_feynman_fig"]
    models = [m for m in RC.OPENAI_LADDER_ORDER if m in fe and fe[m]["complete"] and sc.get(m) is not None] + \
             sorted([m for m in fe if m not in RC.OPENAI_LADDER_ORDER and fe[m]["complete"] and sc.get(m) is not None])
    r = [sc[m] for m in models]; f = [fe[m]["mem_pct"] for m in models]
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11.5, "legend.fontsize": 10.5, "xtick.labelsize": 8.5, "ytick.labelsize": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight"})
    x = np.arange(len(models)); fig, ax = plt.subplots(figsize=(3.05, 2.75))
    ax.fill_between(x, r, f, color=ORANGE, alpha=.10, zorder=0)
    ax.plot(x, f, "-s", color=ORANGE, lw=2.2, ms=5, label="AI-Feynman", zorder=3); ax.plot(x, r, "-o", color=BLUE, lw=2.2, ms=5, label="SciLaws-Real", zorder=3)
    for yv, col in [(r[-1], BLUE), (f[-1], ORANGE)]:
        ax.text(x[-1] + 0.15, yv, f"{yv:.0f}", color=col, fontsize=10, fontweight="bold", va="center")
    ax.set_xticks(x); ax.set_xticklabels([short(m) for m in models], rotation=45, ha="right"); ax.set_ylabel("cold-recalled (%)")
    ax.set_xlim(-0.3, len(models) - 1 + 0.75); ax.set_ylim(0, 100); ax.set_yticks([0, 25, 50, 75, 100]); ax.grid(axis="y", alpha=.3)
    ax.legend(frameon=False, loc="upper left", handlelength=1.3, labelspacing=0.25, borderpad=0.2); fig.tight_layout(pad=0.3)
    save(fig, "fig2_feynman_vs_scilaws")
    return {"models": models, "scilaws_pct": r, "feynman_pct": f}

if __name__ == "__main__":
    stats = {"fig5": fig5()}; appendix(); stats["fig2"] = fig2()
    (RC.OUT / "fig_stats.json").write_text(json.dumps(stats, indent=1)); print("wrote", OUT, "and fig_stats.json")
