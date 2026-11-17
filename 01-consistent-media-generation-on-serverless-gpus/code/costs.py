"""Lesson 05 cost helpers and charts.

The notebook nb-measuring-costs.ipynb uses the helpers to turn a run's metrics_batch.json
into a cost per card, and the charts to draw the guide's four figures from its own prices,
so they follow your prices and your run. .DEV/scripts/make_05_figures.py saves the same
charts as the guide's PNGs. Chart colours mean the same as in the guide: blue is paid
once per run, orange once per card, purple is the total.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator

from common import calculate_cost

# --- Cost helpers ---------------------------------------------------------------------------

CPU_STEPS = {"text", "card", "mix"}  # steps that load no model
ALL_STEPS = {"text", "illustration", "card", "clip", "greeting", "mix"}


def read_run(path: str | Path) -> dict:
    """One run's metrics_batch.json."""
    return json.loads(Path(path).expanduser().read_text())


def model_loads(steps: dict) -> float:
    """Sum of the model load times. Only a CPU step may have no load time; a missing GPU load is an error."""
    total = 0.0
    for name, step in steps.items():
        if step["load_s"] is None:
            if name not in CPU_STEPS:
                raise ValueError(f"{name}: load_s is missing. Measure it, or leave this run out")
            continue
        total += step["load_s"]
    return total


def card_times(steps: dict) -> list[float]:
    """Seconds per card for each kid: one card is the sum of its step times."""
    kids = list(next(iter(steps.values()))["latency_s"])
    return [sum(step["latency_s"][kid] for step in steps.values()) for kid in kids]


def print_steps(steps: dict) -> None:
    """Per-step table: where it runs, load once per run, first and median time per card."""
    kids = list(next(iter(steps.values()))["latency_s"])
    t_gen = statistics.median(card_times(steps))
    print(f"{'step':<13}{'runs on':<9}{'load s':>9}{'first s':>9}{'median s':>10}{'% of card':>11}")
    for name, step in steps.items():
        load = f"{step['load_s']:.1f}" if step["load_s"] is not None else "-"
        first = step["latency_s"][kids[0]]
        share = 100 * step["median_s"] / t_gen
        where = "CPU" if name in CPU_STEPS else "GPU"
        print(f"{name:<13}{where:<9}{load:>9}{first:>9.1f}{step['median_s']:>10.1f}{share:>10.1f}%")


def card_cost(steps: dict, t_start: float, rate: float, counts=(1, 10, 20, 200)) -> dict:
    """calculate_cost() for whole cards: all model loads once, every step per card."""
    return calculate_cost(t_start, model_loads(steps), card_times(steps), rate, counts=counts)


def clip_cost(steps: dict, t_start: float, rate: float, api_price: float,
              counts=(1, 10, 20, 200)) -> dict:
    """calculate_cost() for the clip step alone, with the breakeven against api_price."""
    clip = steps["clip"]
    return calculate_cost(t_start, clip["load_s"], list(clip["latency_s"].values()), rate,
                          counts=counts, api_price=api_price)


def api_price_per_clip(api: dict, num_frames: int) -> float:
    """USD per clip for an API that prices per video second: frames / fps x price per second."""
    return api["usd_per_video_s"] * num_frames / api["frames_per_video_s"]


def rate_for(run: dict, prices: dict) -> tuple[float, str]:
    """The price per hour for a run: the row in prices with the run's platform, spot if preemptible."""
    pricing = "spot" if run.get("billing") == "preemptible" else "regular"
    rows = [g for g, pr in prices.items() if pr["platform"] == run.get("platform")]
    if not rows:
        raise ValueError(f"No price for platform {run.get('platform')!r}: add a row or pass a rate")
    row = prices[rows[0]]
    return row[pricing], f"{rows[0]} {pricing}, {row['region']}"


def per_unit(t_start: float, load: float, times: list[float], rate: float, n: float) -> float:
    """Projected USD per asset at batch size n: F / n + c."""
    cost = calculate_cost(t_start, load, times, rate, counts=(1,))
    return cost["fixed_run_usd"] / n + cost["marginal_compute_usd"]


# --- Charts ---------------------------------------------------------------------------------

BLUE, ORANGE, PURPLE, LIGHT_PURPLE = "#2563EB", "#EA580C", "#9333EA", "#C084FC"
INK, GREY, LIGHT = "#1F2937", "#6B7280", "#D1D5DB"
STEP_LABELS = {
    "text": "Text (CPU)",
    "illustration": "Illustration, Sana",
    "card": "Card (CPU)",
    "clip": "Clip, Wan2.2 I2V",
    "greeting": "Greeting, Kokoro",
    "mix": "Mix (CPU)",
}

# The module's Inter fonts, so the charts look the same on every machine
for _font in (Path(__file__).resolve().parent.parent / "assets" / "fonts").glob("Inter-*.ttf"):
    font_manager.fontManager.addfont(str(_font))

STYLE = {
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": LIGHT,
    "axes.labelcolor": INK,
    "xtick.color": GREY,
    "ytick.color": INK,
    "axes.titlecolor": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def _usd(x: float, _pos: int | None = None) -> str:
    """Axis ticks: two decimals; '$' escaped so matplotlib doesn't read it as maths."""
    return f"\\${x:.2f}"


def _usd3(x: float) -> str:
    """Value labels: three decimals below $0.10, so small differences stay visible."""
    return f"\\${x:.2f}" if x >= 0.1 else f"\\${x:.3f}"


def time_chart(steps: dict) -> Figure:
    """Where the time goes: model load per run (blue) and median time per card (orange)."""
    order = [s for s in reversed(STEP_LABELS) if s in steps]  # drawn bottom-up, read top-down
    labels = [STEP_LABELS[s] for s in order]
    with plt.rc_context(STYLE):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True,
                                          gridspec_kw={"wspace": 0.08})
        loads = [steps[s]["load_s"] for s in order]
        left.barh(labels, [v or 0 for v in loads], color=BLUE, height=0.6)
        top = max(v or 0 for v in loads)
        for y, v in enumerate(loads):
            text = "no model" if v is None else f"{v:,.0f} s"
            left.text((v or 0) + top * 0.012, y, text, va="center",
                      color=GREY if v is None else INK)
        left.set_xlim(0, top * 1.25)
        left.set_title("Model loading: seconds per run, paid once", loc="left", color=BLUE)
        left.set_xlabel("seconds per run")

        medians = [steps[s]["median_s"] for s in order]
        right.barh(labels, medians, color=ORANGE, height=0.6)
        for y, v in enumerate(medians):
            right.text(v + max(medians) * 0.012, y, "<0.1 s" if v < 0.1 else f"{v:.1f} s",
                       va="center", color=INK)
        right.set_xlim(0, max(medians) * 1.24)
        right.set_title("Generation: median seconds per card", loc="left", color=ORANGE)
        right.set_xlabel("seconds per card")

        for ax in (left, right):
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        fig.suptitle("Where does the time of one run go?", x=0.125, y=1.03, ha="left",
                     fontsize=14, fontweight="bold", color=INK)
    return fig


def cost_vs_n_chart(steps: dict, t_start: float, rate: float, max_n: int = 200) -> Figure:
    """Projected cost per card against N, as measured and with two best-case changes."""
    t_gen = statistics.median(card_times(steps))
    load = model_loads(steps)
    clip = steps["clip"]
    ns = [n / 4 for n in range(4, 4 * max_n + 1)]
    lines = [
        ("Current worker", load, t_gen, PURPLE, "-"),
        ("Clip weights cached (best case: clip load time set to zero)",
         load - clip["load_s"], t_gen, BLUE, "--"),
        ("Clip generation twice as fast", load, t_gen - clip["median_s"] / 2, ORANGE, ":"),
    ]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for label, ld, gen, colour, style in lines:
            ys = [per_unit(t_start, ld, [gen], rate, n) for n in ns]
            ax.plot(ns, ys, color=colour, linestyle=style, linewidth=2.4, label=label)
        ax.legend(loc="upper right", frameon=False, fontsize=10, handlelength=3)
        for n in (1, 10, 20, 200):
            if n > max_n:
                continue
            y = per_unit(t_start, load, [t_gen], rate, n)
            ax.plot(n, y, "o", color=PURPLE, markersize=6)
            ax.annotate(f"N={n}: {_usd3(y)}", (n, y), textcoords="offset points",
                        xytext=(-60, 8) if n == 200 else (6, 6), color=PURPLE, fontsize=10)
        crossover = clip["load_s"] / (clip["median_s"] / 2)
        top = per_unit(t_start, load, [t_gen], rate, 1)
        ax.axvline(crossover, color=LIGHT, linewidth=1)
        ax.text(crossover * 1.04, top * 0.35, f"N ≈ {crossover:.0f}: below it, caching\n"
                "saves more; above it, a faster clip does", color=GREY, fontsize=9, va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([n for n in (1, 10, 20, 50, 100, 200) if n <= max_n])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:g}"))
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(_usd))
        ax.set_xlim(0.9, max_n)
        ax.set_xlabel("cards per run, N (log scale)")
        ax.set_ylabel("projected USD per card (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("How does the cost per card fall as the batch grows?", loc="left",
                     fontsize=14, fontweight="bold")
    return fig


def spot_chart(rows: list[tuple[str, float, float]]) -> Figure:
    """Marginal cost per card c for each GPU: regular price (dot) and spot price (diamond).

    rows: (GPU, c at the regular price, c at the spot price), in display order.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 3.6))
        for y, (_gpu, reg, spot) in enumerate(reversed(rows)):
            ax.plot([spot, reg], [y, y], color=LIGHT, linewidth=3, zorder=1)
            ax.plot(reg, y, "o", color=INK, markersize=10, zorder=2)
            ax.plot(spot, y, "D", color="white", markeredgecolor=INK, markeredgewidth=2,
                    markersize=9, zorder=2)
            ax.text(reg, y + 0.22, _usd3(reg), ha="center", color=INK, fontsize=10)
            ax.text(spot, y + 0.22, _usd3(spot), ha="center", color=GREY, fontsize=10)
        ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
        ax.set_ylim(-0.6, len(rows) - 0.3)
        values = [v for r in rows for v in r[1:]]
        ax.set_xlim(min(values) * 0.7, max(values) * 1.15)
        ax.xaxis.set_major_formatter(FuncFormatter(_usd))
        ax.set_xlabel("marginal USD per card, c")
        ax.plot([], [], "o", color=INK, label="regular price")
        ax.plot([], [], "D", color="white", markeredgecolor=INK, markeredgewidth=2,
                label="spot price snapshot")
        ax.legend(loc="lower right", frameon=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title("Does the cheapest GPU change at spot prices?", loc="left", fontsize=14,
                     fontweight="bold")
    return fig


def breakeven_chart(clip: dict, t_start: float, rates: dict[str, float], api_price: float,
                    gpu: str = "H100", max_n: int = 40) -> Figure:
    """Projected USD per clip against clips per run, one curve per price, and the API price.

    rates: {"regular": USD per hour, "spot": USD per hour}.
    """
    times = list(clip["latency_s"].values())
    ns = list(range(1, max_n + 1))
    colours = {"regular": PURPLE, "spot": LIGHT_PURPLE}
    label_offsets = {"regular": (18, -40), "spot": (-20, -70)}
    at_one = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.axhline(api_price, color=INK, linestyle="--", linewidth=1.5)
        ax.text(max_n, api_price + 0.012, f"API: \\${api_price:.4f} per clip", ha="right",
                color=INK)
        for kind, rate in rates.items():
            colour = colours.get(kind, GREY)
            ys = [per_unit(t_start, clip["load_s"], times, rate, n) for n in ns]
            at_one.append(f"\\${ys[0]:.2f} {kind}")
            ax.plot(ns, ys, color=colour, linewidth=2.4, marker="o", markersize=3,
                    label=f"Self-hosted {gpu}, {kind} price (\\${rate:.2f}/h)")
            n_star = calculate_cost(t_start, clip["load_s"], times, rate,
                                    api_price=api_price)["breakeven_assets"]
            if not n_star or n_star > max_n:
                continue
            y_star = per_unit(t_start, clip["load_s"], times, rate, n_star)
            ax.plot(n_star, y_star, "o", color=colour, markersize=9, markerfacecolor="white",
                    markeredgewidth=2)
            ax.annotate(f"{kind}: breakeven at {n_star} clips\n(\\${y_star:.3f} per clip)",
                        (n_star, y_star), textcoords="offset points",
                        xytext=label_offsets.get(kind, (18, -40)), color=INK,
                        arrowprops={"arrowstyle": "-", "color": GREY})
        ax.text(max_n * 0.35, 0.57, f"at N = 1: {', '.join(at_one)} per clip (off the chart)",
                color=GREY, fontsize=9)
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92), frameon=False, fontsize=10)
        ax.set_ylim(0, 0.6)
        ax.set_xlim(0, max_n + 1)
        ax.yaxis.set_major_formatter(FuncFormatter(_usd))
        ax.set_xlabel("clips per run")
        ax.set_ylabel("projected USD per clip")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title("From how many clips does self-hosting beat the API?", loc="left",
                     fontsize=14, fontweight="bold")
    return fig
