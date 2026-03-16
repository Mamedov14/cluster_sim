"""
plots.py — Графики и таблицы результатов экспериментов.

Воспроизводит все рисунки из диплома:
  - Рис. 1: Пропускная способность vs MTTF
  - Рис. 2: P95 задержки vs MTTF
  - Рис. 3: Сравнение политик балансировки (столбчатая)
  - Рис. 4: Сравнение типов отказов (столбчатая)
  - Рис. 5: Стратегии обработки — потери задач (столбчатая)
  - Таблица 1: Сводная таблица всех конфигураций
"""

import matplotlib
matplotlib.use("Agg")   # без GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
from pathlib import Path

# Папка для сохранения
OUT_DIR = Path("output_figures")
OUT_DIR.mkdir(exist_ok=True)

# ── Стиль ───────────────────────────────────────────
PALETTE = {
    "blue":      "#0077B6",
    "teal":      "#00B4D8",
    "green":     "#00897B",
    "orange":    "#F4A261",
    "red":       "#E63946",
    "gray":      "#ADB5BD",
    "dark":      "#1A2E3B",
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
})


def _save(name: str):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Сохранён: {path}")


# ─────────────────────────────────────────
# Рис. 1 — Пропускная способность vs MTTF
# ─────────────────────────────────────────

def plot_throughput_vs_mttf(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.errorbar(
        df["mttf"], df["throughput"],
        yerr=df["throughput_std"],
        color=PALETTE["blue"], marker="o", linewidth=2,
        capsize=4, elinewidth=1.2, markersize=6,
        label="Пропускная способность"
    )

    ax.set_xlabel("MTTF (среднее время наработки на отказ)")
    ax.set_ylabel("Пропускная способность, задач/с")
    ax.set_title("Пропускная способность vs MTTF\n(round_robin, down failures, freeze)")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    _save("fig1_throughput_vs_mttf")


# ─────────────────────────────────────────
# Рис. 2 — P95 задержки vs MTTF
# ─────────────────────────────────────────

def plot_p95_vs_mttf(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.errorbar(
        df["mttf"], df["wait_p95"],
        yerr=df["wait_p95_std"],
        color=PALETTE["teal"], marker="s", linewidth=2,
        capsize=4, elinewidth=1.2, markersize=6,
        label="95-й процентиль времени ожидания"
    )

    ax.set_xlabel("MTTF (среднее время наработки на отказ)")
    ax.set_ylabel("Время ожидания P95, с")
    ax.set_title("95-й процентиль времени ожидания vs MTTF")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))

    _save("fig2_p95_vs_mttf")


# ─────────────────────────────────────────
# Рис. 3 — Политики балансировки
# ─────────────────────────────────────────

def plot_balance_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"]]
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.55

    # P95 задержки
    ax = axes[0]
    bars = ax.bar(x, df["wait_p95"], width=width,
                  color=colors, edgecolor="white", linewidth=0.8)
    ax.bar(x, df["wait_p95_std"], width=width, bottom=df["wait_p95"] - df["wait_p95_std"],
           color="none", edgecolor="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Время ожидания P95, с")
    ax.set_title("P95 задержки по политикам балансировки")

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10)

    # Пропускная способность
    ax = axes[1]
    bars2 = ax.bar(x, df["throughput"], width=width,
                   color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Пропускная способность, задач/с")
    ax.set_title("Пропускная способность по политикам балансировки")

    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    _save("fig3_balance_policies")


# ─────────────────────────────────────────
# Рис. 4 — down vs degrade
# ─────────────────────────────────────────

def plot_failure_type_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = [PALETTE["blue"], PALETTE["red"]]
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.45

    # P95
    ax = axes[0]
    bars = ax.bar(x, df["wait_p95"], width=width, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Время ожидания P95, с")
    ax.set_title("P95 задержки: down vs degrade")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=11)

    # Потери задач
    ax = axes[1]
    bars2 = ax.bar(x, df["tasks_dropped"], width=width, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Потери задач (среднее за прогон)")
    ax.set_title("Потери задач: down vs degrade")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    _save("fig4_failure_types")


# ─────────────────────────────────────────
# Рис. 5 — Стратегии обработки отказов
# ─────────────────────────────────────────

def plot_on_fail_comparison(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    colors = [PALETTE["teal"], PALETTE["orange"], PALETTE["red"]]
    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.45

    # Потери задач
    ax = axes[0]
    bars = ax.bar(x, df["tasks_dropped"], width=width, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Потери задач (среднее за прогон)")
    ax.set_title("Потери задач по стратегиям обработки")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=11)

    # P95 задержки
    ax = axes[1]
    bars2 = ax.bar(x, df["wait_p95"], width=width, color=colors, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Время ожидания P95, с")
    ax.set_title("P95 задержки по стратегиям обработки")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    _save("fig5_on_fail_policies")


# ─────────────────────────────────────────
# Таблица 1 — Сводная
# ─────────────────────────────────────────

def make_summary_table(results: dict) -> pd.DataFrame:
    """Собирает сводную таблицу как в оригинальной статье."""
    rows = []

    def fmt(val, std=None):
        if std is not None:
            return f"{val:.3f}±{std:.3f}"
        return f"{val:.3f}"

    def add_rows(df, label_col="label"):
        for _, r in df.iterrows():
            rows.append({
                "Конфигурация":        r.get(label_col, "—"),
                "X (задач/с)":         fmt(r["throughput"],    r.get("throughput_std")),
                "W (с)":               fmt(r["mean_wait"],     r.get("mean_wait_std")),
                "W₀.₉₅ (с)":          fmt(r["wait_p95"],      r.get("wait_p95_std")),
                "U (доли)":            fmt(r["mean_util"],     r.get("mean_util_std")),
                "Потери задач":        fmt(r["tasks_dropped"], r.get("tasks_dropped_std")),
            })

    add_rows(results["baseline"])
    add_rows(results["mttf"], "label")
    add_rows(results["balance"], "label")
    add_rows(results["failure_type"], "label")
    add_rows(results["on_fail"], "label")
    add_rows(results["deterministic"])

    df = pd.DataFrame(rows)
    return df


def save_summary_table(df: pd.DataFrame):
    """Сохраняет таблицу как CSV."""
    path = OUT_DIR / "table1_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Сохранена таблица: {path}")


# ─────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────

def plot_all(results: dict):
    print("Построение графиков...")

    plot_throughput_vs_mttf(results["mttf"])
    plot_p95_vs_mttf(results["mttf"])
    plot_balance_comparison(results["balance"])
    plot_failure_type_comparison(results["failure_type"])
    plot_on_fail_comparison(results["on_fail"])

    table = make_summary_table(results)
    save_summary_table(table)

    print("\n=== Сводная таблица ===")
    print(table.to_string(index=False))

    return table
