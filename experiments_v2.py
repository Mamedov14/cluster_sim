"""
experiments_v2.py — Эксперименты для двухуровневой архитектуры с контейнерами.

Серии экспериментов:
  1. Базовый режим — без отказов, разное число контейнеров (воспроизводит рис. 9 статьи)
  2. Влияние числа контейнеров на P95 задержки
  3. Сравнение политик балансировки (уровень 0)
  4. Отказы узла: down vs degrade
  5. Отказы контейнера: влияние MTTR контейнера
  6. Сравнение: одноуровневая vs двухуровневая архитектура
"""

import pandas as pd
import numpy as np
from simulator_v2 import SimConfigV2, run_replications_v2

N_REPS = 10


# ─────────────────────────────────────────
# 1. Число контейнеров vs среднее время пребывания
# ─────────────────────────────────────────

def exp_containers_sojourn() -> pd.DataFrame:
    """Воспроизводит рис. 9 статьи Фунга: T vs число контейнеров."""
    rows = []
    for c in range(1, 11):
        cfg = SimConfigV2(
            num_nodes           = 3,
            containers_per_node = c,
            arrival_rate        = 3.0,
            base_service_rate   = 6.0,
            sim_time            = 500.0,
            balance_policy      = "round_robin",
        )
        res = run_replications_v2(cfg, N_REPS)
        res["containers"] = c
        res["label"]      = f"C={c}"
        rows.append(res)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 2. Политики балансировки (уровень 0) с контейнерами
# ─────────────────────────────────────────

def exp_balance_with_containers() -> pd.DataFrame:
    rows = []
    for policy in ["round_robin", "least_loaded", "random"]:
        cfg = SimConfigV2(
            num_nodes           = 3,
            containers_per_node = 3,
            arrival_rate        = 3.0,
            base_service_rate   = 6.0,
            sim_time            = 500.0,
            balance_policy      = policy,
            node_failure_type   = "down",
            node_mttf           = 35.0,
            node_mttr           = 15.0,
        )
        res = run_replications_v2(cfg, N_REPS)
        res["policy"] = policy
        res["label"]  = policy
        rows.append(res)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 3. Отказы узла: down vs degrade
# ─────────────────────────────────────────

def exp_node_failure_types() -> pd.DataFrame:
    rows = []
    for ftype in ["down", "degrade"]:
        cfg = SimConfigV2(
            num_nodes           = 3,
            containers_per_node = 3,
            arrival_rate        = 3.0,
            base_service_rate   = 6.0,
            sim_time            = 500.0,
            balance_policy      = "round_robin",
            node_failure_type   = ftype,
            node_mttf           = 35.0,
            node_mttr           = 15.0,
            node_degrade_factor = 2.0,
        )
        res = run_replications_v2(cfg, N_REPS)
        res["failure_type"] = ftype
        res["label"]        = ftype
        rows.append(res)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 4. Отказы контейнера: влияние MTTR контейнера
# ─────────────────────────────────────────

def exp_container_failures() -> pd.DataFrame:
    rows = []
    mttr_values = [5.0, 10.0, 20.0, 40.0]
    for mttr in mttr_values:
        cfg = SimConfigV2(
            num_nodes           = 3,
            containers_per_node = 3,
            arrival_rate        = 3.0,
            base_service_rate   = 6.0,
            sim_time            = 500.0,
            balance_policy      = "round_robin",
            container_mttf      = 50.0,
            container_mttr      = mttr,
        )
        res = run_replications_v2(cfg, N_REPS)
        res["container_mttr"] = mttr
        res["label"]          = f"MTTR_c={mttr}"
        rows.append(res)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 5. Влияние MTTF узла (с контейнерами)
# ─────────────────────────────────────────

def exp_node_mttf() -> pd.DataFrame:
    rows = []
    for mttf in [10, 20, 35, 40, 60, 80]:
        cfg = SimConfigV2(
            num_nodes           = 3,
            containers_per_node = 3,
            arrival_rate        = 3.0,
            base_service_rate   = 6.0,
            sim_time            = 500.0,
            balance_policy      = "round_robin",
            node_failure_type   = "down",
            node_mttf           = float(mttf),
            node_mttr           = 12.0,
        )
        res = run_replications_v2(cfg, N_REPS)
        res["mttf"]  = mttf
        res["label"] = f"MTTF={mttf}"
        rows.append(res)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 6. Совместные отказы: узел + контейнер
# ─────────────────────────────────────────

def exp_combined_failures() -> pd.DataFrame:
    """Сравниваем: только узел, только контейнер, оба."""
    configs = [
        {"label": "Без отказов",          "node_mttf": 0,    "container_mttf": 0},
        {"label": "Только отказы узла",   "node_mttf": 35.0, "container_mttf": 0},
        {"label": "Только отказы конт.",  "node_mttf": 0,    "container_mttf": 50.0},
        {"label": "Оба вида отказов",     "node_mttf": 35.0, "container_mttf": 50.0},
    ]
    rows = []
    for c in configs:
        cfg = SimConfigV2(
            num_nodes           = 3,
            containers_per_node = 3,
            arrival_rate        = 3.0,
            base_service_rate   = 6.0,
            sim_time            = 500.0,
            balance_policy      = "round_robin",
            node_failure_type   = "down",
            node_mttf           = c["node_mttf"],
            node_mttr           = 15.0,
            container_mttf      = c["container_mttf"],
            container_mttr      = 10.0,
        )
        res = run_replications_v2(cfg, N_REPS)
        res["label"] = c["label"]
        rows.append(res)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Запуск всех
# ─────────────────────────────────────────

def run_all_v2() -> dict:
    print("Запуск экспериментов v2 (двухуровневая архитектура)...")
    results = {}

    print("  [1/6] Число контейнеров vs время пребывания...")
    results["containers"] = exp_containers_sojourn()

    print("  [2/6] Политики балансировки с контейнерами...")
    results["balance"] = exp_balance_with_containers()

    print("  [3/6] Типы отказов узла...")
    results["node_failure"] = exp_node_failure_types()

    print("  [4/6] Отказы контейнеров (MTTR)...")
    results["container_failure"] = exp_container_failures()

    print("  [5/6] Влияние MTTF узла...")
    results["node_mttf"] = exp_node_mttf()

    print("  [6/6] Совместные отказы...")
    results["combined"] = exp_combined_failures()

    print("Готово.")
    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    OUT = Path("output_figures_v2")
    OUT.mkdir(exist_ok=True)

    results = run_all_v2()

    # ── Рис. 1: T vs число контейнеров ──
    df = results["containers"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(df["containers"], df["mean_sojourn"], yerr=df["mean_sojourn_std"],
                color="#0077B6", marker="o", linewidth=2, capsize=4)
    ax.set_xlabel("Число контейнеров на узле (C)")
    ax.set_ylabel("Среднее время пребывания T, с")
    ax.set_title("Зависимость времени пребывания от числа контейнеров\n(3 узла, без отказов)")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_containers_sojourn.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рис. 2: MTTF vs P95 (с контейнерами) ──
    df2 = results["node_mttf"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].errorbar(df2["mttf"], df2["throughput"], yerr=df2["throughput_std"],
                     color="#0077B6", marker="o", linewidth=2, capsize=4)
    axes[0].set_xlabel("MTTF узла")
    axes[0].set_ylabel("Пропускная способность, задач/с")
    axes[0].set_title("Пропускная способность vs MTTF")
    axes[0].grid(True, alpha=0.35, linestyle="--")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].errorbar(df2["mttf"], df2["wait_p95"], yerr=df2["wait_p95_std"],
                     color="#00B4D8", marker="s", linewidth=2, capsize=4)
    axes[1].set_xlabel("MTTF узла")
    axes[1].set_ylabel("P95 времени ожидания, с")
    axes[1].set_title("P95 задержки vs MTTF")
    axes[1].grid(True, alpha=0.35, linestyle="--")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_mttf_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рис. 3: Политики балансировки ──
    df3 = results["balance"]
    colors = ["#0077B6", "#00897B", "#F4A261"]
    labels = df3["label"].tolist()
    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(labels, df3["wait_p95"], color=colors, edgecolor="white")
    for i, v in enumerate(df3["wait_p95"]):
        axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=10)
    axes[0].set_ylabel("P95 времени ожидания, с")
    axes[0].set_title("P95 задержки по политикам балансировки")
    axes[0].grid(True, alpha=0.35, linestyle="--", axis="y")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].bar(labels, df3["throughput"], color=colors, edgecolor="white")
    for i, v in enumerate(df3["throughput"]):
        axes[1].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("Пропускная способность, задач/с")
    axes[1].set_title("Пропускная способность по политикам")
    axes[1].grid(True, alpha=0.35, linestyle="--", axis="y")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_balance_v2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Рис. 4: Совместные отказы ──
    df4 = results["combined"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cols4 = ["#00897B", "#0077B6", "#F4A261", "#E63946"]
    labels4 = df4["label"].tolist()
    axes[0].bar(labels4, df4["wait_p95"], color=cols4, edgecolor="white")
    for i, v in enumerate(df4["wait_p95"]):
        axes[0].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=10)
    axes[0].set_ylabel("P95 времени ожидания, с")
    axes[0].set_title("P95 задержки: типы отказов")
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.35, linestyle="--", axis="y")
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)

    axes[1].bar(labels4, df4["tasks_dropped"], color=cols4, edgecolor="white")
    for i, v in enumerate(df4["tasks_dropped"]):
        axes[1].text(i, v + 0.3, f"{v:.0f}", ha="center", fontsize=10)
    axes[1].set_ylabel("Потери задач (среднее)")
    axes[1].set_title("Потери задач: типы отказов")
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.35, linestyle="--", axis="y")
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT / "fig_combined_failures.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ Графики сохранены в {OUT}/")

    # Сводная таблица
    rows_table = []
    for key, df in results.items():
        for _, r in df.iterrows():
            rows_table.append({
                "Эксперимент": key,
                "Конфигурация": r.get("label", "—"),
                "X": f"{r['throughput']:.3f}±{r['throughput_std']:.3f}",
                "W": f"{r['mean_wait']:.3f}±{r['mean_wait_std']:.3f}",
                "W₀.₉₅": f"{r['wait_p95']:.3f}±{r['wait_p95_std']:.3f}",
                "U_node": f"{r['mean_node_util']:.3f}",
                "Потери": f"{r['tasks_dropped']:.1f}±{r['tasks_dropped_std']:.1f}",
            })
    import pandas as pd
    table = pd.DataFrame(rows_table)
    table.to_csv(OUT / "table_v2.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
