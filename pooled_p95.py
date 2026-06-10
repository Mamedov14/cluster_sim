"""
pooled_p95.py — Pooled P95 анализ + верификация M/M/c.

Решает два замечания рецензента к статье INJOIT:
  1. Pooled P95 — единый процентиль по объединённой выборке всех прогонов.
  2. Верификация M/M/c — сравнение с аналитической формулой Эрланга-C.

Используется simulator_v3.py (классы Node, LoadBalancerNode, TaskGenerator).
"""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import simpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator_v3 import (
    SimConfig, Node, LoadBalancerNode, TaskGenerator,
)

OUT_DIR = Path("output_figures_pooled")
OUT_DIR.mkdir(exist_ok=True)

SIM_TIME = 3000.0
N_REPS   = 10


def run_simulation_raw(cfg):
    rng       = np.random.default_rng(cfg.seed)
    env       = simpy.Environment()
    repairman = simpy.PriorityResource(env, capacity=cfg.repair_capacity)
    nodes = [Node(env, i, cfg, rng, repairman) for i in range(cfg.num_nodes)]
    disp  = LoadBalancerNode(env, nodes, cfg.balance_policy, cfg, rng, repairman)
    gen   = TaskGenerator(env, disp, cfg, rng)
    env.run(until=cfg.sim_time)

    completed = [t for t in gen.all_tasks if t.finish_time > 0 and t.start_time >= 0]
    waits    = [t.start_time - t.arrival_time for t in completed]
    sojourns = [t.finish_time - t.arrival_time for t in completed]
    dropped  = disp.dropped + sum(n.dropped for n in nodes)
    summary = {
        "throughput":     len(completed) / cfg.sim_time,
        "mean_wait":      float(np.mean(waits)) if waits else 0.0,
        "wait_p95":       float(np.percentile(waits, 95)) if waits else 0.0,
        "mean_sojourn":   float(np.mean(sojourns)) if sojourns else 0.0,
        "n_dropped":      int(dropped),
        "n_total":        len(gen.all_tasks),
        "n_completed":    len(completed),
    }
    return waits, summary


def run_replications_pooled(cfg, n_reps=N_REPS):
    all_waits = []
    per_run_p95 = []
    per_run_mean = []
    per_run_through = []
    per_run_dropped = []
    per_run_total = []
    for i in range(n_reps):
        sub = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i})
        waits, s = run_simulation_raw(sub)
        all_waits.extend(waits)
        per_run_p95.append(s["wait_p95"])
        per_run_mean.append(s["mean_wait"])
        per_run_through.append(s["throughput"])
        per_run_dropped.append(s["n_dropped"])
        per_run_total.append(s["n_total"])
    if all_waits:
        pooled_p95  = float(np.percentile(all_waits, 95))
        pooled_mean = float(np.mean(all_waits))
    else:
        pooled_p95 = 0.0
        pooled_mean = 0.0
    return {
        "pooled_w95":        pooled_p95,
        "pooled_mean_wait":  pooled_mean,
        "mean_w95":          float(np.mean(per_run_p95)),
        "std_w95":           float(np.std(per_run_p95)),
        "throughput":        float(np.mean(per_run_through)),
        "throughput_std":    float(np.std(per_run_through)),
        "n_dropped":         float(np.mean(per_run_dropped)),
        "n_total":           float(np.mean(per_run_total)),
        "n_total_waits":     len(all_waits),
    }


def erlang_c_wq(lam, mu, c):
    rho = lam / (c * mu)
    if rho >= 1.0:
        return rho, float("inf")
    a = lam / mu
    sum_terms = sum((a ** k) / math.factorial(k) for k in range(c))
    last_term = (a ** c) / (math.factorial(c) * (1 - rho))
    p0 = 1.0 / (sum_terms + last_term)
    c_erlang = last_term * p0
    lq = c_erlang * rho / (1 - rho)
    wq = lq / lam
    return rho, wq


def base_cfg(**overrides):
    d = dict(
        num_nodes=3, containers_per_node=3,
        arrival_rate=16.0, base_service_rate=22.242,
        sim_time=SIM_TIME,
        balance_policy="least_loaded",
        node_mttf_fail=3500.0, degrade_percent=0.8, node_mttr=15.0,
        container_mttf=500.0, container_mttr=2.0,
        seed=42,
    )
    d.update(overrides)
    return SimConfig(**d)


def series_lambda():
    rows = []
    for lam in [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]:
        cfg = base_cfg(arrival_rate=lam)
        t0 = time.time()
        r = run_replications_pooled(cfg)
        r["lambda"] = lam
        r["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  lam={lam:>5.1f}  pW95={r['pooled_w95']:.3f}  "
              f"mW95={r['mean_w95']:.3f}+/-{r['std_w95']:.3f}  X={r['throughput']:.3f}  "
              f"t={r['elapsed_s']}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "series_lambda.csv", index=False, encoding="utf-8-sig")
    return df


def series_mttf():
    rows = []
    for mttf in [2500, 3000, 3500, 4000, 4500]:
        for p in [0.7, 0.8, 0.9]:
            cfg = base_cfg(node_mttf_fail=float(mttf), degrade_percent=p)
            t0 = time.time()
            r = run_replications_pooled(cfg)
            r["mttf_fail"] = mttf
            r["degrade_percent"] = p
            r["elapsed_s"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"  MTTF={mttf:>5}  p={p}  pW95={r['pooled_w95']:.3f}  "
                  f"mW95={r['mean_w95']:.3f}+/-{r['std_w95']:.3f}  t={r['elapsed_s']}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "series_mttf.csv", index=False, encoding="utf-8-sig")
    return df


def series_mttr():
    rows = []
    for mttr in [5, 10, 15, 20, 25]:
        cfg = base_cfg(node_mttr=float(mttr))
        t0 = time.time()
        r = run_replications_pooled(cfg)
        r["mttr"] = mttr
        r["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  MTTR={mttr:>3}  pW95={r['pooled_w95']:.3f}  "
              f"mW95={r['mean_w95']:.3f}+/-{r['std_w95']:.3f}  t={r['elapsed_s']}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "series_mttr.csv", index=False, encoding="utf-8-sig")
    return df


def series_policy():
    rows = []
    for pol in ["round_robin", "least_loaded", "random"]:
        cfg = base_cfg(balance_policy=pol)
        t0 = time.time()
        r = run_replications_pooled(cfg)
        r["policy"] = pol
        r["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(r)
        print(f"  policy={pol:>13}  pW95={r['pooled_w95']:.3f}  "
              f"mW95={r['mean_w95']:.3f}+/-{r['std_w95']:.3f}  t={r['elapsed_s']}s")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "series_policy.csv", index=False, encoding="utf-8-sig")
    return df


def verification_mmc():
    rows = []
    mu = 22.242
    c = 3
    for lam in [8.0, 12.0, 16.0]:
        cfg = SimConfig(
            num_nodes=1, containers_per_node=3,
            arrival_rate=lam, base_service_rate=mu,
            sim_time=SIM_TIME,
            balance_policy="least_loaded",
            node_mttf_fail=1e12, degrade_percent=0.0, node_mttr=1.0,
            container_mttf=1e12, container_mttr=1.0,
            lb_service_rate=1e6,
            seed=42,
        )
        t0 = time.time()
        r = run_replications_pooled(cfg)
        rho_ana, wq_ana = erlang_c_wq(lam, mu, c)
        rho_sim = lam / (c * mu)
        wq_sim = r["pooled_mean_wait"]
        err_pct = abs(wq_sim - wq_ana) / max(wq_ana, 1e-9) * 100
        rho_err_pct = abs(rho_sim - rho_ana) / max(rho_ana, 1e-9) * 100
        rows.append({
            "lambda":         lam,
            "rho_analytic":   round(rho_ana, 4),
            "rho_sim":        round(rho_sim, 4),
            "rho_err_pct":    round(rho_err_pct, 3),
            "Wq_analytic_s":  round(wq_ana, 4),
            "Wq_sim_s":       round(wq_sim, 4),
            "err_pct":        round(err_pct, 2),
            "n_total_waits":  r["n_total_waits"],
            "elapsed_s":      round(time.time() - t0, 1),
        })
        print(f"  lam={lam}  rho_an={rho_ana:.3f}  Wq_an={wq_ana:.4f}  Wq_sim={wq_sim:.4f}  "
              f"err={err_pct:.2f}%")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "verification_mmc.csv", index=False, encoding="utf-8-sig")
    return df


def _setup_plot_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelcolor": "black",
        "axes.edgecolor":  "black",
        "xtick.color":     "black",
        "ytick.color":     "black",
        "text.color":      "black",
        "axes.facecolor":  "white",
        "figure.facecolor":"white",
        "savefig.facecolor":"white",
    })


def fig_lambda(df):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(df["lambda"], df["pooled_w95"], yerr=df["std_w95"],
                marker="o", lw=2, capsize=4, color="black",
                ecolor="black", label="pooled W0.95")
    ax.set_xlabel("Интенсивность λ (задач/с)")
    ax.set_ylabel("pooled W0.95 (с)")
    ax.set_title("Pooled 95-й процентиль времени ожидания от λ")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_lambda_pooled_p95.png", dpi=150)
    plt.close(fig)


def fig_policy(df):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    labels = df["policy"].tolist()
    y = df["pooled_w95"].values
    yerr = df["std_w95"].values
    xpos = np.arange(len(labels))
    ax.bar(xpos, y, yerr=yerr, capsize=6, color="white", edgecolor="black", lw=1.5,
           ecolor="black")
    for xi, vi in zip(xpos, y):
        ax.text(xi, vi, f"{vi:.3f}", ha="center", va="bottom", color="black")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Политика балансировки")
    ax.set_ylabel("pooled W0.95 (с)")
    ax.set_title("Pooled 95-й процентиль времени ожидания по политикам")
    ax.grid(True, ls="--", alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_policy_pooled_p95.png", dpi=150)
    plt.close(fig)


def fig_mttr(df):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(df["mttr"], df["pooled_w95"], yerr=df["std_w95"],
                marker="s", lw=2, capsize=4, color="black",
                ecolor="black", label="pooled W0.95")
    ax.set_xlabel("MTTR узла (с)")
    ax.set_ylabel("pooled W0.95 (с)")
    ax.set_title("Pooled 95-й процентиль времени ожидания от MTTR")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_mttr_pooled_p95.png", dpi=150)
    plt.close(fig)


def fig_mttf(df):
    _setup_plot_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    markers = {0.7: "o", 0.8: "s", 0.9: "^"}
    linestyles = {0.7: "-", 0.8: "--", 0.9: ":"}
    for p in [0.7, 0.8, 0.9]:
        sub = df[df["degrade_percent"] == p].sort_values("mttf_fail")
        ax.errorbar(sub["mttf_fail"], sub["pooled_w95"], yerr=sub["std_w95"],
                    marker=markers[p], ls=linestyles[p], lw=2, capsize=4,
                    color="black", ecolor="black", label=f"degrade {int(p*100)}%")
    ax.set_xlabel("MTTF узла (с)")
    ax.set_ylabel("pooled W0.95 (с)")
    ax.set_title("Pooled 95-й процентиль времени ожидания от MTTF")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_mttf_pooled_p95.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    print(f"sim_time = {SIM_TIME} c, repeats = {N_REPS}\n")
    t_all = time.time()

    print("=== Series lambda ===")
    df_l = series_lambda()

    print("\n=== Series MTTR ===")
    df_r = series_mttr()

    print("\n=== Series policy ===")
    df_p = series_policy()

    print("\n=== Series MTTF ===")
    df_m = series_mttf()

    print("\n=== Verification M/M/c ===")
    df_v = verification_mmc_clean()

    print("\n--- plots ---")
    fig_lambda(df_l)
    fig_policy(df_p)
    fig_mttr(df_r)
    fig_mttf(df_m)

    print(f"\nDone in {time.time() - t_all:.0f} s")
    print(f"Files in '{OUT_DIR}/'")


# =====================================================================
# MMc verification — independent minimal M/M/c simulator
# =====================================================================

def run_mmc_sim(lam: float, mu: float, c: int, sim_time: float, seed: int):
    """
    Чистый M/M/c: c независимых серверов, бесконечная очередь, FCFS,
    каждый сервер обслуживает ровно с интенсивностью mu (нет таблицы μ(n,m)).
    Возвращает список waits.
    """
    import simpy
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    servers = simpy.Resource(env, capacity=c)

    waits = []
    arrivals_total = [0]

    def task(t_arr):
        with servers.request() as req:
            yield req
            t_start = env.now
            waits.append(t_start - t_arr)
            yield env.timeout(rng.exponential(1.0 / mu))

    def arrivals():
        while True:
            yield env.timeout(rng.exponential(1.0 / lam))
            if env.now >= sim_time:
                break
            arrivals_total[0] += 1
            env.process(task(env.now))

    env.process(arrivals())
    env.run(until=sim_time)
    return waits, arrivals_total[0]


def verification_mmc_clean():
    """Чистая верификация: запускаем настоящий M/M/c, сравниваем с Эрлангом-C."""
    rows = []
    mu = 22.242
    c = 3
    n_reps = 10
    sim_time = SIM_TIME
    for lam in [8.0, 12.0, 16.0]:
        t0 = time.time()
        all_waits = []
        total_arr = 0
        for i in range(n_reps):
            waits, arr = run_mmc_sim(lam, mu, c, sim_time, seed=42 + i)
            all_waits.extend(waits)
            total_arr += arr
        rho_ana, wq_ana = erlang_c_wq(lam, mu, c)
        rho_sim = lam / (c * mu)
        wq_sim = float(np.mean(all_waits)) if all_waits else 0.0
        # Practical utilization: served tasks * 1/mu / (c * sim_time * n_reps)
        rho_sim_emp = (len(all_waits) / (c * mu)) / (sim_time * n_reps)
        err_pct = abs(wq_sim - wq_ana) / max(wq_ana, 1e-9) * 100
        rho_err_pct = abs(rho_sim_emp - rho_ana) / max(rho_ana, 1e-9) * 100
        rows.append({
            "lambda":         lam,
            "rho_analytic":   round(rho_ana, 4),
            "rho_sim":        round(rho_sim_emp, 4),
            "rho_err_pct":    round(rho_err_pct, 3),
            "Wq_analytic_s":  round(wq_ana, 4),
            "Wq_sim_s":       round(wq_sim, 4),
            "err_pct":        round(err_pct, 2),
            "n_total_waits":  len(all_waits),
            "elapsed_s":      round(time.time() - t0, 1),
        })
        print(f"  lam={lam}  rho_an={rho_ana:.4f}  rho_emp={rho_sim_emp:.4f}  "
              f"Wq_an={wq_ana:.5f}  Wq_sim={wq_sim:.5f}  err={err_pct:.2f}%",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "verification_mmc.csv", index=False, encoding="utf-8-sig")
    return df
