"""
experiments.py — Серии экспериментов над симулятором кластера.

Эксперименты:
  1. Базовый режим (без отказов)
  2. Влияние MTTF на производительность
  3. Сравнение политик балансировки
  4. Сравнение типов отказов: down vs degrade
  5. Сравнение стратегий обработки: freeze / drop_queue / drop_all

Каждый эксперимент возвращает DataFrame с результатами.
"""

import pandas as pd
from simulator import SimConfig, FailureEvent, run_replications

N_REPS = 10   # количество прогонов для каждой конфигурации


# ─────────────────────────────────────────
# 1. Базовая конфигурация (без отказов)
# ─────────────────────────────────────────

def exp_baseline() -> pd.DataFrame:
    """Кластер без отказов — точка отсчёта."""
    cfg = SimConfig(
        num_nodes      = 3,
        arrival_rate   = 3.0,
        service_rate   = 1.5,
        sim_time       = 500.0,
        balance_policy = "round_robin",
    )
    res = run_replications(cfg, N_REPS)
    res["label"] = "baseline (no failures)"
    return pd.DataFrame([res])


# ─────────────────────────────────────────
# 2. Влияние MTTF
# ─────────────────────────────────────────

def exp_mttf_impact() -> pd.DataFrame:
    """Как меняются метрики при разном MTTF (MTTR фиксирован = 12)."""
    mttf_values = [10, 20, 35, 40, 60, 80]
    rows = []

    for mttf in mttf_values:
        cfg = SimConfig(
            num_nodes      = 3,
            arrival_rate   = 3.0,
            service_rate   = 1.5,
            sim_time       = 500.0,
            balance_policy = "round_robin",
            failure_type   = "down",
            on_fail_policy = "freeze",
            mttf           = mttf,
            mttr           = 12.0,
        )
        res = run_replications(cfg, N_REPS)
        res["mttf"]  = mttf
        res["mttr"]  = 12.0
        res["label"] = f"MTTF={mttf}"
        rows.append(res)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 3. Сравнение политик балансировки
# ─────────────────────────────────────────

def exp_balance_policies() -> pd.DataFrame:
    """round_robin vs least_loaded vs random при одинаковых отказах."""
    policies = ["round_robin", "least_loaded", "random"]
    rows = []

    for policy in policies:
        cfg = SimConfig(
            num_nodes      = 3,
            arrival_rate   = 3.0,
            service_rate   = 1.5,
            sim_time       = 500.0,
            balance_policy = policy,
            failure_type   = "down",
            on_fail_policy = "freeze",
            mttf           = 35.0,
            mttr           = 15.0,
        )
        res = run_replications(cfg, N_REPS)
        res["policy"] = policy
        res["label"]  = policy
        rows.append(res)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 4. down vs degrade
# ─────────────────────────────────────────

def exp_failure_types() -> pd.DataFrame:
    """Сравнение полного отказа и деградации производительности."""
    rows = []

    for ftype in ["down", "degrade"]:
        cfg = SimConfig(
            num_nodes      = 3,
            arrival_rate   = 3.0,
            service_rate   = 1.5,
            sim_time       = 500.0,
            balance_policy = "round_robin",
            failure_type   = ftype,
            on_fail_policy = "freeze",
            mttf           = 35.0,
            mttr           = 15.0,
            degrade_factor = 3.0,
        )
        res = run_replications(cfg, N_REPS)
        res["failure_type"] = ftype
        res["label"]        = ftype
        rows.append(res)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 5. Стратегии обработки отказов
# ─────────────────────────────────────────

def exp_on_fail_policies() -> pd.DataFrame:
    """freeze vs drop_queue vs drop_all."""
    rows = []

    for policy in ["freeze", "drop_queue", "drop_all"]:
        cfg = SimConfig(
            num_nodes      = 3,
            arrival_rate   = 3.0,
            service_rate   = 1.5,
            sim_time       = 500.0,
            balance_policy = "round_robin",
            failure_type   = "down",
            on_fail_policy = policy,
            mttf           = 35.0,
            mttr           = 15.0,
        )
        res = run_replications(cfg, N_REPS)
        res["on_fail_policy"] = policy
        res["label"]          = policy
        rows.append(res)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 6. Детерминированный сценарий (бонус)
# ─────────────────────────────────────────

def exp_deterministic() -> pd.DataFrame:
    """Плановое отключение узла 0 в момент t=100 на 30 секунд."""
    cfg = SimConfig(
        num_nodes      = 3,
        arrival_rate   = 3.0,
        service_rate   = 1.5,
        sim_time       = 500.0,
        balance_policy = "least_loaded",
        failure_type   = "down",
        on_fail_policy = "freeze",
        det_failures   = [FailureEvent(node_id=0, start_time=100.0, duration=30.0)],
    )
    res = run_replications(cfg, N_REPS)
    res["label"] = "deterministic (node0 down t=100..130)"
    return pd.DataFrame([res])


# ─────────────────────────────────────────
# Запуск всех экспериментов
# ─────────────────────────────────────────

def run_all() -> dict:
    print("Запуск экспериментов...")

    results = {}

    print("  [1/6] Базовый режим...")
    results["baseline"] = exp_baseline()

    print("  [2/6] Влияние MTTF...")
    results["mttf"] = exp_mttf_impact()

    print("  [3/6] Политики балансировки...")
    results["balance"] = exp_balance_policies()

    print("  [4/6] Типы отказов...")
    results["failure_type"] = exp_failure_types()

    print("  [5/6] Стратегии обработки...")
    results["on_fail"] = exp_on_fail_policies()

    print("  [6/6] Детерминированный сценарий...")
    results["deterministic"] = exp_deterministic()

    print("Готово.")
    return results


if __name__ == "__main__":
    results = run_all()

    # Сводная таблица
    cols = ["label", "throughput", "mean_wait", "wait_p95", "mean_util", "tasks_dropped"]

    all_rows = []
    for key, df in results.items():
        for _, row in df.iterrows():
            all_rows.append({c: row.get(c, "—") for c in cols})

    summary = pd.DataFrame(all_rows)
    print("\n=== Сводная таблица ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x))
