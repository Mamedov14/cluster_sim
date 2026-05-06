"""
Быстрый прогон экспериментов с N=3, C=3 — для перегенерации графиков диссертации.
Использует n_reps=5 (вместо 10) для скорости. Результаты статистически приемлемы.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v3 import (
    SimConfig, run_replications, OUT_DIR, generate_plots, format_row
)
import pandas as pd

NREPS = 3  # компромисс: качество vs скорость
SIM_TIME = 3000.0


def run_series_mttf():
    print("[1/4] MTTF_fail × 3 кривые деградации...")
    rows = []
    for mttf in [2500, 3000, 3500, 4000, 4500]:
        for pct in [0.7, 0.8, 0.9]:
            t0 = time.time()
            cfg = SimConfig(node_mttf_fail=float(mttf), degrade_percent=pct,
                            sim_time=SIM_TIME)
            res = run_replications(cfg, n_reps=NREPS)
            res["mttf_fail"]       = mttf
            res["degrade_percent"] = pct
            rows.append(res)
            print(f"  mttf={mttf}, deg={pct}: {time.time()-t0:.1f}с — "
                  f"X={res['throughput']:.2f}, W95={res['wait_p95']:.3f}, D={res['tasks_dropped']:.0f}")
    return rows


def run_series_mttr():
    print("[2/4] MTTR...")
    rows = []
    for mttr in [5, 10, 15, 20, 25]:
        cfg = SimConfig(node_mttr=float(mttr), node_mttf_fail=3500.0,
                        degrade_percent=0.8, sim_time=SIM_TIME)
        res = run_replications(cfg, n_reps=NREPS)
        res["mttr"] = mttr
        rows.append(res)
        print(f"  mttr={mttr}: D={res['tasks_dropped']:.0f}, W95={res['wait_p95']:.3f}")
    return rows


def run_series_pol():
    print("[3/4] Политики...")
    rows = []
    for pol in ["round_robin", "least_loaded", "random"]:
        cfg = SimConfig(balance_policy=pol, node_mttf_fail=3500.0,
                        degrade_percent=0.8, node_mttr=15.0, sim_time=SIM_TIME)
        res = run_replications(cfg, n_reps=NREPS)
        res["policy"] = pol
        rows.append(res)
        print(f"  {pol}: W95={res['wait_p95']:.3f}, D={res['tasks_dropped']:.0f}")
    return rows


def run_series_lam():
    print("[4/4] Лямбда...")
    rows = []
    for lam in [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]:
        cfg = SimConfig(arrival_rate=lam, node_mttf_fail=3500.0,
                        degrade_percent=0.8, balance_policy="least_loaded",
                        sim_time=SIM_TIME)
        res = run_replications(cfg, n_reps=NREPS)
        res["lambda"] = lam
        rows.append(res)
        print(f"  λ={lam}: W={res['mean_wait']:.3f}, W95={res['wait_p95']:.3f}, X={res['throughput']:.2f}")
    return rows


if __name__ == "__main__":
    series = sys.argv[1] if len(sys.argv) > 1 else "all"
    data = {}
    t0 = time.time()
    if series in ("mttf", "all"):
        data["fail"] = run_series_mttf()
    if series in ("mttr", "all"):
        data["mttr"] = run_series_mttr()
    if series in ("pol", "all"):
        data["bal"] = run_series_pol()
    if series in ("lam", "all"):
        data["lam"] = run_series_lam()

    print(f"\nВсего: {time.time()-t0:.1f}с")
    if series == "all":
        # Сохранить в CSV каждую серию
        for k in data:
            df = pd.DataFrame(data[k])
            df.to_csv(OUT_DIR / f"series_{k}.csv", index=False, encoding="utf-8-sig")
        # генерируем графики
        generate_plots(data)
    else:
        df = pd.DataFrame(data[list(data.keys())[0]])
        df.to_csv(OUT_DIR / f"series_{series}.csv", index=False, encoding="utf-8-sig")
        print(df.to_string(index=False))
