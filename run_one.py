"""Запуск одной точки эксперимента, сохранение в инкрементальный CSV.
Использование:
    python run_one.py <series> <param>=<value> [<param>=<value> ...]
Пример:
    python run_one.py mttf node_mttf_fail=2500 degrade_percent=0.7
    python run_one.py pol balance_policy=least_loaded
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
import pandas as pd

from simulator_v3 import SimConfig, run_replications, OUT_DIR

NREPS = 10
SIM_TIME = 5000.0


def parse_kwargs(args):
    kw = {}
    for a in args:
        k, v = a.split("=", 1)
        if v.replace(".", "").replace("-", "").isdigit():
            kw[k] = float(v) if "." in v else int(v)
        elif v in ("True", "False"):
            kw[k] = v == "True"
        else:
            kw[k] = v
    return kw


if __name__ == "__main__":
    series = sys.argv[1]
    kwargs = parse_kwargs(sys.argv[2:])
    cfg = SimConfig(sim_time=SIM_TIME, **kwargs)
    t0 = time.time()
    res = run_replications(cfg, n_reps=NREPS)
    res.update(kwargs)
    res["elapsed"] = time.time() - t0
    res["series"]  = series

    # инкрементальный csv
    csv_path = OUT_DIR / f"series_{series}_v3.csv"
    df_new = pd.DataFrame([res])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # сводка
    print(json.dumps({k: v for k, v in res.items()
                      if not k.endswith("_std") and isinstance(v, (int, float, str))},
                     ensure_ascii=False, indent=2))
    print(f"elapsed={res['elapsed']:.1f}s  saved={csv_path}")
