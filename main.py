"""
main.py — Точка входа. Запускает все эксперименты и ML-модуль.

Использование:
    python main.py            # полный запуск
    python main.py --sim      # только симуляция и графики
    python main.py --ml       # только ML-модуль
    python main.py --quick    # быстрый тест (мало прогонов)
"""

import sys
import time

def check_dependencies():
    missing = []
    try:
        import simpy
    except ImportError:
        missing.append("simpy")
    try:
        import sklearn
    except ImportError:
        missing.append("scikit-learn")

    if missing:
        print("❌ Не хватает зависимостей. Установите:")
        print(f"   pip install {' '.join(missing)}")
        print("\nПолный список зависимостей:")
        print("   pip install simpy numpy pandas matplotlib scikit-learn")
        sys.exit(1)


def run_sim(quick=False):
    from experiments import run_all
    from plots import plot_all
    import experiments

    if quick:
        experiments.N_REPS = 3
        print("⚡ Быстрый режим: 3 прогона на конфигурацию\n")
    else:
        experiments.N_REPS = 10

    t0 = time.time()
    results = run_all()
    print(f"\nВремя симуляции: {time.time()-t0:.1f}с")

    plot_all(results)
    return results


def run_ml(quick=False):
    from ml_module import run_ml_pipeline

    n = 30 if quick else 120
    if quick:
        print("⚡ Быстрый режим: 30 конфигураций для ML\n")

    t0 = time.time()
    clf, le, df = run_ml_pipeline(n_configs=n)
    print(f"\nВремя ML-модуля: {time.time()-t0:.1f}с")
    return clf, le, df


if __name__ == "__main__":
    check_dependencies()

    args = sys.argv[1:]
    quick = "--quick" in args

    print("=" * 55)
    print("  Симулятор отказоустойчивости вычислительного кластера")
    print("  Мамедов Вагиф Вагифович · ИТМО · 2026")
    print("=" * 55)
    print()

    if "--ml" in args:
        run_ml(quick)

    elif "--sim" in args:
        run_sim(quick)

    else:
        # Полный запуск
        print("▶ Шаг 1/2: Симуляция и эксперименты")
        run_sim(quick)

        print("\n▶ Шаг 2/2: ML-модуль")
        run_ml(quick)

    print("\n✅ Готово! Результаты в папке output_figures/")
