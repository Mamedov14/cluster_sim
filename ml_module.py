"""
ml_module.py — ML-модуль для выбора оптимальной дисциплины реконфигурации.

Подход:
  1. Генерируем обучающие данные через симулятор (разные конфигурации)
  2. Определяем «оптимальную» стратегию для каждой конфигурации
     по критерию минимума суммарных потерь задач + взвешенного P95
  3. Обучаем классификатор (Random Forest)
  4. Оцениваем качество на тестовой выборке
  5. Строим графики важности признаков и матрицу ошибок

Признаки (features):
  - mttf, mttr          — надёжность узлов
  - arrival_rate        — интенсивность потока
  - service_rate        — скорость обслуживания
  - num_nodes           — число узлов
  - failure_type_enc    — тип отказа (0=down, 1=degrade)

Целевая переменная (target):
  - оптимальная стратегия on_fail_policy (freeze / drop_queue / drop_all)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder

from simulator import SimConfig, run_replications

OUT_DIR = Path("output_figures")
OUT_DIR.mkdir(exist_ok=True)

POLICIES    = ["freeze", "drop_queue", "drop_all"]
FAILURE_TYPES = ["down", "degrade"]

# Веса для целевой функции: важнее минимизировать потери задач
W_DROPPED = 1.0
W_P95     = 0.3   # нормализованный вес задержки


# ─────────────────────────────────────────
# 1. Генерация обучающих данных
# ─────────────────────────────────────────

def generate_dataset(n_configs: int = 120, seed: int = 0) -> pd.DataFrame:
    """
    Генерирует датасет: для каждой случайной конфигурации кластера
    прогоняет все три стратегии и определяет лучшую.
    """
    rng = np.random.default_rng(seed)
    rows = []

    print(f"  Генерация датасета ({n_configs} конфигураций × 3 стратегии)...")

    for i in range(n_configs):
        # Случайная конфигурация
        mttf         = float(rng.uniform(10, 100))
        mttr         = float(rng.uniform(5, 30))
        arrival_rate = float(rng.uniform(2.0, 5.0))
        service_rate = float(rng.uniform(1.0, 2.5))
        num_nodes    = int(rng.integers(2, 6))
        ftype        = rng.choice(FAILURE_TYPES)

        # Собираем метрики для каждой стратегии
        scores = {}
        metrics_per_policy = {}

        for policy in POLICIES:
            cfg = SimConfig(
                num_nodes      = num_nodes,
                arrival_rate   = arrival_rate,
                service_rate   = service_rate,
                sim_time       = 300.0,
                balance_policy = "round_robin",
                failure_type   = ftype,
                on_fail_policy = policy,
                mttf           = mttf,
                mttr           = mttr,
                seed           = seed + i * 100,
            )
            res = run_replications(cfg, n_reps=5)
            metrics_per_policy[policy] = res

            # Целевая функция: минимизируем
            score = W_DROPPED * res["tasks_dropped"] + W_P95 * res["wait_p95"]
            scores[policy] = score

        # Лучшая стратегия = минимальный score
        best_policy = min(scores, key=scores.get)

        rows.append({
            "mttf":             mttf,
            "mttr":             mttr,
            "arrival_rate":     arrival_rate,
            "service_rate":     service_rate,
            "num_nodes":        num_nodes,
            "failure_type":     ftype,
            "failure_type_enc": 0 if ftype == "down" else 1,
            "best_policy":      best_policy,
            # Метрики победившей стратегии
            "best_throughput":  metrics_per_policy[best_policy]["throughput"],
            "best_wait_p95":    metrics_per_policy[best_policy]["wait_p95"],
            "best_dropped":     metrics_per_policy[best_policy]["tasks_dropped"],
        })

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_configs} конфигураций обработано")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# 2. Обучение классификатора
# ─────────────────────────────────────────

FEATURES = ["mttf", "mttr", "arrival_rate", "service_rate",
            "num_nodes", "failure_type_enc"]

def train_classifier(df: pd.DataFrame):
    """Обучает Random Forest и возвращает модель + отчёт."""
    X = df[FEATURES].values
    y = df["best_policy"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.25, random_state=42, stratify=y_enc
    )

    clf = RandomForestClassifier(
        n_estimators = 100,
        max_depth    = None,
        random_state = 42,
        class_weight = "balanced",
    )
    clf.fit(X_train, y_train)

    # Кросс-валидация
    cv_scores = cross_val_score(clf, X, y_enc, cv=5, scoring="accuracy")

    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )

    print("\n=== Отчёт классификатора ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Кросс-валидация (5-fold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return clf, le, X_test, y_test, y_pred, report


# ─────────────────────────────────────────
# 3. Визуализация ML
# ─────────────────────────────────────────

def plot_feature_importance(clf, le):
    fig, ax = plt.subplots(figsize=(8, 4))

    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names_sorted = [FEATURES[i] for i in indices]

    colors = ["#0077B6" if imp > np.mean(importances) else "#ADB5BD"
              for imp in importances[indices]]

    bars = ax.barh(feature_names_sorted[::-1], importances[indices][::-1],
                   color=colors[::-1], edgecolor="white")

    ax.set_xlabel("Важность признака (Gini)")
    ax.set_title("Важность признаков — Random Forest\n(выбор дисциплины реконфигурации)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, imp in zip(bars, importances[indices][::-1]):
        ax.text(imp + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{imp:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    path = OUT_DIR / "fig6_feature_importance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранён: {path}")


def plot_confusion_matrix(y_test, y_pred, le):
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")

    ax.set_title("Матрица ошибок классификатора\n(выбор оптимальной стратегии реконфигурации)")
    plt.tight_layout()

    path = OUT_DIR / "fig7_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранён: {path}")


def plot_policy_distribution(df: pd.DataFrame):
    """Распределение оптимальных стратегий в датасете."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # По всем конфигурациям
    counts = df["best_policy"].value_counts()
    colors = ["#0077B6", "#00B4D8", "#00897B"]
    axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Частота оптимальных стратегий\n(весь датасет)")
    axes[0].set_ylabel("Количество конфигураций")

    # По типу отказа
    ct = pd.crosstab(df["failure_type"], df["best_policy"])
    ct.plot(kind="bar", ax=axes[1], color=colors, edgecolor="white", rot=0)
    axes[1].set_title("Оптимальная стратегия по типу отказа")
    axes[1].set_xlabel("Тип отказа")
    axes[1].set_ylabel("Количество конфигураций")
    axes[1].legend(title="Стратегия")

    plt.tight_layout()
    path = OUT_DIR / "fig8_policy_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранён: {path}")


# ─────────────────────────────────────────
# 4. Интерфейс предсказания
# ─────────────────────────────────────────

def predict_best_policy(clf, le, mttf: float, mttr: float,
                         arrival_rate: float, service_rate: float,
                         num_nodes: int, failure_type: str) -> str:
    """
    Предсказать оптимальную дисциплину реконфигурации для заданных параметров.

    Пример:
        policy = predict_best_policy(clf, le, mttf=30, mttr=10,
                                     arrival_rate=3.5, service_rate=1.5,
                                     num_nodes=4, failure_type="down")
    """
    ftype_enc = 0 if failure_type == "down" else 1
    x = np.array([[mttf, mttr, arrival_rate, service_rate, num_nodes, ftype_enc]])
    pred_enc = clf.predict(x)[0]
    proba = clf.predict_proba(x)[0]

    policy = le.inverse_transform([pred_enc])[0]

    print(f"\nПредсказание:")
    print(f"  Параметры: MTTF={mttf}, MTTR={mttr}, λ={arrival_rate}, μ={service_rate}, "
          f"узлов={num_nodes}, тип={failure_type}")
    print(f"  Рекомендуемая стратегия: {policy}")
    for cls, p in zip(le.classes_, proba):
        print(f"    {cls}: {p:.3f}")

    return policy


# ─────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────

def run_ml_pipeline(n_configs: int = 120):
    print("=== ML-модуль: выбор дисциплины реконфигурации ===\n")

    print("1. Генерация обучающих данных...")
    df = generate_dataset(n_configs=n_configs)
    df.to_csv(OUT_DIR / "ml_dataset.csv", index=False)
    print(f"   Датасет: {len(df)} строк, сохранён в output_figures/ml_dataset.csv")

    print("\n2. Обучение классификатора...")
    clf, le, X_test, y_test, y_pred, report = train_classifier(df)

    print("\n3. Визуализация...")
    plot_feature_importance(clf, le)
    plot_confusion_matrix(y_test, y_pred, le)
    plot_policy_distribution(df)

    print("\n4. Примеры предсказаний:")
    predict_best_policy(clf, le, mttf=20, mttr=10, arrival_rate=3.0,
                        service_rate=1.5, num_nodes=3, failure_type="down")
    predict_best_policy(clf, le, mttf=80, mttr=5, arrival_rate=2.5,
                        service_rate=2.0, num_nodes=5, failure_type="degrade")

    return clf, le, df


if __name__ == "__main__":
    run_ml_pipeline(n_configs=120)
