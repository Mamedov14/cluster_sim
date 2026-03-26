"""
simulator_v3.py — Двухуровневый симулятор отказоустойчивости вычислительного кластера.

Архитектура (Фунг, Богатырёв, 2025 + расширения аспиранта):
  Уровень 0: Балансировщик нагрузки (SPOF) — единственная точка отказа
  Уровень 1: N рабочих узлов, каждый содержит C контейнеров (Shared Queue)

Ключевые особенности v3:
  - Каскадный цикл отказа узла/LB: UP → DEGRADE → DOWN
  - Единственный системный администратор (simpy.PriorityResource):
      LB имеет приоритет 0 (наивысший), узлы — приоритет 1
  - Бесконечные очереди (Q=∞) — позволяют наблюдать взрывной рост задержек
  - Интенсивность μ(n,m) берётся из экспериментальной таблицы [Фунг и др., 2025]
  - При падении LB новые задачи немедленно сбрасываются (потери D растут)
  - Контейнер ждёт восстановления узла перед собственным перезапуском

Запуск:
  pip install simpy numpy pandas matplotlib
  python simulator_v3.py

Результаты сохраняются в папку output_figures_v3/:
  - full_results_v3.csv  — итоговая таблица всех экспериментов
  - exp1_mttf_fail_*.png — 6 графиков: влияние MTTF_fail (3 кривые старения)
  - exp2_mttr_*.png      — 6 графиков: влияние MTTR восстановления
  - exp3_policy_*.png    — 6 графиков: сравнение политик балансировки
  - exp4_lambda_*.png    — 6 графиков: влияние интенсивности входящего потока

Авторы: Мамедов В.В., ИТМО, 2026
"""

import simpy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

# =====================================================================
# 1. КОНФИГУРАЦИЯ
# =====================================================================

BalancePolicy = Literal["round_robin", "least_loaded", "random"]
OUT_DIR = Path("output_figures_v3")
OUT_DIR.mkdir(exist_ok=True)

# Таблица интенсивности обслуживания μ(n, m) [с⁻¹]
# Источник: Фунг В.К., Богатырёв В.А., До М.К. — Вестник ВКИТ, 2025, №8
# n — всего контейнеров на узле, m — активных в данный момент
MU_TABLE = [
    [22.305],
    [22.248, 14.968],
    [22.242, 13.426, 11.788],
    [22.131, 12.836, 10.473,  9.611],
    [22.137, 12.452,  9.801,  8.667,  7.825],
    [22.097, 12.276,  9.410,  8.041,  7.294,  6.809],
    [22.073, 12.047,  8.999,  7.767,  6.762,  6.333,  5.981],
    [21.989, 11.997,  8.939,  7.472,  6.616,  5.890,  5.598,  5.208],
    [21.957, 11.806,  8.590,  7.175,  6.143,  5.744,  5.363,  4.973, 4.886],
    [21.930, 11.658,  8.641,  6.936,  6.159,  5.566,  5.095,  4.711, 4.439, 4.296],
]


def container_service_rate(base_rate: float, n_total: int, m_active: int) -> float:
    """Интенсивность обслуживания из таблицы μ(n, m), масштабированная к base_rate."""
    if m_active <= 0:
        return 0.0
    n = max(1, min(n_total, len(MU_TABLE)))
    m = max(1, min(m_active, len(MU_TABLE[n - 1])))
    table_val = MU_TABLE[n - 1][m - 1]
    table_base = MU_TABLE[n - 1][0]
    return table_val * (base_rate / table_base if table_base > 0 else 1.0)


@dataclass
class SimConfig:
    # Топология кластера
    num_nodes:           int   = 3       # число рабочих узлов N
    containers_per_node: int   = 3       # контейнеров на узел C

    # Очередь (бесконечная — для наблюдения взрывного роста задержек)
    queue_capacity:      int   = 999_999

    # Входной поток
    arrival_rate:        float = 16.0   # λ, задач/с
    base_service_rate:   float = 22.242 # μ₀ = μ(3,1), с⁻¹

    # Время прогона
    sim_time:            float = 5000.0  # секунд

    # Политика балансировки нагрузки
    balance_policy:      BalancePolicy = "least_loaded"

    # Параметры отказов узла (каскад UP→DEGRADE→DOWN)
    node_mttf_fail:      float = 40.0   # MTTF аппаратного отказа
    degrade_percent:     float = 0.8    # MTTF_degrade = mttf_fail * degrade_percent
    node_mttr:           float = 15.0   # среднее время восстановления узла
    node_degrade_factor: float = 2.0    # во сколько раз снижается μ при degrade

    # Параметры отказов контейнера
    container_mttf:      float = 50.0
    container_mttr:      float = 10.0

    # Параметры балансировщика (SPOF)
    lb_service_rate:     float = 50.0   # интенсивность маршрутизации LB
    lb_concurrency:      int   = 1      # параллельных маршрутизаций
    lb_queue_capacity:   int   = 999_999

    # Воспроизводимость
    seed:                int   = 42
    repair_capacity:     int   = 1      # один системный администратор


@dataclass
class Task:
    task_id:      int
    arrival_time: float
    start_time:   float = -1.0
    finish_time:  float = -1.0


# =====================================================================
# 2. КЛАССЫ СИМУЛЯТОРА
# =====================================================================

class Container:
    """Один виртуальный контейнер внутри узла."""

    def __init__(self, env, container_id, node, rng):
        self.env          = env
        self.container_id = container_id
        self.node         = node
        self.rng          = rng
        self.state        = "up"
        self.busy         = False
        self.busy_time    = 0.0
        self.fail_process = env.process(self._failure_process())

    def _failure_process(self):
        """Случайный цикл отказ/восстановление контейнера."""
        try:
            while True:
                # Работаем до случайного отказа
                yield self.env.timeout(self.rng.exponential(self.node.cfg.container_mttf))
                if self.env.now >= self.node.cfg.sim_time:
                    break

                self.state = "down"
                self.node.on_container_failure(self)

                # Восстанавливаемся (оркестратор — авто)
                yield self.env.timeout(self.rng.exponential(self.node.cfg.container_mttr))

                # Ждём, пока родительский узел не поднимется
                while self.node.node_state == "down":
                    yield self.env.timeout(1.0)

                if self.env.now < self.node.cfg.sim_time:
                    self.state = "up"
                    self.node.on_container_recovery()
        except simpy.Interrupt:
            pass  # прерван из-за краша узла


class Node:
    """
    Рабочий узел уровня 1.
    Реализует каскадный цикл: UP → DEGRADE → DOWN.
    Восстановление через единственного ремонтника (приоритет 1).
    """

    def __init__(self, env, node_id, cfg, rng, repairman):
        self.env        = env
        self.node_id    = node_id
        self.cfg        = cfg
        self.rng        = rng
        self.repairman  = repairman

        self.containers = [Container(env, i, self, rng) for i in range(cfg.containers_per_node)]
        self.queue                = []
        self.node_state           = "up"
        self._queue_frozen        = False
        self._freeze_event        = env.event()
        self.busy_time            = 0.0
        self._last_busy_t         = 0.0
        self.dropped              = 0
        self._work_event          = env.event()
        self.active_serve_processes = set()

        env.process(self._worker())
        env.process(self._node_cascade_failure())

    # ── Публичный интерфейс ──────────────────────────────────────────

    def accept(self, task) -> bool:
        """Принять задачу в очередь. False = узел недоступен."""
        if self.node_state == "down":
            return False
        if len(self.queue) >= self.cfg.queue_capacity:
            self.dropped += 1
            return False
        self.queue.append(task)
        if not self._work_event.triggered:
            self._work_event.succeed()
        return True

    def active_containers(self) -> int:
        return sum(1 for c in self.containers if c.state == "up" and self.node_state != "down")

    def utilization(self) -> float:
        return self.busy_time / max(self.env.now, 1e-9)

    def on_container_failure(self, c):
        """Заморозить очередь при отказе контейнера."""
        if not self._queue_frozen:
            self._queue_frozen  = True
            self._freeze_event  = self.env.event()

    def on_container_recovery(self):
        """Разморозить очередь, когда все контейнеры восстановлены."""
        if all(c.state == "up" for c in self.containers):
            self._queue_frozen = False
            if not self._freeze_event.triggered:
                self._freeze_event.succeed()
            if not self._work_event.triggered:
                self._work_event.succeed()

    # ── Каскадный отказ узла ─────────────────────────────────────────

    def _node_cascade_failure(self):
        """
        Цикл: ждём MTTF_degrade → переходим в DEGRADE →
              ждём оставшееся время → переходим в DOWN →
              запрашиваем ремонтника → восстанавливаемся.
        """
        while True:
            mttf_degrade = self.cfg.node_mttf_fail * self.cfg.degrade_percent
            yield self.env.timeout(self.rng.exponential(mttf_degrade))
            if self.env.now >= self.cfg.sim_time:
                break

            # Фаза деградации: узел медленнее, но работает
            self.node_state = "degrade"
            mttf_remain = self.cfg.node_mttf_fail - mttf_degrade
            yield self.env.timeout(self.rng.exponential(max(mttf_remain, 0.1)))
            if self.env.now >= self.cfg.sim_time:
                break

            # Фаза отказа: узел полностью выключен
            self.node_state = "down"
            self.dropped += len(self.queue)
            self.queue.clear()

            # Прерываем все текущие обслуживания
            for p in list(self.active_serve_processes):
                if p.is_alive:
                    p.interrupt("NODE_CRASH")
            self.active_serve_processes.clear()

            # Останавливаем все контейнеры
            for c in self.containers:
                if c.fail_process and c.fail_process.is_alive:
                    c.fail_process.interrupt("NODE_CRASH")
                c.state = "down"
                c.busy  = False

            # Запрашиваем ремонтника (приоритет 1 — обычный)
            with self.repairman.request(priority=1) as req:
                yield req
                yield self.env.timeout(self.rng.exponential(self.cfg.node_mttr))

            # Восстановление: поднимаем узел и все контейнеры
            if self.env.now < self.cfg.sim_time:
                self.node_state    = "up"
                self._queue_frozen = False
                for c in self.containers:
                    c.state        = "up"
                    c.fail_process = self.env.process(c._failure_process())
                if not self._work_event.triggered:
                    self._work_event.succeed()

    # ── Воркер (обработчик очереди) ──────────────────────────────────

    def _worker(self):
        while True:
            while not self.queue or self.node_state == "down" or self._queue_frozen:
                self._work_event = self.env.event()
                yield self._work_event

            free = [c for c in self.containers if not c.busy and c.state == "up"]
            if not free:
                self._work_event = self.env.event()
                yield self._work_event
                continue

            task      = self.queue.pop(0)
            container = free[0]
            p = self.env.process(self._serve(task, container))
            self.active_serve_processes.add(p)

    def _serve(self, task, container):
        task.start_time = self.env.now
        container.busy  = True
        start_t         = self.env.now

        rate = container_service_rate(
            self.cfg.base_service_rate,
            self.cfg.containers_per_node,
            self.active_containers()
        )
        if self.node_state == "degrade":
            rate /= self.cfg.node_degrade_factor

        try:
            yield self.env.timeout(self.rng.exponential(1.0 / rate) if rate > 0 else 10.0)
            if container.state == "down" or self.node_state == "down":
                task.finish_time = -2.0
                self.dropped    += 1
            else:
                task.finish_time      = self.env.now
                container.busy_time  += self.env.now - start_t
                self.busy_time       += task.finish_time - task.start_time
        except simpy.Interrupt as e:
            if e.cause == "NODE_CRASH":
                task.finish_time = -2.0
                self.dropped    += 1
        finally:
            container.busy = False
            self.active_serve_processes.discard(self.env.active_process)
            if not self._work_event.triggered:
                self._work_event.succeed()


class LoadBalancerNode:
    """
    Балансировщик нагрузки — единственная точка отказа (SPOF).
    Каскад: UP → DEGRADE → DOWN.
    Восстановление: ремонтник с наивысшим приоритетом 0.
    При DOWN новые задачи немедленно сбрасываются.
    """

    def __init__(self, env, nodes, policy, cfg, rng, repairman):
        self.env        = env
        self.nodes      = nodes
        self.policy     = policy
        self.cfg        = cfg
        self.rng        = rng
        self.repairman  = repairman
        self._rr_idx    = 0
        self.dropped    = 0
        self.state      = "up"
        self.queue      = []
        self._work_event            = env.event()
        self.active_serve_processes = set()

        env.process(self._worker())
        env.process(self._failure_process())

    def dispatch(self, task):
        """Принять задачу от генератора."""
        if self.state == "down":
            # Нет питания — TCP-соединение физически невозможно
            self.dropped        += 1
            task.finish_time     = -2.0
            return
        if len(self.queue) >= self.cfg.lb_queue_capacity:
            self.dropped        += 1
            task.finish_time     = -2.0
            return
        self.queue.append(task)
        if not self._work_event.triggered:
            self._work_event.succeed()

    def _worker(self):
        while True:
            while (not self.queue or self.state == "down" or
                   len(self.active_serve_processes) >= self.cfg.lb_concurrency):
                self._work_event = self.env.event()
                yield self._work_event

            task = self.queue.pop(0)
            p = self.env.process(self._serve(task))
            self.active_serve_processes.add(p)

    def _serve(self, task):
        rate = self.cfg.lb_service_rate
        if self.state == "degrade":
            rate /= self.cfg.node_degrade_factor  # перегрев замедляет маршрутизацию
        try:
            yield self.env.timeout(self.rng.exponential(1.0 / rate))
            if self.state == "down":
                self.queue.insert(0, task)  # вернуть в очередь, ждать ремонта
            else:
                node = self._select()
                if not node or not node.accept(task):
                    self.dropped += 1
        except simpy.Interrupt as e:
            if e.cause == "LB_CRASH":
                self.queue.insert(0, task)
        finally:
            self.active_serve_processes.discard(self.env.active_process)
            if not self._work_event.triggered:
                self._work_event.succeed()

    def _select(self):
        """Выбрать рабочий узел согласно политике балансировки."""
        avail = [n for n in self.nodes if n.node_state != "down"]
        if not avail:
            return None
        if self.policy == "round_robin":
            n = avail[self._rr_idx % len(avail)]
            self._rr_idx += 1
            return n
        elif self.policy == "least_loaded":
            return min(avail, key=lambda n: len(n.queue))
        else:  # random
            return self.rng.choice(avail)

    def _failure_process(self):
        """Каскадный отказ балансировщика (приоритет ремонта = 0, наивысший)."""
        while True:
            mttf_degrade = self.cfg.node_mttf_fail * self.cfg.degrade_percent
            yield self.env.timeout(self.rng.exponential(mttf_degrade))
            if self.env.now >= self.cfg.sim_time:
                break

            self.state = "degrade"
            mttf_remain = self.cfg.node_mttf_fail - mttf_degrade
            yield self.env.timeout(self.rng.exponential(max(mttf_remain, 0.1)))
            if self.env.now >= self.cfg.sim_time:
                break

            self.state = "down"
            for p in list(self.active_serve_processes):
                if p.is_alive:
                    p.interrupt("LB_CRASH")
            self.active_serve_processes.clear()

            # Наивысший приоритет — блокирует ремонт узлов
            with self.repairman.request(priority=0) as req:
                yield req
                yield self.env.timeout(self.rng.exponential(self.cfg.node_mttr))

            if self.env.now < self.cfg.sim_time:
                self.state = "up"
                if not self._work_event.triggered:
                    self._work_event.succeed()


class TaskGenerator:
    """Пуассоновский генератор задач (межприходящие интервалы ~ Exp(1/λ))."""

    def __init__(self, env, dispatcher, cfg, rng):
        self.env        = env
        self.dispatcher = dispatcher
        self.cfg        = cfg
        self.rng        = rng
        self.all_tasks  = []
        self._id        = 0
        env.process(self._run())

    def _run(self):
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / self.cfg.arrival_rate))
            if self.env.now >= self.cfg.sim_time:
                break
            task = Task(task_id=self._id, arrival_time=self.env.now)
            self._id += 1
            self.all_tasks.append(task)
            self.dispatcher.dispatch(task)


# =====================================================================
# 3. ЗАПУСК ОДНОГО ПРОГОНА / СЕРИИ ПРОГОНОВ
# =====================================================================

def run_simulation(cfg: SimConfig) -> dict:
    """Один прогон симуляции, возвращает словарь метрик."""
    rng       = np.random.default_rng(cfg.seed)
    env       = simpy.Environment()
    repairman = simpy.PriorityResource(env, capacity=cfg.repair_capacity)

    nodes = [Node(env, i, cfg, rng, repairman) for i in range(cfg.num_nodes)]
    disp  = LoadBalancerNode(env, nodes, cfg.balance_policy, cfg, rng, repairman)
    gen   = TaskGenerator(env, disp, cfg, rng)

    env.run(until=cfg.sim_time)

    completed = [t for t in gen.all_tasks if t.finish_time > 0 and t.start_time >= 0]
    waits     = [t.start_time  - t.arrival_time for t in completed]
    sojourns  = [t.finish_time - t.arrival_time for t in completed]

    return {
        "throughput":    len(completed) / cfg.sim_time,
        "mean_wait":     float(np.mean(waits))           if waits     else 0.0,
        "wait_p95":      float(np.percentile(waits, 95)) if waits     else 0.0,
        "mean_sojourn":  float(np.mean(sojourns))        if sojourns  else 0.0,
        "mean_node_util":float(np.mean([n.utilization() for n in nodes])),
        "tasks_dropped": disp.dropped + sum(n.dropped for n in nodes),
        "total_tasks":   len(gen.all_tasks),
        "success_tasks": len(completed),
    }


def run_replications(cfg: SimConfig, n_reps: int = 10) -> dict:
    """
    10 независимых прогонов с разными seed.
    Возвращает средние и стандартные отклонения по всем метрикам.
    """
    results = [
        run_simulation(SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i}))
        for i in range(n_reps)
    ]
    metrics = ["throughput", "mean_wait", "wait_p95", "mean_sojourn",
               "mean_node_util", "tasks_dropped", "total_tasks", "success_tasks"]
    summary = {}
    for m in metrics:
        vals = [r[m] for r in results]
        summary[m]           = float(np.mean(vals))
        summary[m + "_std"]  = float(np.std(vals))
    return summary


def format_row(group: str, cfg: SimConfig, res: dict) -> dict:
    """Сформировать строку итоговой таблицы."""
    config_str = (f"λ={cfg.arrival_rate}, Pol={cfg.balance_policy}, "
                  f"Fail={cfg.node_mttf_fail}, Deg={cfg.degrade_percent}, "
                  f"MTTR={cfg.node_mttr}")
    return {
        "Группа":           group,
        "Конфигурация":     config_str,
        "N_gen":            f"{res['total_tasks']:.1f}",
        "N_succ":           f"{res['success_tasks']:.1f}",
        "X, зад/с":         f"{res['throughput']:.3f} ± {res['throughput_std']:.3f}",
        "W, с":             f"{res['mean_wait']:.3f} ± {res['mean_wait_std']:.3f}",
        "W0.95, с":         f"{res['wait_p95']:.3f} ± {res['wait_p95_std']:.3f}",
        "T, с":             f"{res['mean_sojourn']:.3f} ± {res['mean_sojourn_std']:.3f}",
        "Unode":            f"{res['mean_node_util']:.3f}",
        "D":                f"{res['tasks_dropped']:.1f} ± {res['tasks_dropped_std']:.1f}",
    }


# =====================================================================
# 4. ПЛАН ЭКСПЕРИМЕНТОВ (4 серии × 6 метрик = 24 графика)
# =====================================================================

def execute_all_experiments() -> dict:
    """
    Серия 1: Влияние MTTF_fail — 3 кривые старения (70%, 80%, 90%)
    Серия 2: Влияние MTTR восстановления
    Серия 3: Сравнение политик балансировки
    Серия 4: Влияние интенсивности входящего потока λ
    """
    print("Запуск симуляции v3 (каскад UP→DEGRADE→DOWN, PriorityRepairman)...\n")
    rows      = []
    plot_data = {"fail": [], "mttr": [], "bal": [], "lam": []}

    # --- Серия 1: MTTF_fail ---
    print("[1/4] Влияние MTTF_fail (кривые старения 70% / 80% / 90%)...")
    for mttf in [20, 30, 40, 50, 60, 80]:
        for pct in [0.7, 0.8, 0.9]:
            cfg = SimConfig(node_mttf_fail=float(mttf), degrade_percent=pct)
            res = run_replications(cfg)
            res["mttf_fail"]       = mttf
            res["degrade_percent"] = pct
            plot_data["fail"].append(res)
            rows.append(format_row("1. MTTF_fail", cfg, res))
    print("  Готово.")

    # --- Серия 2: MTTR ---
    print("[2/4] Влияние MTTR восстановления сервера...")
    for mttr in [5, 10, 15, 20, 25]:
        cfg = SimConfig(node_mttr=float(mttr), node_mttf_fail=40.0, degrade_percent=0.8)
        res = run_replications(cfg)
        res["mttr"] = mttr
        plot_data["mttr"].append(res)
        rows.append(format_row("2. MTTR", cfg, res))
    print("  Готово.")

    # --- Серия 3: Политики балансировки ---
    print("[3/4] Сравнение политик балансировки...")
    for pol in ["round_robin", "least_loaded", "random"]:
        cfg = SimConfig(balance_policy=pol, node_mttf_fail=40.0,
                        degrade_percent=0.8, node_mttr=15.0)
        res = run_replications(cfg)
        res["policy"] = pol
        plot_data["bal"].append(res)
        rows.append(format_row("3. Политика", cfg, res))
    print("  Готово.")

    # --- Серия 4: Интенсивность λ ---
    print("[4/4] Влияние интенсивности λ = [10 .. 30] задач/с...")
    for lam in [10.0, 14.0, 18.0, 22.0, 26.0, 30.0]:
        cfg = SimConfig(arrival_rate=lam, node_mttf_fail=40.0,
                        degrade_percent=0.8, balance_policy="least_loaded")
        res = run_replications(cfg)
        res["lambda"] = lam
        plot_data["lam"].append(res)
        rows.append(format_row("4. λ", cfg, res))
    print("  Готово.\n")

    # Сохраняем сводную таблицу
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "full_results_v3.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Таблица сохранена: {csv_path}")
    print(df.to_string(index=False))

    return plot_data


# =====================================================================
# 5. ГЕНЕРАЦИЯ 24 ГРАФИКОВ
# =====================================================================

METRICS = [
    ("throughput",    "throughput_std",    "X (задач/с)",    "Пропускная способность"),
    ("mean_wait",     "mean_wait_std",     "W (с)",          "Среднее время ожидания"),
    ("wait_p95",      "wait_p95_std",      "W₀.₉₅ (с)",     "95-й процентиль ожидания"),
    ("mean_sojourn",  "mean_sojourn_std",  "T (с)",          "Среднее время пребывания"),
    ("mean_node_util","mean_node_util_std","U_node",          "Утилизация рабочих узлов"),
    ("tasks_dropped", "tasks_dropped_std", "D (потери)",      "Сброшенные задачи"),
]


def _save_fig(filename: str):
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранён: {filename}")


def plot_mttf_series(df, metric, metric_err, y_label, title, subtitle, filename):
    """Серия 1: три кривые по degrade_percent."""
    plt.figure(figsize=(7, 4.5))
    colors = {0.7: "#E63946", 0.8: "#0077B6", 0.9: "#2A9D8F"}
    labels = {0.7: "Degrade 70%", 0.8: "Degrade 80%", 0.9: "Degrade 90%"}
    for pct in [0.7, 0.8, 0.9]:
        sub = df[df["degrade_percent"] == pct]
        plt.errorbar(sub["mttf_fail"], sub[metric], yerr=sub[metric_err],
                     color=colors[pct], marker="o", lw=2, capsize=4, label=labels[pct])
    plt.xlabel("MTTF_fail (с)")
    plt.ylabel(y_label)
    plt.title(title, pad=18)
    plt.suptitle(f"Конфиг: {subtitle}", fontsize=9, color="gray", y=0.94)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.45)
    _save_fig(filename)


def plot_line(x, y, y_err, x_label, y_label, title, subtitle, filename):
    """Серии 2 и 4: линейный график с полосами ошибок."""
    plt.figure(figsize=(7, 4.5))
    plt.errorbar(x, y, yerr=y_err, color="#0077B6", marker="o", lw=2, capsize=4)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=18)
    plt.suptitle(f"Конфиг: {subtitle}", fontsize=9, color="gray", y=0.94)
    plt.grid(True, ls="--", alpha=0.45)
    _save_fig(filename)


def plot_bar(labels, values, x_label, y_label, title, subtitle, filename):
    """Серия 3: столбчатый график по политикам."""
    colors = ["#0077B6", "#00897B", "#F4A261"]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, values, color=colors[:len(labels)], edgecolor="white")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, pad=18)
    plt.suptitle(f"Конфиг: {subtitle}", fontsize=9, color="gray", y=0.94)
    plt.grid(True, ls="--", alpha=0.45, axis="y")
    _save_fig(filename)


def generate_plots(data: dict):
    matplotlib.use("Agg")
    print("\nГенерация 24 графиков...")

    cfg_fail = "λ=16, Pol=LL, MTTR=15, Q=∞"
    cfg_mttr = "λ=16, Pol=LL, Fail=40, Deg=80%, Q=∞"
    cfg_bal  = "λ=16, Fail=40, Deg=80%, MTTR=15, Q=∞"
    cfg_lam  = "Pol=LL, Fail=40, Deg=80%, MTTR=15, Q=∞"

    df_f = pd.DataFrame(data["fail"])
    df_m = pd.DataFrame(data["mttr"])
    df_b = pd.DataFrame(data["bal"])
    df_l = pd.DataFrame(data["lam"])

    for m_val, m_err, y_lab, title_prefix in METRICS:
        # Серия 1 — MTTF_fail
        plot_mttf_series(
            df_f, m_val, m_err, y_lab,
            f"{title_prefix} vs MTTF_fail",
            cfg_fail,
            f"exp1_mttf_{m_val}.png"
        )
        # Серия 2 — MTTR
        plot_line(
            df_m["mttr"], df_m[m_val], df_m[m_err],
            "MTTR сервера (с)", y_lab,
            f"{title_prefix} vs MTTR",
            cfg_mttr,
            f"exp2_mttr_{m_val}.png"
        )
        # Серия 3 — Политики
        plot_bar(
            df_b["policy"].tolist(), df_b[m_val].tolist(),
            "Политика балансировки", y_lab,
            f"{title_prefix} по политикам",
            cfg_bal,
            f"exp3_policy_{m_val}.png"
        )
        # Серия 4 — λ
        plot_line(
            df_l["lambda"], df_l[m_val], df_l[m_err],
            "Интенсивность λ (задач/с)", y_lab,
            f"{title_prefix} vs λ",
            cfg_lam,
            f"exp4_lambda_{m_val}.png"
        )

    print(f"\n✅ 24 графика сохранены в папку '{OUT_DIR}/'")


# =====================================================================
# ТОЧКА ВХОДА
# =====================================================================

if __name__ == "__main__":
    plot_data = execute_all_experiments()
    generate_plots(plot_data)