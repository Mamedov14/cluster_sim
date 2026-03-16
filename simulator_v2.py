"""
simulator_v2.py — Двухуровневый симулятор вычислительного кластера с контейнерной виртуализацией.

Архитектура (по Фунг, Богатырёв, 2025):
  Уровень 0: Балансировщик → N узлов (round_robin / least_loaded / random)
  Уровень 1: Каждый узел содержит C контейнеров с ОБЩЕЙ очередью (Shared Queue)
             Интенсивность обслуживания контейнеров зависит от числа активных
             контейнеров (нелинейно, как в табл. 1 статьи).

Стратегии отказов:
  Отказ УЗЛА:
    - down:    все C контейнеров недоступны (узел не принимает задачи)
    - degrade: S' = k * S для всех контейнеров узла

  Отказ КОНТЕЙНЕРА:
    - Задача в этом контейнере: drop_all (теряется)
    - Очередь узла: freeze (задачи ждут восстановления контейнера)
    - После MTTR контейнера — восстановление

Зависимости: simpy, numpy
"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Literal

BalancePolicy = Literal["round_robin", "least_loaded", "random"]
NodeFailureType = Literal["down", "degrade", "none"]


# ─────────────────────────────────────────
# Таблица интенсивности обслуживания
# (из статьи Фунг/Богатырёв, табл. 1)
# μ(n, m) — интенсивность при n контейнерах всего, m активных
# Упрощённая модель: μ(m) = μ_base / (1 + α*(C-m)/C)
# где α — коэффициент деградации из-за конкуренции за ресурсы
# ─────────────────────────────────────────

def container_service_rate(base_rate: float, n_total: int, m_active: int, alpha: float = 0.5) -> float:
    """
    Интенсивность обслуживания одного активного контейнера.
    При большем числе активных контейнеров — ниже из-за разделения ресурсов.
    base_rate: базовая интенсивность при одном активном контейнере
    n_total:   общее число контейнеров на узле
    m_active:  текущее число активных контейнеров
    alpha:     коэффициент снижения (0 = нет деградации, 1 = линейная)
    """
    if m_active <= 0:
        return 0.0
    # Нелинейное снижение как в эксперименте статьи
    rate = base_rate * (1.0 / (1.0 + alpha * (m_active - 1) / n_total))
    return max(rate, base_rate * 0.1)


# ─────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────

@dataclass
class SimConfigV2:
    # Кластер
    num_nodes:           int   = 3      # число узлов (N)
    containers_per_node: int   = 3      # контейнеров на узел (C) — оптимум по статье
    queue_capacity:      int   = 50     # ёмкость общей очереди узла

    # Входной поток
    arrival_rate:        float = 3.0    # λ (задач/с)
    base_service_rate:   float = 6.0    # базовая μ контейнера (задач/с)
    alpha_resource:      float = 0.5    # коэффициент конкуренции за ресурсы

    # Время симуляции
    sim_time:            float = 500.0

    # Политика балансировки (уровень 0)
    balance_policy:      BalancePolicy = "round_robin"

    # Отказы УЗЛА
    node_failure_type:   NodeFailureType = "down"
    node_mttf:           float = 0.0    # 0 = без отказов узла
    node_mttr:           float = 0.0
    node_degrade_factor: float = 2.0    # k для degrade узла

    # Отказы КОНТЕЙНЕРА
    container_mttf:      float = 0.0    # 0 = без отказов контейнера
    container_mttr:      float = 0.0    # MTTR контейнера

    seed:                int   = 42


@dataclass
class SimResultV2:
    throughput:          float
    mean_wait:           float
    wait_p95:            float
    mean_sojourn:        float
    mean_node_util:      float
    mean_container_util: float
    tasks_dropped:       int
    tasks_done:          int
    config:              SimConfigV2 = field(default_factory=SimConfigV2)


# ─────────────────────────────────────────
# Контейнер
# ─────────────────────────────────────────

class Container:
    """Один контейнер внутри узла."""

    def __init__(self, env: simpy.Environment, container_id: int, node: 'Node', rng: np.random.Generator):
        self.env          = env
        self.container_id = container_id
        self.node         = node
        self.rng          = rng

        self.state        = "up"     # "up" | "down"
        self.busy         = False
        self.busy_time    = 0.0

        # Запускаем вероятностные отказы если заданы
        if node.cfg.container_mttf > 0:
            env.process(self._failure_process())

    def utilization(self) -> float:
        return self.busy_time / max(self.env.now, 1e-9)

    def _failure_process(self):
        """Вероятностный отказ контейнера."""
        while True:
            up_time = self.rng.exponential(self.node.cfg.container_mttf)
            yield self.env.timeout(up_time)
            if self.env.now >= self.node.cfg.sim_time:
                break

            self.state = "down"
            # Текущая задача теряется — сигнализируем узлу
            self.node.on_container_failure(self)

            down_time = self.rng.exponential(self.node.cfg.container_mttr)
            yield self.env.timeout(down_time)

            if self.env.now < self.node.cfg.sim_time:
                self.state = "up"
                self.node.on_container_recovery()


# ─────────────────────────────────────────
# Узел (уровень 1: C контейнеров + общая очередь)
# ─────────────────────────────────────────

class Node:
    """
    Вычислительный узел с C контейнерами и общей очередью.
    Реализует модель ResourceSharingFiniteQueue из статьи Фунга.
    """

    def __init__(self, env: simpy.Environment, node_id: int, cfg: SimConfigV2, rng: np.random.Generator):
        self.env        = env
        self.node_id    = node_id
        self.cfg        = cfg
        self.rng        = rng

        # Контейнеры
        self.containers = [Container(env, i, self, rng) for i in range(cfg.containers_per_node)]

        # Общая очередь узла
        self.queue: list = []

        # Состояние узла
        self.node_state = "up"      # "up" | "down" | "degrade"

        # Флаг заморозки очереди (при отказе контейнера)
        self._queue_frozen     = False
        self._freeze_event     = env.event()

        # Статистика
        self.busy_time      = 0.0
        self._last_busy_t   = 0.0

        # Пробуждение обработчика
        self._work_event    = env.event()

        # Счётчик потерь
        self.dropped        = 0

        # Запускаем воркер и отказы узла
        env.process(self._worker())
        if cfg.node_mttf > 0:
            env.process(self._node_failure_process())

    # ── Публичный интерфейс ────────────────

    def accept(self, task) -> bool:
        """Попытаться принять задачу. False = отказано."""
        if self.node_state == "down":
            return False
        if len(self.queue) >= self.cfg.queue_capacity:
            self.dropped += 1
            return False
        self.queue.append(task)
        if not self._work_event.triggered:
            self._work_event.succeed()
        return True

    def queue_length(self) -> int:
        return len(self.queue)

    def active_containers(self) -> int:
        """Число работоспособных контейнеров."""
        return sum(1 for c in self.containers if c.state == "up" and self.node_state != "down")

    def utilization(self) -> float:
        return self.busy_time / max(self.env.now, 1e-9)

    def container_utilization(self) -> float:
        return float(np.mean([c.utilization() for c in self.containers]))

    # ── Отказы контейнера ─────────────────

    def on_container_failure(self, container: Container):
        """Вызывается при отказе контейнера."""
        # Текущая задача в этом контейнере теряется (drop_all для задачи)
        # Очередь узла замораживается
        if not self._queue_frozen:
            self._queue_frozen = True
            self._freeze_event = self.env.event()

    def on_container_recovery(self):
        """Вызывается при восстановлении контейнера."""
        self._queue_frozen = False
        if not self._freeze_event.triggered:
            self._freeze_event.succeed()
        # Будим воркер
        if not self._work_event.triggered:
            self._work_event.succeed()

    # ── Отказы узла ───────────────────────

    def _node_failure_process(self):
        while True:
            up_t = self.rng.exponential(self.cfg.node_mttf)
            yield self.env.timeout(up_t)
            if self.env.now >= self.cfg.sim_time:
                break

            self.node_state = self.cfg.node_failure_type

            down_t = self.rng.exponential(self.cfg.node_mttr)
            yield self.env.timeout(down_t)

            if self.env.now < self.cfg.sim_time:
                self.node_state = "up"
                if not self._work_event.triggered:
                    self._work_event.succeed()

    # ── Воркер ────────────────────────────

    def _worker(self):
        """Обслуживает задачи из общей очереди через доступные контейнеры."""
        while True:
            # Ждём задачи или разморозки
            while not self.queue or self.node_state == "down" or self._queue_frozen:
                self._work_event = self.env.event()
                yield self._work_event

            # Ждём свободного контейнера
            free = [c for c in self.containers if not c.busy and c.state == "up"]
            if not free:
                self._work_event = self.env.event()
                yield self._work_event
                continue

            task = self.queue.pop(0)
            container = free[0]

            # Запускаем обслуживание в контейнере
            self.env.process(self._serve(task, container))

    def _serve(self, task, container: Container):
        """Обслуживает одну задачу в контейнере."""
        task.start_time = self.env.now
        container.busy  = True

        self._last_busy_t = self.env.now

        # Рассчитываем интенсивность с учётом числа активных контейнеров
        m_active = self.active_containers()
        rate = container_service_rate(
            self.cfg.base_service_rate,
            self.cfg.containers_per_node,
            m_active,
            self.cfg.alpha_resource
        )

        # Применяем degrade узла
        if self.node_state == "degrade":
            rate /= self.cfg.node_degrade_factor

        svc_time = self.rng.exponential(1.0 / rate) if rate > 0 else 10.0

        yield self.env.timeout(svc_time)

        # Проверяем: контейнер мог упасть за это время
        if container.state == "down":
            task.finish_time = -2.0  # потеряна
            container.busy   = False
            return

        container.busy_time += self.env.now - self._last_busy_t
        container.busy       = False
        task.finish_time     = self.env.now
        self.busy_time      += svc_time

        # Будим воркер для следующей задачи
        if not self._work_event.triggered:
            self._work_event.succeed()


# ─────────────────────────────────────────
# Задача
# ─────────────────────────────────────────

@dataclass
class Task:
    task_id:      int
    arrival_time: float
    start_time:   float = -1.0
    finish_time:  float = -1.0


# ─────────────────────────────────────────
# Балансировщик (уровень 0)
# ─────────────────────────────────────────

class Dispatcher:
    def __init__(self, nodes: List[Node], policy: BalancePolicy, rng: np.random.Generator):
        self.nodes   = nodes
        self.policy  = policy
        self.rng     = rng
        self._rr_idx = 0
        self.dropped = 0

    def dispatch(self, task: Task) -> bool:
        node = self._select()
        if node is None:
            self.dropped += 1
            return False
        ok = node.accept(task)
        if not ok:
            self.dropped += 1
        return ok

    def _select(self) -> Optional[Node]:
        available = [n for n in self.nodes if n.node_state != "down"]
        if not available:
            return None
        if self.policy == "round_robin":
            for _ in range(len(self.nodes)):
                n = self.nodes[self._rr_idx % len(self.nodes)]
                self._rr_idx += 1
                if n.node_state != "down":
                    return n
            return None
        elif self.policy == "least_loaded":
            return min(available, key=lambda n: n.queue_length())
        elif self.policy == "random":
            return self.rng.choice(available)
        return available[0]


# ─────────────────────────────────────────
# Генератор задач
# ─────────────────────────────────────────

class TaskGenerator:
    def __init__(self, env, dispatcher, cfg, rng):
        self.env        = env
        self.dispatcher = dispatcher
        self.cfg        = cfg
        self.rng        = rng
        self.all_tasks  = []
        self._id        = 0

    def start(self):
        self.env.process(self._run())

    def _run(self):
        while True:
            iat = self.rng.exponential(1.0 / self.cfg.arrival_rate)
            yield self.env.timeout(iat)
            if self.env.now >= self.cfg.sim_time:
                break
            task = Task(task_id=self._id, arrival_time=self.env.now)
            self._id += 1
            self.all_tasks.append(task)
            self.dispatcher.dispatch(task)


# ─────────────────────────────────────────
# Запуск симуляции
# ─────────────────────────────────────────

def run_simulation_v2(cfg: SimConfigV2) -> SimResultV2:
    rng = np.random.default_rng(cfg.seed)
    env = simpy.Environment()

    nodes      = [Node(env, i, cfg, rng) for i in range(cfg.num_nodes)]
    dispatcher = Dispatcher(nodes, cfg.balance_policy, rng)
    generator  = TaskGenerator(env, dispatcher, cfg, rng)
    generator.start()

    env.run(until=cfg.sim_time)

    completed = [t for t in generator.all_tasks if t.finish_time > 0 and t.start_time >= 0]
    waits     = [t.start_time - t.arrival_time for t in completed]
    sojourns  = [t.finish_time - t.arrival_time for t in completed]

    all_dropped = dispatcher.dropped + sum(n.dropped for n in nodes)

    return SimResultV2(
        throughput          = len(completed) / cfg.sim_time,
        mean_wait           = float(np.mean(waits))           if waits    else 0.0,
        wait_p95            = float(np.percentile(waits, 95)) if waits    else 0.0,
        mean_sojourn        = float(np.mean(sojourns))        if sojourns else 0.0,
        mean_node_util      = float(np.mean([n.utilization() for n in nodes])),
        mean_container_util = float(np.mean([n.container_utilization() for n in nodes])),
        tasks_dropped       = all_dropped,
        tasks_done          = len(completed),
        config              = cfg,
    )


def run_replications_v2(cfg: SimConfigV2, n_reps: int = 10) -> dict:
    results = []
    for i in range(n_reps):
        c = SimConfigV2(**{**cfg.__dict__, "seed": cfg.seed + i})
        results.append(run_simulation_v2(c))

    metrics = ["throughput", "mean_wait", "wait_p95", "mean_sojourn",
               "mean_node_util", "mean_container_util", "tasks_dropped"]
    summary = {}
    for m in metrics:
        vals = [getattr(r, m) for r in results]
        summary[m]          = float(np.mean(vals))
        summary[m + "_std"] = float(np.std(vals))
    return summary
