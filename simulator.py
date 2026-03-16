"""
simulator.py — Ядро дискретно-событийного симулятора вычислительного кластера.

Поддерживает:
  - Типы отказов: down (полный), degrade (деградация производительности)
  - Режимы отказов: вероятностный (MTTF/MTTR) и детерминированный (расписание)
  - Стратегии обработки задач при отказе: freeze, drop_queue, drop_all
  - Политики балансировки: round_robin, least_loaded, random
  - Метрики QoS: пропускная способность, время ожидания, P95, утилизация, потери

Зависимости: simpy, numpy
"""

import simpy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal
from enum import Enum


# ─────────────────────────────────────────
# Типы и константы
# ─────────────────────────────────────────

FailureType    = Literal["down", "degrade"]
OnFailPolicy   = Literal["freeze", "drop_queue", "drop_all"]
BalancePolicy  = Literal["round_robin", "least_loaded", "random"]


@dataclass
class Task:
    """Одна задача в системе."""
    task_id:      int
    arrival_time: float
    service_time: float          # базовое время обслуживания
    start_time:   float = -1.0   # момент начала обслуживания
    finish_time:  float = -1.0   # момент завершения


@dataclass
class FailureEvent:
    """Детерминированный сценарий отказа."""
    node_id:      int
    start_time:   float   # когда наступает отказ
    duration:     float   # сколько длится


@dataclass
class SimConfig:
    """Все параметры одного прогона симуляции."""
    # Кластер
    num_nodes:        int   = 3
    queue_capacity:   int   = 50     # макс. задач в очереди узла

    # Входной поток (пуассоновский)
    arrival_rate:     float = 3.0    # задач/сек (λ)
    service_rate:     float = 1.5    # задач/сек на узел (μ)

    # Время симуляции
    sim_time:         float = 500.0

    # Политики
    balance_policy:   BalancePolicy  = "round_robin"
    failure_type:     FailureType    = "down"
    on_fail_policy:   OnFailPolicy   = "freeze"

    # Вероятностные отказы (0 = отключены)
    mttf:             float = 0.0    # среднее время наработки на отказ
    mttr:             float = 0.0    # среднее время восстановления

    # Детерминированные отказы (список FailureEvent)
    det_failures:     List[FailureEvent] = field(default_factory=list)

    # Параметр деградации
    degrade_factor:   float = 3.0    # во сколько раз замедляется узел

    # Воспроизводимость
    seed:             int   = 42


@dataclass
class SimResult:
    """Результаты одного прогона."""
    throughput:     float   # задач/сек
    mean_wait:      float   # среднее время ожидания
    wait_p95:       float   # 95-й процентиль ожидания
    mean_sojourn:   float   # среднее время пребывания в системе
    mean_util:      float   # средняя утилизация узлов
    tasks_dropped:  int     # потерянные задачи
    tasks_done:     int     # обслуженные задачи

    config:         SimConfig = field(default_factory=SimConfig)


# ─────────────────────────────────────────
# Узел кластера
# ─────────────────────────────────────────

class Node:
    """Один вычислительный узел."""

    def __init__(self, env: simpy.Environment, node_id: int, cfg: SimConfig, rng: np.random.Generator):
        self.env      = env
        self.node_id  = node_id
        self.cfg      = cfg
        self.rng      = rng

        # Очередь задач
        self.queue: List[Task] = []

        # Состояние
        self.state   = "up"           # "up" | "down" | "degrade"
        self.busy    = False

        # Статистика
        self.busy_time   = 0.0
        self.last_busy_start = 0.0

        # Для freeze: задача «на паузе»
        self._frozen_task:       Optional[Task]  = None
        self._frozen_remaining:  float           = 0.0
        self._frozen_pause_at:   float           = 0.0

        # Событие «появилась задача» (для пробуждения обработчика)
        self._task_event = env.event()

        # Запускаем процесс обслуживания
        env.process(self._worker())

    # ── публичный интерфейс ──────────────────

    def accept(self, task: Task) -> bool:
        """Попытаться поставить задачу в очередь. Возвращает False если отказано."""
        if self.state == "down":
            return False
        if len(self.queue) >= self.cfg.queue_capacity:
            return False
        self.queue.append(task)
        # будим обработчик если он ждёт
        if not self._task_event.triggered:
            self._task_event.succeed()
        return True

    def queue_length(self) -> int:
        return len(self.queue)

    def utilization(self) -> float:
        """Доля времени узла в состоянии 'занят'."""
        return self.busy_time / max(self.env.now, 1e-9)

    # ── отказы ──────────────────────────────

    def fail(self, failure_type: FailureType):
        """Перевести узел в состояние отказа."""
        if self.state != "up":
            return

        self.state = failure_type

        if failure_type == "down":
            self._handle_down_fail()
        # degrade — обработчик сам заметит изменение state

    def recover(self):
        """Восстановить узел."""
        prev_state = self.state
        self.state = "up"

        if prev_state == "down" and self.cfg.on_fail_policy == "freeze" and self._frozen_task:
            # Возобновляем замороженную задачу
            elapsed_frozen = self.env.now - self._frozen_pause_at
            # помещаем обратно в начало очереди с оставшимся временем
            task = self._frozen_task
            self._frozen_task = None
            remaining = self._frozen_remaining
            self.env.process(self._resume_frozen(task, remaining))

        # Будим обработчика
        if not self._task_event.triggered:
            self._task_event.succeed()

    # ── внутренняя логика ────────────────────

    def _handle_down_fail(self):
        policy = self.cfg.on_fail_policy

        if policy == "freeze":
            # Ничего не теряем; если сейчас что-то выполняется — это
            # обрабатывается в _worker через проверку state
            pass

        elif policy == "drop_queue":
            # Теряем всё из очереди, текущая задача — отдельно (считаем завершённой)
            self.queue.clear()

        elif policy == "drop_all":
            self.queue.clear()
            self._frozen_task = None

    def _worker(self):
        """Основной процесс узла: берёт задачи из очереди и обслуживает."""
        while True:
            # Ждём задачу
            while not self.queue or self.state == "down":
                self._task_event = self.env.event()
                yield self._task_event

            task = self.queue.pop(0)
            task.start_time = self.env.now

            # Время обслуживания с учётом деградации
            svc = task.service_time
            if self.state == "degrade":
                svc *= self.cfg.degrade_factor

            self.busy = True
            self.last_busy_start = self.env.now

            # Обслуживаем (с поддержкой freeze при down-отказе)
            yield from self._serve(task, svc)

            self.busy_time += self.env.now - self.last_busy_start
            self.busy = False

            task.finish_time = self.env.now

    def _serve(self, task: Task, remaining: float):
        """Обслуживает задачу, поддерживая паузу при down-отказе (freeze)."""
        while remaining > 0:
            start = self.env.now

            try:
                yield self.env.timeout(remaining)
                remaining = 0  # завершили

            except simpy.Interrupt:
                # Прерваны (не используется напрямую, но оставим для расширения)
                elapsed = self.env.now - start
                remaining -= elapsed

            # Проверяем состояние узла
            if self.state == "down":
                if self.cfg.on_fail_policy == "freeze":
                    # Запоминаем и ждём восстановления
                    self._frozen_task      = task
                    self._frozen_remaining = remaining
                    self._frozen_pause_at  = self.env.now
                    # Ждём пока узел не поднимется
                    while self.state == "down":
                        self._task_event = self.env.event()
                        yield self._task_event
                    self._frozen_task = None
                elif self.cfg.on_fail_policy in ("drop_queue", "drop_all"):
                    # Считаем задачу потерянной — прерываем обслуживание
                    task.finish_time = -2.0  # маркер потери
                    return

            elif self.state == "degrade":
                # Пересчитываем оставшееся время с коэффициентом
                elapsed = self.env.now - start
                remaining = (remaining - elapsed) * self.cfg.degrade_factor

    def _resume_frozen(self, task: Task, remaining: float):
        """Продолжить обслуживание размороженной задачи."""
        self.busy = True
        self.last_busy_start = self.env.now
        yield from self._serve(task, remaining)
        self.busy_time += self.env.now - self.last_busy_start
        self.busy = False
        task.finish_time = self.env.now


# ─────────────────────────────────────────
# Диспетчер (балансировщик нагрузки)
# ─────────────────────────────────────────

class Dispatcher:
    """Централизованный диспетчер задач."""

    def __init__(self, nodes: List[Node], policy: BalancePolicy, rng: np.random.Generator):
        self.nodes   = nodes
        self.policy  = policy
        self.rng     = rng
        self._rr_idx = 0
        self.dropped = 0

    def dispatch(self, task: Task) -> bool:
        """Направить задачу на узел. Возвращает False если задача потеряна."""
        node = self._select_node()
        if node is None:
            self.dropped += 1
            return False

        ok = node.accept(task)
        if not ok:
            self.dropped += 1
            return False
        return True

    def _select_node(self) -> Optional[Node]:
        available = [n for n in self.nodes if n.state != "down"]
        if not available:
            return None

        if self.policy == "round_robin":
            # Обходим по кругу, пропускаем недоступные
            for _ in range(len(self.nodes)):
                node = self.nodes[self._rr_idx % len(self.nodes)]
                self._rr_idx += 1
                if node.state != "down":
                    return node
            return None

        elif self.policy == "least_loaded":
            return min(available, key=lambda n: n.queue_length())

        elif self.policy == "random":
            return self.rng.choice(available)

        return available[0]


# ─────────────────────────────────────────
# Модуль отказов
# ─────────────────────────────────────────

class FailureModule:
    """Генерирует отказы и восстановления для узлов."""

    def __init__(self, env: simpy.Environment, nodes: List[Node], cfg: SimConfig, rng: np.random.Generator):
        self.env   = env
        self.nodes = nodes
        self.cfg   = cfg
        self.rng   = rng

    def start(self):
        """Запустить все процессы отказов."""
        # Вероятностные
        if self.cfg.mttf > 0 and self.cfg.mttr > 0:
            for node in self.nodes:
                self.env.process(self._probabilistic_failures(node))

        # Детерминированные
        for event in self.cfg.det_failures:
            self.env.process(self._deterministic_failure(event))

    def _probabilistic_failures(self, node: Node):
        """Чередует периоды работы и восстановления для одного узла."""
        while True:
            # Работаем MTTF времени
            up_time = self.rng.exponential(self.cfg.mttf)
            yield self.env.timeout(up_time)

            if self.env.now >= self.cfg.sim_time:
                break

            # Отказываем
            node.fail(self.cfg.failure_type)

            # Восстанавливаемся через MTTR
            down_time = self.rng.exponential(self.cfg.mttr)
            yield self.env.timeout(down_time)

            if self.env.now < self.cfg.sim_time:
                node.recover()

    def _deterministic_failure(self, event: FailureEvent):
        """Отказ по расписанию."""
        yield self.env.timeout(event.start_time)
        node = self.nodes[event.node_id]
        node.fail(self.cfg.failure_type)

        yield self.env.timeout(event.duration)
        node.recover()


# ─────────────────────────────────────────
# Генератор задач
# ─────────────────────────────────────────

class TaskGenerator:
    """Генерирует задачи по пуассоновскому закону."""

    def __init__(self, env: simpy.Environment, dispatcher: Dispatcher,
                 cfg: SimConfig, rng: np.random.Generator):
        self.env        = env
        self.dispatcher = dispatcher
        self.cfg        = cfg
        self.rng        = rng
        self.all_tasks: List[Task] = []
        self._task_id   = 0

    def start(self):
        self.env.process(self._generate())

    def _generate(self):
        while True:
            # Межприбытийное время (экспоненциальное)
            iat = self.rng.exponential(1.0 / self.cfg.arrival_rate)
            yield self.env.timeout(iat)

            if self.env.now >= self.cfg.sim_time:
                break

            svc = self.rng.exponential(1.0 / self.cfg.service_rate)
            task = Task(
                task_id      = self._task_id,
                arrival_time = self.env.now,
                service_time = svc,
            )
            self._task_id += 1
            self.all_tasks.append(task)
            self.dispatcher.dispatch(task)


# ─────────────────────────────────────────
# Запуск симуляции
# ─────────────────────────────────────────

def run_simulation(cfg: SimConfig) -> SimResult:
    """Запустить один прогон и вернуть результаты."""
    rng = np.random.default_rng(cfg.seed)
    env = simpy.Environment()

    # Создаём узлы
    nodes = [Node(env, i, cfg, rng) for i in range(cfg.num_nodes)]

    # Диспетчер
    dispatcher = Dispatcher(nodes, cfg.balance_policy, rng)

    # Модуль отказов
    failures = FailureModule(env, nodes, cfg, rng)
    failures.start()

    # Генератор задач
    generator = TaskGenerator(env, dispatcher, cfg, rng)
    generator.start()

    # Запуск
    env.run(until=cfg.sim_time)

    # ── Сбор метрик ──
    completed = [t for t in generator.all_tasks
                 if t.finish_time > 0 and t.start_time >= 0]

    wait_times    = [t.start_time - t.arrival_time for t in completed]
    sojourn_times = [t.finish_time - t.arrival_time for t in completed]

    throughput   = len(completed) / cfg.sim_time
    mean_wait    = float(np.mean(wait_times))    if wait_times    else 0.0
    wait_p95     = float(np.percentile(wait_times, 95)) if wait_times else 0.0
    mean_sojourn = float(np.mean(sojourn_times)) if sojourn_times else 0.0
    mean_util    = float(np.mean([n.utilization() for n in nodes]))

    return SimResult(
        throughput    = throughput,
        mean_wait     = mean_wait,
        wait_p95      = wait_p95,
        mean_sojourn  = mean_sojourn,
        mean_util     = mean_util,
        tasks_dropped = dispatcher.dropped,
        tasks_done    = len(completed),
        config        = cfg,
    )


def run_replications(cfg: SimConfig, n_reps: int = 10) -> dict:
    """
    Запустить n_reps независимых прогонов (разные seed) и вернуть
    средние значения и стандартные отклонения метрик.
    """
    results = []
    for i in range(n_reps):
        c = SimConfig(**{**cfg.__dict__, "seed": cfg.seed + i,
                         "det_failures": cfg.det_failures})
        results.append(run_simulation(c))

    metrics = ["throughput", "mean_wait", "wait_p95", "mean_sojourn",
               "mean_util", "tasks_dropped"]

    summary = {}
    for m in metrics:
        vals = [getattr(r, m) for r in results]
        summary[m]         = float(np.mean(vals))
        summary[m + "_std"] = float(np.std(vals))

    return summary
