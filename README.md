# Streaming RL: recurrent agents and memory benchmarks

Исследовательский код для сравнения потоковых и батчевых алгоритмов обучения
с подкреплением в задачах, где агенту требуется долговременная память.
Репозиторий объединяет реализации PPO, Streaming Actor-Critic, e-prop и
оконного TBPTT, конфигурации воспроизводимых экспериментов и тесты корректности
рекуррентных градиентов.

## Возможности

- агенты `ppo`, `stream_ac`, `stream_eprop` и `stream_tbptt`;
- Vanilla RNN, GRU, LSTM и RTU/RTRL-блоки;
- symmetric, random и adaptive feedback для e-prop;
- замороженная или предобученная state-space memory с обучаемыми readout-слоями;
- auxiliary-cue objective и варианты без memory readout;
- одиночные запуски, сетки гиперпараметров, несколько seed и сценарии смены среды;
- контрольные среды Passive/Active T-Maze, forced Passive T-Maze, Delayed Cue,
  lifelong CT-graph и задачи POPGym;
- логирование, checkpointing, оценка замороженной политики и аналитические
  скрипты для исследования скрытых представлений.

## Структура

```text
stream_rl/
├── main.py                         # одиночный запуск
├── pyproject.toml                  # зависимости проекта
└── stream_rl/
    ├── experiments/
    │   ├── configs/
    │   │   ├── agents/             # алгоритм и архитектура
    │   │   ├── envs/               # среда и её параметры
    │   │   ├── runner/             # бюджет, логирование и ссылки на конфиги
    │   │   ├── experiments/        # сетки и многосидовые кампании
    │   │   └── scenario/           # последовательные фазы эксперимента
    │   └── runners/                # исполнители одиночных и пакетных запусков
    ├── src/
    │   ├── agents/                 # PPO/StreamAC/e-prop/TBPTT
    │   ├── env/                    # JAX-среды и обёртки
    │   ├── models/                 # рекуррентные и memory-блоки
    │   └── utils/
    └── test/                       # проверки алгоритмов и сред
```

## Установка

Требуется Python 3.12. Рекомендуемый менеджер окружения —
[uv](https://docs.astral.sh/uv/).

```bash
cd stream_rl
uv sync
```

Файл `uv.lock` фиксирует версии зависимостей. Пакеты `memorax` и `lox`
подключаются из закреплённых Git-ревизий. Для GPU следует отдельно проверить
совместимость установленного JAX с CUDA на целевой машине.

## Быстрый старт

Одиночный запуск по готовому runner-конфигу:

```bash
cd stream_rl
uv run python main.py \
  --config stream_rl/experiments/configs/runner/tmaze_passive_stream_rtu_rtrl_20k.yaml \
  --seed 0
```

Runner-конфиг связывает три уровня настроек:

```yaml
agent_config: ../agents/stream_ac_rtu_rtrl.yaml
env_config: ../envs/tmaze_passive.yaml
scenario_config: null

experiment_name: autoresearch_memory_classic_tmaze
log_root: logs/autoresearch_memory_classic_tmaze
total_timesteps: 20000
log_every: 4000
step_chunk: 1000
checkpoint_every: -1
eval_every: -1
```

Относительные пути разрешаются от файла runner-конфига. Результаты записываются
в указанный `log_root`; сгенерированные логи и checkpoints намеренно не
версионируются.

## Пакетные эксперименты

Meta-конфиги из `experiments/configs/experiments/` задают seed, переопределения,
сетки параметров и последовательный либо параллельный режим запуска.

Сначала рекомендуется проверить материализацию команд:

```bash
cd stream_rl
uv run python -m stream_rl.experiments.runners.runner \
  --meta-config stream_rl/experiments/configs/experiments/ctgraph_lifelong_pilot.yaml \
  --dry-run
```

Для фактического запуска удалите `--dry-run`.

## Семантика конфигов

| Каталог | Назначение |
|---|---|
| `configs/agents` | алгоритм, оптимизация, recurrent cell, размер состояния и memory-режим |
| `configs/envs` | namespace среды, `env_id` и аргументы конструктора |
| `configs/runner` | бюджет, интервалы логирования/оценки/checkpoint и ссылки на остальные конфиги |
| `configs/experiments` | наборы запусков, seed, grid/overrides и параллелизм |
| `configs/scenario` | фазы curriculum, заморозка параметров и замена среды |
| `configs/campaigns` | полностью материализованные конфиги воспроизводимых кампаний |

Поддерживаемые пространства задач:

- `tmaze`: base, active, passive и forced-passive варианты;
- `delayed_cue`: разреженная, плотная и шумная отложенная подсказка;
- `ctgraph`: стационарные и lifelong-варианты со сменой целевой ветви;
- `popgym`: memory-бенчмарки, включая Tuple observations и MultiDiscrete actions;
- среды из `memorax` через соответствующий namespace.

## Проверки

```bash
cd stream_rl
uv run pytest stream_rl/test
```

Набор тестов проверяет:

- локальные eligibility traces RNN/GRU/LSTM относительно JAX autodiff;
- сброс recurrent state и traces между эпизодами;
- режимы feedback и воспроизводимость `feedback_seed`;
- RTU/RTRL и frozen-memory архитектуры;
- T-Maze, Delayed Cue, CT-graph и POPGym wrappers;
- совместимость scan-метаданных `lox` с текущим JAX.

## Воспроизводимость

Для публикуемого результата сохраняйте runner-, agent- и env-конфиги вместе с
seed и Git commit. Исполнитель автоматически материализует конфиги в каталог
запуска. Сравнение алгоритмов следует проводить при одинаковом числе переходов,
нескольких seed и заранее заданном критерии агрегации.

## Текущий исследовательский статус

Реализации и диагностические тесты предназначены для исследовательских
экспериментов. Короткие пилоты на T-Maze и POPGym не подтверждают устойчивого
преимущества streaming/e-prop над контрольными алгоритмами; выводы требуют
полных бюджетов, 5–10 seed и доверительных интервалов.

## Лицензия

Отдельный файл лицензии в репозитории пока отсутствует. До его добавления код
не следует считать открыто лицензированным для повторного распространения.
