# Pong RL — Deep Q-Learning with MC-Dropout Exploration

> Кастомное окружение Pong на `gymnasium` + агент на базе DQN с нестандартным механизмом exploration через MC-Dropout (UCB), предобученный имитацией эвристики и дообученный онлайн-RL.

<p align="center">
  <!-- TODO: вставь сюда GIF или скриншот игры -->
  <img src="docs/demo.gif" alt="Pong agent gameplay demo" width="600">
</p>

---

## 🇷🇺 Русский

### О проекте

Это не «ещё один DQN на Pong» — цель проекта была не просто решить задачу, а спроектировать и обкатать нестандартный подход к exploration в DQN: вместо классического ε-greedy агент использует **MC-Dropout как оценку эпистемической неопределённости** и выбирает действия по принципу **UCB** (upper confidence bound), балансируя между ожидаемой ценностью действия и уверенностью сети в этой оценке.

### Цели проекта

- Реализовать среду Pong «с нуля» на `gymnasium` — с нормализованными наблюдениями, конфигурируемой сложностью соперника и режимом рендеринга.
- Реализовать и обучить DQN-агента, дообучаемого поверх imitation-learning предобучения на эвристике.
- Исследовать **MC-Dropout как источник exploration** в RL (альтернатива ε-greedy / bootstrapped DQN) — на практике разобраться, где эта идея работает, а где даёт неожиданные побочные эффекты (нестабильность обучения, коллапс политики).
- Довести агента до обыгрывания эвристического бейзлайна, не просто «играть», а находить нетривиальные стратегии (например, отбивать мяч краем ракетки ради ускорения).

### Что внутри

| Файл | Назначение |
|---|---|
| `env_pong.py` | Кастомное `gymnasium`-окружение Pong: физика мяча/ракеток, нормализованные наблюдения, режимы рендеринга и демо-матчей |
| `models.py` | Игроки: `Baseline` (эвристика — следование за мячом), `HumanPlayer`, `DQN_StBaselines3` (обёртка над `stable-baselines3`), `MyDQN` — основной агент |
| `mydqn_components.py` | `ReplayBuffer` и `Q_network` (MLP с dropout) |
| `pong_game.ipynb` | Ноутбук: сбор датасета на эвристике → предобучение (behavior cloning) → RL-дообучение → оценка |

### Архитектура и ключевые технические решения

**Наблюдение (state):** 6 нормализованных признаков — позиция мяча, скорость мяча по обеим осям, позиции обеих ракеток.

**Действия:** вниз / вверх / стоять.

**Пайплайн обучения:**
1. **Behavior cloning** — предобучение `Q_network` на датасете действий эвристики `Baseline` (imitation learning), чтобы дать агенту разумную стартовую точку и не учить базовым навыкам (следить за мячом) с нуля через RL.
2. **RL fine-tuning (DQN)** — дообучение на взаимодействии со средой:
   - **MC-Dropout UCB exploration** — при выборе действия сеть делает несколько stochastic forward-проходов (dropout включён), из разброса предсказаний считается оценка неуверенности, действие выбирается по `mean_Q + β(t)·std_Q`, где `β(t)` затухает со временем;
   - **ε-greedy поверх UCB** — дополнительный гарантированный источник разнообразия действий на старте обучения (нужен, чтобы уверенная после предобучения сеть не схлопывалась в одно действие);
   - **Double DQN** — снижение систематической переоценки Q-значений;
   - **Soft target update (Polyak averaging)** — плавное обновление target-сети вместо резкой периодической синхронизации;
   - **Reward shaping** — небольшой бонус за отбитие мячом ближе к краю ракетки (даёт мячу ускорение, против которого эвристика не успевает реагировать).

### Как запустить

```bash
pip install -r requirements.txt
```

Открыть `pong_game.ipynb` и последовательно выполнить ячейки: сбор датасета → предобучение → RL-дообучение → оценка (`env.demo()`).

Для визуальной демонстрации обученного агента:
```python
from env_pong import PongEnv
from models import MyDQN, Baseline

env = PongEnv(render_mode="human", opponent=Baseline(difficult=1))
stats = env.demo(left_player=agent)  # agent — обученный MyDQN
```

### Результаты

Агент обучен против эвристического бейзлайна (`Baseline(difficult=1)`), который следует за мячом и возвращается к центру поля. Итоговая политика находит и использует стратегию отбивания мяча краем ракетки, придающую дополнительное ускорение — против которого бейзлайн не успевает адаптироваться.

<!-- TODO: заполнить финальными метриками после последнего прогона -->
| Метрика | Значение |
|---|---|
| Win rate против `Baseline(difficult=1)` | `TODO` |
| Среднее вознаграждение за розыгрыш | `TODO` |
| Число шагов обучения | `TODO` |

### Возможные дальнейшие улучшения

- Bootstrapped DQN / ансамбль голов как более устойчивая альтернатива MC-Dropout для exploration.
- Prioritized Experience Replay.
- Обучение против нескольких уровней сложности эвристики / self-play.

### Лицензия

Учебный проект, лицензия не устанавливалась — код можно свободно использовать и изучать, но без формальных гарантий и обязательств.

---

## 🇬🇧 English

### About

This isn't just "another DQN on Pong" — the goal was to design and stress-test a non-standard exploration mechanism for DQN: instead of classic ε-greedy, the agent estimates epistemic uncertainty via **MC-Dropout** and selects actions using a **UCB**-style rule, balancing expected action value against the network's confidence in that estimate.

### Goals

- Build a Pong environment from scratch on `gymnasium` — normalized observations, configurable opponent difficulty, rendering and demo modes.
- Train a DQN agent fine-tuned on top of an imitation-learning warm start.
- Explore **MC-Dropout as an exploration signal** in RL (an alternative to ε-greedy / bootstrapped DQN) — understand in practice where this idea works and where it introduces side effects (training instability, policy collapse).
- Get the agent to beat a heuristic baseline, ideally by discovering non-trivial strategies (e.g. hitting the ball with the paddle's edge for extra speed).

### Repository layout

| File | Purpose |
|---|---|
| `env_pong.py` | Custom `gymnasium` Pong environment: ball/paddle physics, normalized observations, rendering and demo-match support |
| `models.py` | Players: `Baseline` (ball-following heuristic), `HumanPlayer`, `DQN_StBaselines3` (wrapper around `stable-baselines3`), `MyDQN` — the main agent |
| `mydqn_components.py` | `ReplayBuffer` and `Q_network` (dropout MLP) |
| `pong_game.ipynb` | Notebook: dataset collection from the heuristic → behavior cloning pretraining → RL fine-tuning → evaluation |

### Architecture & key design choices

**Observation:** 6 normalized features — ball position, ball velocity (both axes), both paddle positions.

**Actions:** down / up / stay.

**Training pipeline:**
1. **Behavior cloning** — pretrain `Q_network` on the `Baseline` heuristic's action dataset, giving the agent a sane starting point instead of learning basic ball-tracking from scratch via RL.
2. **RL fine-tuning (DQN):**
   - **MC-Dropout UCB exploration** — multiple stochastic forward passes (dropout enabled) at action-selection time; the spread across samples estimates uncertainty, and the action is chosen via `mean_Q + β(t)·std_Q`, with `β(t)` annealing over time;
   - **ε-greedy on top of UCB** — a guaranteed source of action diversity early in training (needed because the post-pretraining network was confident enough to collapse onto a single action otherwise);
   - **Double DQN** — reduces systematic Q-value overestimation;
   - **Soft target updates (Polyak averaging)** — smooth target-network updates instead of abrupt periodic syncing;
   - **Reward shaping** — a small bonus for hitting the ball closer to the paddle's edge, which adds speed the heuristic opponent can't keep up with.

### Getting started

```bash
pip install -r requirements.txt
```

Open `pong_game.ipynb` and run the cells in order: dataset collection → pretraining → RL fine-tuning → evaluation (`env.demo()`).

For a visual demo of the trained agent:
```python
from env_pong import PongEnv
from models import MyDQN, Baseline

env = PongEnv(render_mode="human", opponent=Baseline(difficult=1))
stats = env.demo(left_player=agent)  # agent is the trained MyDQN
```

### Results

The agent was trained against a heuristic baseline (`Baseline(difficult=1)`) that tracks the ball and returns to center. The resulting policy discovers and exploits edge-of-paddle hits for extra ball speed, which the baseline fails to react to in time.

<!-- TODO: fill in final metrics after the last training run -->
| Metric | Value |
|---|---|
| Win rate vs `Baseline(difficult=1)` | `TODO` |
| Average reward per round | `TODO` |
| Training steps | `TODO` |

### Possible future work

- Bootstrapped DQN / ensemble heads as a more stable alternative to MC-Dropout for exploration.
- Prioritized Experience Replay.
- Training against multiple heuristic difficulty levels / self-play.

### License

Educational project — no license attached; feel free to use and study the code, but without formal guarantees.
