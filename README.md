# 💊 PharmacyInventoryEnv

A **real-world OpenEnv environment** for pharmacy inventory management.  
An AI agent learns to stock medicines wisely, fulfill prescriptions on time, and minimize losses from expired stock.

---

## 🧠 Environment Description

The agent manages a pharmacy with **6 essential medicines** over N days.  
Each day brings new prescriptions. The agent must decide when to reorder, which prescriptions to fill, and whether to discard near-expiry stock.

**This is a real-world task** — it mirrors actual decisions made by pharmacists daily.

---

## ⚙️ API

Follows the standard OpenEnv `step()` / `reset()` / `state()` interface:

```python
from pharmacy_env import PharmacyInventoryEnv

env = PharmacyInventoryEnv(task="easy", seed=42)
state = env.reset()

while True:
    action = my_agent(state)          # your agent here
    state, reward, done, info = env.step(action)
    if done:
        print(f"Final Score: {state.score}")
        break
```

---

## 🎯 Tasks (3 difficulty levels)

| Task   | Days | Rx/Day | Expiry Range | Start Cash | Target Score |
|--------|------|--------|-------------|------------|-------------|
| easy   | 14   | 1–2    | 30–90 days  | ₹5,000     | ≥ 0.65      |
| medium | 30   | 2–5    | 15–60 days  | ₹3,000     | ≥ 0.55      |
| hard   | 60   | 3–8    | 7–30 days   | ₹1,500     | ≥ 0.45      |

---

## 👁️ Observation Space

Each `state` is a `PharmacyState` Pydantic model:

```
PharmacyState:
  day                  int           Current day (1..N)
  inventory            List[MedicineItem]
    ├── name           str           Medicine name
    ├── stock          int           Units on shelf
    ├── min_stock      int           Reorder threshold
    ├── max_stock      int           Shelf capacity
    ├── expiry_days    int           Days until batch expires
    ├── unit_price     float         Selling price (₹)
    └── prescription_required bool
  pending_prescriptions List[dict]  Open prescriptions to fulfill
  cash_balance          float        Available ₹
  expired_losses        float        Cumulative ₹ lost to expiry
  stockout_events       int          Total missed prescriptions
  score                 float        Current normalized score [0,1]
```

---

## 🎮 Action Space

Actions are plain Python dicts:

| Action    | Fields                          | Effect                        |
|-----------|---------------------------------|-------------------------------|
| `reorder` | `medicine_idx`, `qty`           | Purchase stock, deduct cash   |
| `fulfill` | `prescription_id`               | Sell medicine, gain revenue   |
| `discard` | `medicine_idx`                  | Clear expiring batch          |
| `wait`    | _(none)_                        | Do nothing                    |

---

## 🏆 Reward Function (Partial Progress Signals)

| Event                        | Reward  |
|------------------------------|---------|
| Prescription fulfilled       | +0.15   |
| Prescription missed (stockout)| -0.05  |
| Proactive reorder            | +0.02   |
| Medicine batch expired       | -0.02   |
| Insufficient cash / invalid  | -0.01   |

Score is normalized to **0.0–1.0** as a weighted combination of:
- Prescription fulfillment rate (50%)
- Cash balance ratio (30%)
- Expiry loss penalty (20%)

---

## 🤖 Graders

```python
from graders import EasyGrader, MediumGrader, HardGrader

def my_agent(state): ...    # your agent

result = EasyGrader().run(my_agent, seed=42)
print(result)
# {'task': 'easy', 'final_score': 0.71, 'passed': True, ...}
```

---

## 📊 Baseline Results (Rule-based Agent)

Run `python baseline_agent.py` for reproducible scores:

```
Task: EASY  (target ≥ 0.65)
  Avg final score : 0.6821
  Pass rate       : 100%

Task: MEDIUM (target ≥ 0.55)
  Avg final score : 0.5934
  Pass rate       : 67%

Task: HARD  (target ≥ 0.45)
  Avg final score : 0.4712
  Pass rate       : 33%
```

---

## 🚀 Setup

```bash
pip install pydantic gradio pyyaml
python baseline_agent.py          # run baseline on all tasks
python app.py                     # launch Gradio UI
```

---

## 🐳 Docker / Hugging Face Spaces

```bash
docker build -t pharmacy-env .
docker run -p 7860:7860 pharmacy-env
```

Or deploy directly to Hugging Face Spaces — the `Dockerfile` is included.

---

## 📁 File Structure

```
pharmacy_env/
├── pharmacy_env.py      # Main environment (PharmacyInventoryEnv)
├── graders.py           # EasyGrader, MediumGrader, HardGrader
├── baseline_agent.py    # Rule-based baseline + CLI runner
├── app.py               # Gradio web UI for HF Spaces
├── openenv.yaml         # OpenEnv spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🏥 Real-World Relevance

This environment mirrors genuine pharmacy management decisions:
- **Inventory optimization** under cash constraints
- **Expiry risk management** for perishable medicines
- **Prescription fulfillment** with partial availability
- **Reorder timing** with stochastic demand

Suitable for training RL agents, LLM tool-use agents, or rule-based planning systems.
