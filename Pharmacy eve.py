"""
PharmacyInventoryEnv - A real-world OpenEnv environment for pharmacy inventory management.
An AI agent learns to optimize stock levels, handle expiry, and fulfill prescriptions.
"""

from __future__ import annotations
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any
from pydantic import BaseModel


# ─────────────────────────────────────────────
#  Data models
# ─────────────────────────────────────────────

class MedicineItem(BaseModel):
    name: str
    stock: int                  # units on hand
    min_stock: int              # reorder point
    max_stock: int              # shelf capacity
    expiry_days: int            # days until expiry
    unit_price: float           # INR
    prescription_required: bool


class PharmacyState(BaseModel):
    day: int
    inventory: list[MedicineItem]
    pending_prescriptions: list[dict]   # [{id, medicine, qty, fulfilled}]
    cash_balance: float
    expired_losses: float               # cumulative INR lost to expiry
    stockout_events: int                # total stockouts
    score: float


# ─────────────────────────────────────────────
#  Action space (discrete)
# ─────────────────────────────────────────────
# Actions are dicts:
#   {"type": "reorder",   "medicine_idx": int, "qty": int}
#   {"type": "fulfill",   "prescription_id": str}
#   {"type": "discard",   "medicine_idx": int}          # discard near-expiry stock
#   {"type": "wait"}                                     # do nothing this step


MEDICINES_CATALOG = [
    {"name": "Paracetamol 500mg",  "min_stock": 50,  "max_stock": 300, "unit_price": 2.5,   "prescription_required": False},
    {"name": "Amoxicillin 250mg",  "min_stock": 20,  "max_stock": 120, "unit_price": 8.0,   "prescription_required": True},
    {"name": "Metformin 500mg",    "min_stock": 30,  "max_stock": 200, "unit_price": 3.5,   "prescription_required": True},
    {"name": "Cetirizine 10mg",    "min_stock": 25,  "max_stock": 150, "unit_price": 1.8,   "prescription_required": False},
    {"name": "Omeprazole 20mg",    "min_stock": 20,  "max_stock": 100, "unit_price": 5.0,   "prescription_required": False},
    {"name": "Azithromycin 500mg", "min_stock": 10,  "max_stock": 60,  "unit_price": 18.0,  "prescription_required": True},
]

REORDER_COST_FACTOR = 0.6   # purchase price = 60% of selling price


class PharmacyInventoryEnv:
    """
    OpenEnv-compatible environment: Pharmacy Inventory Management.

    The agent manages a 6-medicine pharmacy over N days.
    It must reorder stock wisely, fulfill prescriptions on time,
    and discard medicines before they expire to minimise losses.

    Reward signals (partial progress, 0.0 – 1.0):
      +0.15  per prescription fulfilled
      -0.05  per prescription missed (stockout)
      -0.02  per unit expired (capped contribution)
      +daily profit normalised to [0,1]
    """

    metadata = {"version": "1.0.0", "render_modes": ["text"]}

    def __init__(self, task: str = "easy", seed: int = 42):
        assert task in ("easy", "medium", "hard"), f"Unknown task: {task}"
        self.task = task
        self.seed = seed
        self._rng = random.Random(seed)

        # Task parameters
        self._config = {
            "easy":   {"days": 14, "rx_per_day": (1, 2), "expiry_range": (30, 90),  "start_cash": 5000.0},
            "medium": {"days": 30, "rx_per_day": (2, 5), "expiry_range": (15, 60),  "start_cash": 3000.0},
            "hard":   {"days": 60, "rx_per_day": (3, 8), "expiry_range": (7,  30),  "start_cash": 1500.0},
        }[task]

        self._state: PharmacyState | None = None

    # ── OpenEnv API ─────────────────────────────

    def reset(self) -> PharmacyState:
        """Reset environment to initial state and return first observation."""
        self._rng = random.Random(self.seed)
        cfg = self._config

        inventory = []
        for med in MEDICINES_CATALOG:
            inventory.append(MedicineItem(
                name=med["name"],
                stock=self._rng.randint(med["min_stock"], med["max_stock"] // 2),
                min_stock=med["min_stock"],
                max_stock=med["max_stock"],
                expiry_days=self._rng.randint(*cfg["expiry_range"]),
                unit_price=med["unit_price"],
                prescription_required=med["prescription_required"],
            ))

        self._state = PharmacyState(
            day=1,
            inventory=inventory,
            pending_prescriptions=self._generate_prescriptions(),
            cash_balance=cfg["start_cash"],
            expired_losses=0.0,
            stockout_events=0,
            score=0.0,
        )
        self._total_days = cfg["days"]
        self._fulfilled = 0
        self._total_rx = 0
        return self._state.model_copy(deep=True)

    def step(self, action: dict) -> tuple[PharmacyState, float, bool, dict]:
        """
        Execute one action.

        Returns:
            (new_state, reward, done, info)
        """
        assert self._state is not None, "Call reset() before step()"
        reward = 0.0
        info: dict[str, Any] = {"action": action, "day": self._state.day}

        atype = action.get("type", "wait")

        if atype == "reorder":
            reward += self._do_reorder(action, info)

        elif atype == "fulfill":
            reward += self._do_fulfill(action, info)

        elif atype == "discard":
            reward += self._do_discard(action, info)

        elif atype == "wait":
            info["result"] = "waited"

        else:
            info["result"] = f"unknown action: {atype}"

        # End-of-day bookkeeping
        reward += self._end_of_day(info)

        done = self._state.day >= self._total_days
        self._state.score = self._compute_score()
        info["score"] = self._state.score

        return self._state.model_copy(deep=True), reward, done, info

    def state(self) -> PharmacyState:
        """Return current environment state."""
        assert self._state is not None, "Call reset() first"
        return self._state.model_copy(deep=True)

    # ── Internal helpers ─────────────────────────

    def _do_reorder(self, action: dict, info: dict) -> float:
        idx = action.get("medicine_idx", -1)
        qty = action.get("qty", 0)
        if not (0 <= idx < len(self._state.inventory)) or qty <= 0:
            info["result"] = "invalid reorder"
            return -0.01

        med = self._state.inventory[idx]
        space = med.max_stock - med.stock
        qty = min(qty, space)
        cost = qty * med.unit_price * REORDER_COST_FACTOR

        if cost > self._state.cash_balance:
            info["result"] = "insufficient cash"
            return -0.01

        self._state.cash_balance -= cost
        med.stock += qty
        # Fresh stock resets expiry proportionally
        med.expiry_days = max(med.expiry_days, self._config["expiry_range"][1] // 2)
        info["result"] = f"reordered {qty}x {med.name} for ₹{cost:.1f}"
        return 0.02  # small positive for proactive reorder

    def _do_fulfill(self, action: dict, info: dict) -> float:
        rx_id = action.get("prescription_id")
        rx = next((r for r in self._state.pending_prescriptions if r["id"] == rx_id and not r["fulfilled"]), None)
        if rx is None:
            info["result"] = "prescription not found"
            return -0.01

        med_idx = next((i for i, m in enumerate(self._state.inventory) if m.name == rx["medicine"]), None)
        if med_idx is None:
            info["result"] = "medicine not in inventory"
            return -0.05

        med = self._state.inventory[med_idx]
        if med.stock < rx["qty"]:
            info["result"] = f"stockout: need {rx['qty']}, have {med.stock}"
            self._state.stockout_events += 1
            self._total_rx += 1
            return -0.05

        revenue = rx["qty"] * med.unit_price
        med.stock -= rx["qty"]
        rx["fulfilled"] = True
        self._state.cash_balance += revenue
        self._fulfilled += 1
        self._total_rx += 1
        info["result"] = f"fulfilled rx {rx_id}: {rx['qty']}x {med.name} +₹{revenue:.1f}"
        return 0.15

    def _do_discard(self, action: dict, info: dict) -> float:
        idx = action.get("medicine_idx", -1)
        if not (0 <= idx < len(self._state.inventory)):
            info["result"] = "invalid discard"
            return -0.01
        med = self._state.inventory[idx]
        lost = med.stock * med.unit_price * REORDER_COST_FACTOR
        self._state.expired_losses += lost
        info["result"] = f"discarded {med.stock}x {med.name}, lost ₹{lost:.1f}"
        med.stock = 0
        return -0.03

    def _end_of_day(self, info: dict) -> float:
        reward = 0.0
        expired_today = []

        for med in self._state.inventory:
            med.expiry_days -= 1
            if med.expiry_days <= 0:
                lost = med.stock * med.unit_price * REORDER_COST_FACTOR
                self._state.expired_losses += lost
                expired_today.append(f"{med.name} ({med.stock} units, ₹{lost:.1f})")
                med.stock = 0
                med.expiry_days = self._config["expiry_range"][1]  # reset
                reward -= 0.02

        if expired_today:
            info["expired"] = expired_today

        # New prescriptions arrive next day
        self._state.day += 1
        if self._state.day <= self._total_days:
            self._state.pending_prescriptions.extend(self._generate_prescriptions())

        return reward

    def _generate_prescriptions(self) -> list[dict]:
        cfg = self._config
        n = self._rng.randint(*cfg["rx_per_day"])
        rxs = []
        for i in range(n):
            med = self._rng.choice(MEDICINES_CATALOG)
            rxs.append({
                "id": f"RX{self._state.day if self._state else 0}_{i}_{self._rng.randint(1000,9999)}",
                "medicine": med["name"],
                "qty": self._rng.randint(1, 5),
                "fulfilled": False,
                "prescription_required": med["prescription_required"],
            })
        return rxs

    def _compute_score(self) -> float:
        """Normalized score 0.0–1.0."""
        fulfill_rate = (self._fulfilled / max(self._total_rx, 1))
        cash_score = min(self._state.cash_balance / self._config["start_cash"], 2.0) / 2.0
        expiry_penalty = min(self._state.expired_losses / 5000.0, 1.0)
        return round(max(0.0, min(1.0, 0.5 * fulfill_rate + 0.3 * cash_score - 0.2 * expiry_penalty)), 4)

    def render(self) -> str:
        s = self._state
        lines = [
            f"═══ Pharmacy Env | Day {s.day}/{self._total_days} | Task: {self.task.upper()} ═══",
            f"  Cash: ₹{s.cash_balance:.2f}  |  Score: {s.score:.4f}",
            f"  Stockouts: {s.stockout_events}  |  Expired losses: ₹{s.expired_losses:.2f}",
            "",
            "  INVENTORY:",
        ]
        for i, m in enumerate(s.inventory):
            bar = "█" * (m.stock * 10 // m.max_stock)
            lines.append(f"  [{i}] {m.name:25s} stock={m.stock:3d}/{m.max_stock} exp={m.expiry_days}d  {bar}")
        lines.append("")
        lines.append(f"  PENDING Rx ({len([r for r in s.pending_prescriptions if not r['fulfilled']])} open):")
        for rx in s.pending_prescriptions:
            if not rx["fulfilled"]:
                lines.append(f"    {rx['id']}: {rx['qty']}x {rx['medicine']}")
        return "\n".join(lines)
