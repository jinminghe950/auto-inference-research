"""Deterministic warehouse simulator: state, tool schemas, and execution.

The simulator is the ground truth. Tool calls mutate it; the oracle in
trajectory.py reads it. Everything is seeded, so two runs with the same seed
see identical initial state and identical instruction scripts.
"""

from __future__ import annotations

import random
from typing import Any

ADJ = ["steel", "brass", "nylon", "carbon", "copper", "rubber", "titanium", "aluminum", "ceramic", "oak"]
NOUN = ["bracket", "washer", "gasket", "hinge", "valve", "flange", "coupler", "spindle", "bushing", "clamp"]
BINS = [f"{r}{c}" for r in "ABCDE" for c in range(1, 9)]
CARRIERS = ["UPS", "FedEx", "DHL"]


def norm_sku(v: Any) -> str:
    return str(v).strip().upper()


def norm_bin(v: Any) -> str:
    return str(v).strip().upper()


class Warehouse:
    """Mutable warehouse state with validated tool execution."""

    def __init__(self, seed: int, n_items: int = 40):
        rng = random.Random(seed)
        skus = rng.sample(range(1000, 9999), n_items)
        names = rng.sample([f"{a} {n}" for a in ADJ for n in NOUN], n_items)
        self.items: dict[str, dict] = {}
        for sku_n, name in zip(skus, names):
            sku = f"SKU-{sku_n}"
            self.items[sku] = {
                "sku": sku,
                "name": name,
                "qty": rng.randint(5, 240),
                "bin": rng.choice(BINS),
                "price": round(rng.uniform(0.5, 90.0), 2),
            }
        self.orders: dict[str, dict] = {}
        self.notes: list[str] = []

    # ------------------------------------------------------------------ tools

    def lookup_item(self, sku: str) -> dict:
        it = self.items.get(norm_sku(sku))
        if not it:
            return {"error": f"unknown sku: {sku}"}
        return dict(it)

    def search_items(self, query: str) -> dict:
        q = str(query).strip().strip("'\"").lower()
        if not q:
            return {"error": "empty query"}
        hits = [
            {"sku": it["sku"], "name": it["name"]}
            for it in self.items.values()
            if q in it["name"].lower()
        ]
        hits.sort(key=lambda h: h["sku"])
        return {"query": q, "matches": hits[:5], "total_matches": len(hits)}

    def move_stock(self, sku: str, from_bin: str, to_bin: str, qty: int) -> dict:
        it = self.items.get(norm_sku(sku))
        if not it:
            return {"error": f"unknown sku: {sku}"}
        if norm_bin(from_bin) != it["bin"]:
            return {"error": f"{it['sku']} is not in bin {norm_bin(from_bin)} (it is in {it['bin']})"}
        if norm_bin(to_bin) not in BINS:
            return {"error": f"unknown bin: {to_bin}"}
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            return {"error": f"qty must be an integer, got {qty!r}"}
        if qty != it["qty"]:
            return {"error": f"partial moves not supported: {it['sku']} has {it['qty']} units, you asked to move {qty}"}
        it["bin"] = norm_bin(to_bin)
        return {"sku": it["sku"], "moved_qty": qty, "from_bin": norm_bin(from_bin), "to_bin": it["bin"]}

    def set_quantity(self, sku: str, new_qty: int, reason: str) -> dict:
        it = self.items.get(norm_sku(sku))
        if not it:
            return {"error": f"unknown sku: {sku}"}
        try:
            new_qty = int(new_qty)
        except (TypeError, ValueError):
            return {"error": f"new_qty must be an integer, got {new_qty!r}"}
        if new_qty < 0:
            return {"error": "new_qty must be >= 0"}
        prev = it["qty"]
        it["qty"] = new_qty
        return {"sku": it["sku"], "previous_qty": prev, "new_qty": new_qty, "reason": str(reason)}

    def create_order(self, order_id: str, sku: str, qty: int) -> dict:
        oid = str(order_id).strip().upper()
        if oid in self.orders:
            return {"error": f"order {oid} already exists"}
        it = self.items.get(norm_sku(sku))
        if not it:
            return {"error": f"unknown sku: {sku}"}
        try:
            qty = int(qty)
        except (TypeError, ValueError):
            return {"error": f"qty must be an integer, got {qty!r}"}
        if qty <= 0 or qty > it["qty"]:
            return {"error": f"qty {qty} out of range for {it['sku']} (on hand: {it['qty']})"}
        self.orders[oid] = {"order_id": oid, "sku": it["sku"], "qty": qty, "status": "open", "carrier": None}
        return dict(self.orders[oid])

    def ship_order(self, order_id: str, carrier: str) -> dict:
        oid = str(order_id).strip().upper()
        o = self.orders.get(oid)
        if not o:
            return {"error": f"unknown order: {order_id}"}
        if o["status"] != "open":
            return {"error": f"order {oid} is {o['status']}, not open"}
        c = str(carrier).strip()
        match = next((k for k in CARRIERS if k.lower() == c.lower()), None)
        if not match:
            return {"error": f"unknown carrier: {carrier} (use one of {CARRIERS})"}
        item = self.items[o["sku"]]
        item["qty"] -= o["qty"]
        o["status"] = "shipped"
        o["carrier"] = match
        return dict(o)

    def get_order(self, order_id: str) -> dict:
        o = self.orders.get(str(order_id).strip().upper())
        if not o:
            return {"error": f"unknown order: {order_id}"}
        return dict(o)

    def cancel_order(self, order_id: str) -> dict:
        o = self.orders.get(str(order_id).strip().upper())
        if not o:
            return {"error": f"unknown order: {order_id}"}
        if o["status"] != "open":
            return {"error": f"order {o['order_id']} is {o['status']}, cannot cancel"}
        o["status"] = "cancelled"
        return dict(o)

    def update_price(self, sku: str, new_price: float) -> dict:
        it = self.items.get(norm_sku(sku))
        if not it:
            return {"error": f"unknown sku: {sku}"}
        try:
            p = round(float(new_price), 2)
        except (TypeError, ValueError):
            return {"error": f"new_price must be a number, got {new_price!r}"}
        if p <= 0:
            return {"error": "new_price must be > 0"}
        prev = it["price"]
        it["price"] = p
        return {"sku": it["sku"], "previous_price": prev, "new_price": p}

    def audit_bin(self, bin: str) -> dict:
        b = norm_bin(bin)
        if b not in BINS:
            return {"error": f"unknown bin: {bin}"}
        contents = sorted(
            ({"sku": it["sku"], "name": it["name"], "qty": it["qty"]} for it in self.items.values() if it["bin"] == b),
            key=lambda x: x["sku"],
        )
        return {"bin": b, "items": contents, "distinct_skus": len(contents)}

    def delete_item(self, sku: str) -> dict:
        it = self.items.pop(norm_sku(sku), None)
        if not it:
            return {"error": f"unknown sku: {sku}"}
        return {"deleted": it["sku"]}

    def log_note(self, text: str) -> dict:
        t = str(text).strip()
        if not t:
            return {"error": "empty note"}
        self.notes.append(t)
        return {"note_id": len(self.notes), "text": t}

    # ------------------------------------------------------------- dispatcher

    def execute(self, name: str, args: dict) -> dict:
        spec = TOOL_PARAMS.get(name)
        if spec is None:
            return {"error": f"unknown tool: {name}"}
        if not isinstance(args, dict):
            return {"error": "arguments must be an object"}
        missing = [p for p in spec if p not in args]
        if missing:
            return {"error": f"missing required arguments: {missing}"}
        try:
            return getattr(self, name)(**{k: args[k] for k in spec})
        except Exception as e:  # defensive: sim must never crash the run
            return {"error": f"tool raised {type(e).__name__}: {e}"}


# Required parameter names per tool (order matters for dispatch only).
TOOL_PARAMS: dict[str, list[str]] = {
    "lookup_item": ["sku"],
    "search_items": ["query"],
    "move_stock": ["sku", "from_bin", "to_bin", "qty"],
    "set_quantity": ["sku", "new_qty", "reason"],
    "create_order": ["order_id", "sku", "qty"],
    "ship_order": ["order_id", "carrier"],
    "get_order": ["order_id"],
    "cancel_order": ["order_id"],
    "update_price": ["sku", "new_price"],
    "audit_bin": ["bin"],
    "delete_item": ["sku"],
    "log_note": ["text"],
}

_S = {"type": "string"}
_I = {"type": "integer"}
_N = {"type": "number"}

_TOOL_DESCS: dict[str, tuple[str, dict[str, dict]]] = {
    "lookup_item": ("Look up one item by SKU; returns name, qty on hand, bin, and price.", {"sku": _S}),
    "search_items": ("Search item names by substring; returns up to 5 matching SKUs.", {"query": _S}),
    "move_stock": (
        "Move the full on-hand quantity of a SKU from its current bin to another bin. "
        "qty must equal the full on-hand quantity and from_bin must be the item's current bin.",
        {"sku": _S, "from_bin": _S, "to_bin": _S, "qty": _I},
    ),
    "set_quantity": ("Set the absolute on-hand quantity of a SKU, with a short reason.", {"sku": _S, "new_qty": _I, "reason": _S}),
    "create_order": ("Create a new open order for qty units of a SKU.", {"order_id": _S, "sku": _S, "qty": _I}),
    "ship_order": ("Ship an open order with a carrier (UPS, FedEx, or DHL); decrements stock.", {"order_id": _S, "carrier": _S}),
    "get_order": ("Fetch one order by id.", {"order_id": _S}),
    "cancel_order": ("Cancel an open order.", {"order_id": _S}),
    "update_price": ("Set a new unit price for a SKU.", {"sku": _S, "new_price": _N}),
    "audit_bin": ("List every SKU currently stored in a bin.", {"bin": _S}),
    "delete_item": ("Permanently remove a SKU from the catalog.", {"sku": _S}),
    "log_note": ("Append a free-text note to the operations log.", {"text": _S}),
}


def tool_schemas() -> list[dict]:
    """OpenAI-format tool definitions, all parameters required."""
    out = []
    for name, (desc, props) in _TOOL_DESCS.items():
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": list(props),
                    },
                },
            }
        )
    return out
