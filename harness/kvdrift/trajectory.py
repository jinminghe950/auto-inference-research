"""Scripted user: deterministic instruction script + state-aware oracle.

Key invariant: instruction TEXT is a pure function of the trajectory seed, so
every server config sees byte-identical prompts. Only the EXPECTED tool
arguments are resolved against live simulator state at turn time (the model
may have drifted the state; the oracle judges "the right action given the
instruction and the current true state").

Argument-fidelity probes are recall-shaped on purpose: dependent turns need
values (current bin, on-hand qty) that appeared in an earlier tool result,
so degraded KV for those earlier tokens shows up as wrong arguments here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .warehouse import BINS, CARRIERS, NOUN, Warehouse

SYSTEM_PROMPT = """You are a warehouse operations assistant with tool access.

Rules:
- When an instruction requires an action or a data lookup, make exactly one tool call for it.
- When asked a direct question (especially one that says not to use tools), answer in plain text from what you already know from this conversation.
- Argument values must be exact: SKUs, bin codes, order ids, and quantities must match the warehouse records you have seen.
- Keep any free-text replies to one or two sentences.

Session: {session_id}"""


@dataclass
class Turn:
    idx: int
    kind: str
    instruction: str
    expected_tool: str | None  # None => free-text turn, no tool expected
    # field -> spec; spec is one of:
    #   {"eq": value}          exact match after normalization
    #   {"eq_num": value}      numeric match (int/float, tol 1e-6)
    #   {"contains_ci": str}   case-insensitive substring must appear
    #   {"nonempty": True}     any non-empty string
    expected_args: dict[str, dict] = field(default_factory=dict)
    # For recall turns: facts that must appear in the text answer.
    recall_facts: dict[str, Any] = field(default_factory=dict)
    recall_ref_turn: int | None = None
    scoreable: bool = True


class ScriptedUser:
    """Emits turns one at a time; expectations resolved against live state."""

    def __init__(self, seed: int, sim: Warehouse, n_turns: int):
        self.rng = random.Random(seed ^ 0x5EED)
        self.sim = sim
        self.n_turns = n_turns
        self.session_id = f"traj-{seed:08x}"
        # Seed-fixed schedule of concrete entities so instruction text never
        # depends on state or model behaviour.
        skus = sorted(sim.items)
        self.sku_schedule = self.rng.sample(skus, min(len(skus), max(4, n_turns // 4 + 2)))
        self.to_bins = [self.rng.choice(BINS) for _ in self.sku_schedule]
        self.order_qtys = [self.rng.randint(1, 4) for _ in self.sku_schedule]
        self.carriers = [self.rng.choice(CARRIERS) for _ in self.sku_schedule]
        self.prices = [round(self.rng.uniform(1.0, 99.0), 2) for _ in self.sku_schedule]
        self.search_terms = [self.rng.choice(NOUN) for _ in range(n_turns)]
        self.audit_bins = [self.rng.choice(BINS) for _ in range(n_turns)]
        self.restock_adds = [self.rng.randint(5, 30) for _ in self.sku_schedule]
        # Runtime bookkeeping (facts actually shown to the model).
        self.lookup_results: dict[str, tuple[int, dict]] = {}  # sku -> (turn_idx, tool result)
        self.recalled: set[str] = set()

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(session_id=self.session_id)

    def _block(self, idx: int) -> tuple[int, int]:
        return idx // 8, idx % 8

    def next_turn(self, idx: int) -> Turn:
        blk, pos = self._block(idx)
        sku = self.sku_schedule[blk % len(self.sku_schedule)]
        item_live = self.sim.items.get(sku)  # live truth (may be None if model deleted it)

        if pos == 0:
            return Turn(idx, "lookup", f"Look up {sku} and tell me what we have on hand.",
                        "lookup_item", {"sku": {"eq": sku}})

        if pos == 1:
            to_bin = self.to_bins[blk % len(self.to_bins)]
            t = Turn(idx, "dependent_move",
                     f"Now move all of {sku}'s stock out of its current bin into bin {to_bin}. "
                     "Use the exact current bin and the exact on-hand quantity.",
                     "move_stock",
                     {"sku": {"eq": sku}, "to_bin": {"eq": to_bin}})
            if item_live:
                t.expected_args["from_bin"] = {"eq": item_live["bin"]}
                t.expected_args["qty"] = {"eq_num": item_live["qty"]}
            else:
                t.scoreable = False
            return t

        if pos == 2:
            add = self.restock_adds[blk % len(self.restock_adds)]
            t = Turn(idx, "computed_restock",
                     f"Receive a delivery for {sku}: add {add} units on top of its current on-hand "
                     "quantity and set the new absolute quantity, with reason 'delivery'.",
                     "set_quantity", {"sku": {"eq": sku}, "reason": {"nonempty": True}})
            if item_live:
                t.expected_args["new_qty"] = {"eq_num": item_live["qty"] + add}
            else:
                t.scoreable = False
            return t

        if pos == 3:
            oid = f"ORD-{blk + 1:03d}"
            qty = self.order_qtys[blk % len(self.order_qtys)]
            return Turn(idx, "create_order",
                        f"Create order {oid} for {qty} units of {sku}.",
                        "create_order",
                        {"order_id": {"eq": oid}, "sku": {"eq": sku}, "qty": {"eq_num": qty}})

        if pos == 4:
            oid = f"ORD-{blk + 1:03d}"
            carrier = self.carriers[blk % len(self.carriers)]
            return Turn(idx, "ship_order",
                        f"Ship {oid} with {carrier}.",
                        "ship_order", {"order_id": {"eq": oid}, "carrier": {"eq": carrier}})

        if pos == 5:
            if blk % 2 == 0:
                term = self.search_terms[idx % len(self.search_terms)]
                return Turn(idx, "search", f"Search the catalog for '{term}'.",
                            "search_items", {"query": {"eq": term}})
            b = self.audit_bins[idx % len(self.audit_bins)]
            return Turn(idx, "audit", f"Audit bin {b} and report what is stored there.",
                        "audit_bin", {"bin": {"eq": b}})

        if pos == 6:
            if blk % 2 == 0:
                price = self.prices[blk % len(self.prices)]
                return Turn(idx, "price", f"Set the unit price of {sku} to {price:.2f}.",
                            "update_price", {"sku": {"eq": sku}, "new_price": {"eq_num": price}})
            b = self.audit_bins[(idx + 3) % len(self.audit_bins)]
            return Turn(idx, "note", f"Log a note that the cycle count for bin {b} is complete.",
                        "log_note", {"text": {"contains_ci": b}})

        # pos == 7: long-range recall of an earlier lookup's tool result.
        candidates = [s for s, (t_idx, _) in self.lookup_results.items()
                      if s not in self.recalled and idx - t_idx >= 3]
        if candidates:
            ref_sku = sorted(candidates, key=lambda s: self.lookup_results[s][0])[0]
            ref_turn, ref_result = self.lookup_results[ref_sku]
            self.recalled.add(ref_sku)
            t = Turn(idx, "recall_text",
                     f"Without calling any tool, answer directly from this conversation: when you first "
                     f"looked up {ref_sku} earlier in this session, what quantity was on hand and which bin "
                     "was it stored in at that time? One sentence.",
                     None, {}, recall_ref_turn=ref_turn)
            if isinstance(ref_result, dict) and "qty" in ref_result and "bin" in ref_result:
                t.recall_facts = {"qty": ref_result["qty"], "bin": ref_result["bin"]}
            else:
                t.scoreable = False
            return t
        term = self.search_terms[(idx + 5) % len(self.search_terms)]
        return Turn(idx, "search", f"Search the catalog for '{term}'.",
                    "search_items", {"query": {"eq": term}})

    def observe_tool_result(self, turn: Turn, tool_name: str, args: dict, result: dict) -> None:
        """Record what the model actually saw, for later recall probes."""
        if tool_name == "lookup_item" and isinstance(result, dict) and "error" not in result:
            sku = str(result.get("sku", ""))
            if sku and sku not in self.lookup_results:
                self.lookup_results[sku] = (turn.idx, result)
