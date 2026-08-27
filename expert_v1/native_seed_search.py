"""Bounded authoritative seed search for source-order expert replays."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .native_replay_plan import (
    DEFAULT_NATIVE_SEED,
    BattlePlan,
    materialize_replay,
    native_layout_order,
)
from .upgrade_base_cycles import INITIAL_MASKS, INITIAL_QUEUES


DEFAULT_MAXIMUM_SEEDS_TO_TEST = 4096
DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS = 8


class NativeSeedSearchError(RuntimeError):
    """No authoritative libg layout satisfied both observed cycle constraints."""

    def __init__(
        self,
        *,
        battle_tag: str,
        seeds_tested: int,
        maximum_seeds_to_test: int,
        preferred_seed: int,
    ) -> None:
        self.battle_tag = battle_tag
        self.seeds_tested = seeds_tested
        self.maximum_seeds_to_test = maximum_seeds_to_test
        self.preferred_seed = preferred_seed
        super().__init__(
            "native_compatible_seed_not_found:"
            f"battle={battle_tag},tested={seeds_tested},"
            f"limit={maximum_seeds_to_test},preferred={preferred_seed}"
        )


@dataclass(frozen=True)
class NativeSeedResolution:
    preferred_seed: int
    chosen_seed: int
    seeds_tested: int
    maximum_seeds_to_test: int
    cache_hit: bool
    cache_validated: bool
    native_resets: int
    source_seed_recovered: bool
    layouts: tuple[tuple[int, ...], tuple[int, ...]]
    replay: dict[str, Any]
    mappings: tuple[tuple[int, ...], tuple[int, ...]]
    state: dict[str, Any]
    resolution_mode: str = "source_order_bounded_native_seed_search"

    def audit(self) -> dict[str, Any]:
        return {
            "layout_resolution_mode": self.resolution_mode,
            "preferred_seed": self.preferred_seed,
            "chosen_seed": self.chosen_seed,
            "seeds_tested": self.seeds_tested,
            "maximum_seeds_to_test": self.maximum_seeds_to_test,
            "seed_search_cache_hit": self.cache_hit,
            "seed_search_cache_validated": self.cache_validated,
            "seed_search_native_resets": self.native_resets,
            "source_seed_recovered": self.source_seed_recovered,
            "native_layouts": [list(layout) for layout in self.layouts],
        }


@dataclass(frozen=True)
class _CachedSeed:
    seed: int
    seeds_tested: int


@dataclass
class CompatibleNativeSeedSearch:
    """Lazy canonical scan yielding at most N layout-compatible seeds.

    Raw seed scans and semantic candidates are deliberately separate counters:
    incompatible layouts never consume the bounded semantic-preflight budget.
    The object is stateful only so callers can publish exact scan/reset counts
    after stopping early on the first fully successful semantic preflight.
    """

    env: Any
    plan: BattlePlan
    template: Mapping[str, Any]
    preferred_seed: int = DEFAULT_NATIVE_SEED
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST
    maximum_compatible_seeds: int = DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS
    warmup_tick: int = 10
    seeds_scanned: int = 0
    native_resets: int = 0
    compatible_seeds_yielded: int = 0

    def __post_init__(self) -> None:
        if self.maximum_seeds_to_test <= 0:
            raise ValueError("maximum_seeds_to_test must be positive")
        if self.maximum_compatible_seeds <= 0:
            raise ValueError("maximum_compatible_seeds must be positive")
        if self.warmup_tick < 0:
            raise ValueError("warmup_tick must be nonnegative")

    def __iter__(self) -> Iterable[NativeSeedResolution]:
        for seed in _candidate_seeds(
            int(self.preferred_seed), int(self.maximum_seeds_to_test)
        ):
            replay, mappings, layouts, state = _reset_candidate(
                self.env,
                self.plan,
                self.template,
                seed,
                int(self.warmup_tick),
            )
            self.seeds_scanned += 1
            self.native_resets += 1
            players = sorted(state["players"], key=lambda item: int(item["side"]))
            if not all(layouts_accept_plan(self.plan, players)):
                continue
            self.compatible_seeds_yielded += 1
            yield NativeSeedResolution(
                preferred_seed=int(self.preferred_seed),
                chosen_seed=seed,
                seeds_tested=self.seeds_scanned,
                maximum_seeds_to_test=int(self.maximum_seeds_to_test),
                cache_hit=False,
                cache_validated=False,
                native_resets=self.native_resets,
                source_seed_recovered=False,
                layouts=layouts,
                replay=replay,
                mappings=mappings,
                state=state,
                resolution_mode="layout_compatible_semantic_candidate_v1",
            )
            if self.compatible_seeds_yielded >= self.maximum_compatible_seeds:
                return
        if self.compatible_seeds_yielded == 0:
            raise NativeSeedSearchError(
                battle_tag=self.plan.battle_tag,
                seeds_tested=self.seeds_scanned,
                maximum_seeds_to_test=int(self.maximum_seeds_to_test),
                preferred_seed=int(self.preferred_seed),
            )


def compatible_native_seed_search(
    env: Any,
    plan: BattlePlan,
    template: Mapping[str, Any],
    *,
    preferred_seed: int = DEFAULT_NATIVE_SEED,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    maximum_compatible_seeds: int = DEFAULT_MAXIMUM_COMPATIBLE_SEMANTIC_SEEDS,
    warmup_tick: int = 10,
) -> CompatibleNativeSeedSearch:
    """Create a lazy, bounded search for semantic-preflight candidates."""

    return CompatibleNativeSeedSearch(
        env=env,
        plan=plan,
        template=template,
        preferred_seed=int(preferred_seed),
        maximum_seeds_to_test=int(maximum_seeds_to_test),
        maximum_compatible_seeds=int(maximum_compatible_seeds),
        warmup_tick=int(warmup_tick),
    )


_CACHE: dict[tuple[Any, ...], _CachedSeed] = {}
_CACHE_LOCK = threading.Lock()


def clear_native_seed_cache() -> None:
    with _CACHE_LOCK:
        _CACHE.clear()


def native_seed_cache_size() -> int:
    with _CACHE_LOCK:
        return len(_CACHE)


def side_play_sequence(plan: BattlePlan, side: int) -> tuple[int, ...]:
    return tuple(
        action.logical_card_index for action in plan.actions if action.side == side
    )


def layout_accepts_sequence(
    layout: Sequence[int], played: Iterable[int]
) -> bool:
    """Apply the exact eight-card hand/refill transition to one native layout."""
    values = tuple(int(value) for value in layout)
    if len(values) != 8 or set(values) != set(range(8)):
        return False
    hand = set(values[:4])
    queue = list(values[4:])
    for raw in played:
        card = int(raw)
        if card not in hand:
            return False
        incoming = queue.pop(0)
        hand.remove(card)
        hand.add(incoming)
        queue.append(card)
    return True


def layouts_accept_plan(
    plan: BattlePlan, players: Sequence[Mapping[str, Any]]
) -> tuple[bool, bool]:
    if len(players) != 2:
        return False, False
    return tuple(
        layout_accepts_sequence(
            native_layout_order(players[side]), side_play_sequence(plan, side)
        )
        for side in range(2)
    )  # type: ignore[return-value]


def _cycle_constraint_digest(played: Sequence[int]) -> str:
    """Hash the complete compatible-origin set, not merely one chosen state."""
    masks = INITIAL_MASKS.copy()
    queues = INITIAL_QUEUES.copy()
    origin_masks = masks.copy()
    origin_queues = queues.copy()
    for raw in played:
        card = int(raw)
        keep = ((masks >> np.uint16(card)) & np.uint16(1)).astype(bool)
        masks, queues = masks[keep], queues[keep]
        origin_masks, origin_queues = origin_masks[keep], origin_queues[keep]
        if len(masks) == 0:
            raise ValueError("cycle constraint has no compatible initial state")
        incoming = queues[:, 0].astype(np.uint16)
        masks = (
            masks & np.uint16(~(1 << card) & 0xFFFF)
        ) | (np.uint16(1) << incoming)
        queues = np.concatenate(
            (queues[:, 1:], np.full((len(queues), 1), card, dtype=np.uint8)),
            axis=1,
        )
    digest = hashlib.sha256()
    for mask, queue in zip(origin_masks, origin_queues, strict=True):
        digest.update(int(mask).to_bytes(2, "little", signed=False))
        digest.update(bytes(int(value) for value in queue))
    return digest.hexdigest()


def seed_cache_key(plan: BattlePlan) -> tuple[Any, ...]:
    decks = tuple(
        tuple(
            (int(card.card_id), int(card.form_flags), int(card.level or 11))
            for card in side.deck
        )
        for side in plan.sides
    )
    tower_troops = tuple(side.tower_troop for side in plan.sides)
    constraints = tuple(
        _cycle_constraint_digest(side_play_sequence(plan, side))
        for side in range(2)
    )
    return decks + tower_troops + constraints


def _candidate_seeds(preferred_seed: int, limit: int) -> Iterable[int]:
    if limit <= 0:
        raise ValueError("maximum_seeds_to_test must be positive")
    # The source seed is unknown.  Search a canonical ascending domain so the
    # same deck/constraint key always resolves to the same minimum compatible
    # seed regardless of Worker scheduling or a legacy preferred-seed value.
    del preferred_seed
    yield from range(1, limit + 1)


def _reset_candidate(
    env: Any,
    plan: BattlePlan,
    template: Mapping[str, Any],
    seed: int,
    warmup_tick: int,
) -> tuple[
    dict[str, Any],
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
    dict[str, Any],
]:
    replay, mappings = materialize_replay(plan, template, seed=seed)
    state = env.reset(replay, warmup_steps=warmup_tick)
    players = sorted(state["players"], key=lambda item: int(item["side"]))
    if len(players) != 2:
        raise RuntimeError("native seed search did not expose two players")
    layouts = tuple(native_layout_order(player) for player in players)
    return replay, mappings, layouts, state  # type: ignore[return-value]


def resolve_native_seed(
    env: Any,
    plan: BattlePlan,
    template: Mapping[str, Any],
    *,
    preferred_seed: int = DEFAULT_NATIVE_SEED,
    maximum_seeds_to_test: int = DEFAULT_MAXIMUM_SEEDS_TO_TEST,
    warmup_tick: int = 10,
) -> NativeSeedResolution:
    """Find and validate a seed; never claim it is the missing source RNG seed."""
    if maximum_seeds_to_test <= 0:
        raise ValueError("maximum_seeds_to_test must be positive")
    if warmup_tick < 0:
        raise ValueError("warmup_tick must be nonnegative")
    key = seed_cache_key(plan)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    native_resets = 0
    if (
        cached is not None
        and cached.seeds_tested <= maximum_seeds_to_test
    ):
        replay, mappings, layouts, state = _reset_candidate(
            env, plan, template, cached.seed, warmup_tick
        )
        native_resets += 1
        players = sorted(state["players"], key=lambda item: int(item["side"]))
        if all(layouts_accept_plan(plan, players)):
            return NativeSeedResolution(
                preferred_seed=int(preferred_seed),
                chosen_seed=cached.seed,
                seeds_tested=cached.seeds_tested,
                maximum_seeds_to_test=maximum_seeds_to_test,
                cache_hit=True,
                cache_validated=True,
                native_resets=native_resets,
                source_seed_recovered=False,
                layouts=layouts,
                replay=replay,
                mappings=mappings,
                state=state,
            )
        with _CACHE_LOCK:
            _CACHE.pop(key, None)

    seeds_tested = 0
    for seed in _candidate_seeds(int(preferred_seed), maximum_seeds_to_test):
        replay, mappings, layouts, state = _reset_candidate(
            env, plan, template, seed, warmup_tick
        )
        native_resets += 1
        seeds_tested += 1
        players = sorted(state["players"], key=lambda item: int(item["side"]))
        if not all(layouts_accept_plan(plan, players)):
            continue
        with _CACHE_LOCK:
            _CACHE[key] = _CachedSeed(seed=seed, seeds_tested=seeds_tested)
        return NativeSeedResolution(
            preferred_seed=int(preferred_seed),
            chosen_seed=seed,
            seeds_tested=seeds_tested,
            maximum_seeds_to_test=maximum_seeds_to_test,
            cache_hit=False,
            cache_validated=False,
            native_resets=native_resets,
            source_seed_recovered=False,
            layouts=layouts,
            replay=replay,
            mappings=mappings,
            state=state,
        )
    raise NativeSeedSearchError(
        battle_tag=plan.battle_tag,
        seeds_tested=seeds_tested,
        maximum_seeds_to_test=maximum_seeds_to_test,
        preferred_seed=int(preferred_seed),
    )


def resolve_fixed_native_seed(
    env: Any,
    plan: BattlePlan,
    template: Mapping[str, Any],
    *,
    chosen_seed: int,
    warmup_tick: int = 10,
) -> NativeSeedResolution:
    """Reset exactly once with a seed selected by a prior preflight.

    This is deliberately not a one-element seed *search*.  The caller has
    already paid for the bounded search and must carry that exact seed into a
    second, instrumented replay.  The returned state is still inspected by
    :func:`layouts_accept_plan` in the normal runner path, so a changed native
    layout fails closed rather than silently starting another search.
    """
    if warmup_tick < 0:
        raise ValueError("warmup_tick must be nonnegative")
    seed = int(chosen_seed)
    replay, mappings, layouts, state = _reset_candidate(
        env, plan, template, seed, warmup_tick
    )
    return NativeSeedResolution(
        preferred_seed=seed,
        chosen_seed=seed,
        seeds_tested=0,
        maximum_seeds_to_test=0,
        cache_hit=False,
        cache_validated=False,
        native_resets=1,
        source_seed_recovered=False,
        layouts=layouts,
        replay=replay,
        mappings=mappings,
        state=state,
        resolution_mode="fixed_preflight_seed_replay",
    )
