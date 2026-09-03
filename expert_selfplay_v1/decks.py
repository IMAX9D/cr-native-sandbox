"""Deterministic fixed-learner/high-frequency-opponent deck curriculum."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _deck(value: Mapping[str, Any], side: int = 0) -> dict[str, Any]:
    deck = deepcopy(dict(value["battle"][f"deck{side}"]))
    spells = deck.get("sp")
    if not isinstance(spells, list) or len(spells) != 8:
        raise ValueError("self-play deck must contain exactly eight cards")
    if len({int(card["d"]) for card in spells}) != 8:
        raise ValueError("self-play deck contains duplicate base cards")
    return deck


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DeckFixture:
    episode_index: int
    learner_side: int
    learner_deck_sha256: str
    opponent_deck_sha256: str
    opponent_preset: str
    replay: dict[str, Any]


class DeckScheduler:
    def __init__(
        self,
        *,
        learner_preset: Path,
        opponent_presets: list[Path],
    ) -> None:
        if not learner_preset.is_file() or not opponent_presets:
            raise FileNotFoundError("self-play deck presets are incomplete")
        if any(not path.is_file() for path in opponent_presets):
            raise FileNotFoundError("high-frequency opponent preset is missing")
        self.learner_preset = learner_preset.resolve()
        self.opponent_presets = [path.resolve() for path in opponent_presets]
        self.learner_replay = _read(self.learner_preset)
        self.learner_deck = _deck(self.learner_replay)
        self.learner_sha256 = file_sha256(self.learner_preset)

    def build_batch(self, *, episode_count: int, seed: int) -> list[DeckFixture]:
        if episode_count < 1:
            raise ValueError("deck batch must contain episodes")
        rng = random.Random(seed)
        order = [self.opponent_presets[index % len(self.opponent_presets)]
                 for index in range(episode_count)]
        rng.shuffle(order)
        sides = [index % 2 for index in range(episode_count)]
        rng.shuffle(sides)
        fixtures: list[DeckFixture] = []
        for index, (path, learner_side) in enumerate(zip(order, sides, strict=True)):
            opponent_replay = _read(path)
            opponent_deck = _deck(opponent_replay)
            replay = deepcopy(self.learner_replay)
            replay["battle"][f"deck{learner_side}"] = deepcopy(self.learner_deck)
            replay["battle"][f"deck{1 - learner_side}"] = opponent_deck
            replay["rndSeed"] = seed + index
            fixtures.append(DeckFixture(
                episode_index=index,
                learner_side=learner_side,
                learner_deck_sha256=self.learner_sha256,
                opponent_deck_sha256=file_sha256(path),
                opponent_preset=str(path),
                replay=replay,
            ))
        return fixtures
