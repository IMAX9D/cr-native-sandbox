"""A small vocabulary adapter over the existing versioned native card catalog."""
from __future__ import annotations

from dataclasses import dataclass

from native_core.card_catalog import catalog, form_index
from .config import digest

PAD, UNKNOWN = 0, 1


@dataclass(frozen=True)
class CardVocabulary:
    native_ids: tuple[int, ...]
    base_ids: tuple[int, ...]

    def __post_init__(self):
        if len(self.native_ids) != len(self.base_ids) or len(set(self.native_ids)) != len(self.native_ids):
            raise ValueError('invalid vocabulary identity mapping')
        if any(type(value) is not int or value <= 0 for value in (*self.native_ids, *self.base_ids)):
            raise ValueError('native card ids must be positive integers')
        object.__setattr__(self, '_tokens', {key: i + 2 for i, key in enumerate(self.native_ids)})
        object.__setattr__(self, '_bases', dict(zip(self.native_ids, self.base_ids)))

    @classmethod
    def from_native(cls) -> 'CardVocabulary':
        rows = catalog()
        forms = form_index()
        ids = tuple(sorted(set(rows) | set(forms)))
        return cls(ids, tuple(forms[key]['base_card_id'] if key in forms else key for key in ids))

    @property
    def size(self) -> int:
        return len(self.native_ids) + 2

    @property
    def sha256(self) -> str:
        return digest({'schema': 'r0-card-vocabulary.v1', 'ids': self.native_ids, 'bases': self.base_ids, 'pad': PAD, 'unknown': UNKNOWN})

    def token(self, native_id: int | None) -> int:
        return UNKNOWN if native_id is None else self._tokens.get(native_id, UNKNOWN)

    def base(self, native_id: int) -> int:
        return self._bases.get(native_id, native_id)
