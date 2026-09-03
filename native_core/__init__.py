"""Stable Python API for the persistent Surface-free libg worker."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .env import NativeHostError, NativeRoyaleEnv

__all__ = ["NativeHostError", "NativeRoyaleEnv"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .env import NativeHostError, NativeRoyaleEnv

        return {
            "NativeHostError": NativeHostError,
            "NativeRoyaleEnv": NativeRoyaleEnv,
        }[name]
    raise AttributeError(name)
