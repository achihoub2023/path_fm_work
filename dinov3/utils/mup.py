# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""Utilities for integrating the μP parameterization."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger("dinov3")


def is_mup_enabled(cfg) -> bool:
    """Return ``True`` when the current config enables μP."""

    if cfg is None or not hasattr(cfg, "optim"):
        return False
    optim_cfg = cfg.optim
    try:
        return bool(getattr(optim_cfg, "use_mup"))
    except AttributeError:
        # OmegaConf's DictConfig supports ``get`` but not ``__contains__`` when
        # structure is frozen, so use ``getattr`` with a default.
        return bool(optim_cfg.get("use_mup", False))  # type: ignore[attr-defined]


def _import_mup():
    try:
        import mup  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - defensive error path
        raise ImportError(
            "μP support requested but the `mup` package is not available."
            " Install it with `pip install mup` and try again."
        ) from exc
    return mup


def apply_mup_shapes(module, base_factory: Callable[[], object], *, tag: str | None = None) -> None:
    """Attach μP base shapes to ``module`` using the ``base_factory`` template."""

    if module is None:
        return

    mup = _import_mup()
    base_module = base_factory()
    try:
        mup.set_base_shapes(module, base_module)
    finally:
        del base_module
    if tag:
        logger.info("μP base shapes set for %s", tag)


def maybe_apply_mup_shapes(cfg, module, base_factory: Callable[[], object], *, tag: str | None = None) -> None:
    """Conditionally attach μP base shapes when the feature is enabled."""

    if not is_mup_enabled(cfg):
        return
    apply_mup_shapes(module, base_factory, tag=tag)


__all__ = ["apply_mup_shapes", "is_mup_enabled", "maybe_apply_mup_shapes"]
