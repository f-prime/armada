"""Prepared weights keyed by immutable model generation. Mutations use mesh.LOCK."""

from .schema import FloatArray

type Weights = dict[str, FloatArray]
resident: dict[str, Weights] = {}
