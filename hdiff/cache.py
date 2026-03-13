"""On-disk NPZ cache for simulation results.

Each cache entry is a compressed ``*.npz`` file stored in a directory, keyed by
the SHA-256 hex digest of the simulation's ``spec_json``.  The ``completed``
field must be ``1`` for a record to be considered valid; in-progress runs
written with ``completed=0`` are silently skipped on load.

File layout::

    <root_dir>/<sha256_hex>.npz
        t_s              float64[n]        time points (s)
        y                float64[n, m]     state matrix
        timestep         float64 scalar    base_out_dt_s that produced this file
        reason           str               human-readable note (e.g. 'completed')
        completed        int scalar        1 = valid; 0 = partial/aborted
        schema_version   int scalar
        cache_key        str               SHA-256 hex
        spec_json        str               canonical JSON spec
        order_json       str               state layout order
        boundaries_json  str               JSON list of SegmentBoundary dicts
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .result import RunResult, boundaries_from_jsonable, boundaries_to_jsonable


class CacheStore:
    """File-system cache backed by compressed NPZ files.

    Args:
        root_dir: Directory where ``*.npz`` cache files are stored.
            Created automatically if it does not exist.
    """

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root_dir / f"{key}.npz"

    def has(self, key: str) -> bool:
        return self._path(key).is_file()

    def load(
        self,
        key: str,
        *,
        requested_spec_json: str | None = None,
    ) -> tuple[RunResult, str, str]:
        """Load a completed cache record.

        Args:
            key: SHA-256 hex digest.
            requested_spec_json: If provided, the stored spec must match exactly
                or a ``ValueError`` is raised (guards against hash collisions).

        Returns:
            ``(result, order_json, boundaries_json)``

        Raises:
            FileNotFoundError: No file exists for ``key``.
            ValueError: Record is incomplete (``completed != 1``) or spec mismatch.
        """
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(f"cache file not found for key={key}")

        with np.load(path, allow_pickle=False) as data:
            completed = int(data["completed"].item())
            if completed != 1:
                raise ValueError(f"cache record not completed for key={key}")

            stored_spec_json = str(data["spec_json"].item())
            if requested_spec_json is not None and stored_spec_json != requested_spec_json:
                raise ValueError("cache spec_json mismatch")

            order_json = str(data["order_json"].item())
            boundaries_json = str(data["boundaries_json"].item())
            boundaries = boundaries_from_jsonable(json.loads(boundaries_json))
            result = RunResult(
                t_s=np.asarray(data["t_s"]),
                y=np.asarray(data["y"]),
                boundaries=boundaries,
                cache_key=str(data["cache_key"].item()),
                schema_version=int(data["schema_version"].item()),
                spec_json=stored_spec_json,
            )
            return result, order_json, boundaries_json

    def save(
        self,
        *,
        key: str,
        result: RunResult,
        order_json: str,
        timestep: float,
        reason: str,
        completed: int,
    ) -> Path:
        """Write a simulation result to disk as a compressed NPZ file.

        Returns the path of the written file.
        """
        path = self._path(key)
        boundaries_json = json.dumps(
            boundaries_to_jsonable(result.boundaries),
            sort_keys=True,
            separators=(",", ":"),
        )
        np.savez_compressed(
            path,
            t_s=result.t_s,
            y=result.y,
            timestep=float(timestep),
            reason=str(reason),
            completed=int(completed),
            schema_version=int(result.schema_version),
            cache_key=str(result.cache_key),
            spec_json=str(result.spec_json),
            order_json=str(order_json),
            boundaries_json=boundaries_json,
        )
        return path
