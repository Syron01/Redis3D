"""64-bit Morton serialization (roadfinal.md §3.1.3).

key(x,y,z,l) = (l << 60) | dilate3(x) | (dilate3(y) << 1) | (dilate3(z) << 2)

All sparse lookups run through sorted int64 buffers + torch.searchsorted:
zero Python dicts, zero spconv/torch_scatter builds.
"""

from __future__ import annotations

import torch

_LEVEL_SHIFT = 60
_AXIS_BITS = 20  # supports coordinates up to 2^20 per axis
# For octree levels <= 10 the Morton body uses only bits 0..29. Bits 30..32
# carry the edge axis / sheet id WITHOUT shifting the level nibble (a plain
# key*2 or key*4 would overflow int64 at level 10, where l<<60 sets bit 63).
EDGE_AXIS_SHIFT = 30
SHEET_SHIFT = 32
_SHEET_MASK = ~(1 << SHEET_SHIFT)
_AXIS_MASK = ~(3 << EDGE_AXIS_SHIFT)


def edge_key(cell_keys: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return cell_keys | (axis.to(torch.int64) << EDGE_AXIS_SHIFT)


def edge_split(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return keys & _AXIS_MASK, (keys >> EDGE_AXIS_SHIFT) & 3


def token_key(cell_keys: torch.Tensor, sheet: torch.Tensor) -> torch.Tensor:
    return cell_keys | (sheet.to(torch.int64) << SHEET_SHIFT)


def token_split(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return keys & _SHEET_MASK, (keys >> SHEET_SHIFT) & 1


def _dilate3(v: torch.Tensor) -> torch.Tensor:
    """Spread 20 bits across 60 bit positions (one bit every three)."""
    v = (v | (v << 32)) & 0x001F00000000FFFF
    v = (v | (v << 16)) & 0x001F0000FF0000FF
    v = (v | (v << 8)) & 0x00100F00F00F00F
    v = (v | (v << 4)) & 0x0010C30C30C30C3
    v = (v | (v << 2)) & 0x001249249249249
    return v


def _compact3(v: torch.Tensor) -> torch.Tensor:
    v = v & 0x001249249249249
    v = (v | (v >> 2)) & 0x0010C30C30C30C3
    v = (v | (v >> 4)) & 0x00100F00F00F00F
    v = (v | (v >> 8)) & 0x001F0000FF0000FF
    v = (v | (v >> 16)) & 0x001F00000000FFFF
    v = (v | (v >> 32)) & 0x00000000000FFFFF
    return v


def morton_encode(coords: torch.Tensor, level: int | torch.Tensor) -> torch.Tensor:
    """coords: [..., 3] int64 in [0, 2^20); level: int or [...] int64."""
    coords = coords.to(torch.int64)
    if not torch.is_tensor(level):
        level = torch.full(coords.shape[:-1], int(level), dtype=torch.int64, device=coords.device)
    x, y, z = coords.unbind(-1)
    return (level.to(torch.int64) << _LEVEL_SHIFT) | _dilate3(x) | (_dilate3(y) << 1) | (_dilate3(z) << 2)


def morton_decode(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (coords [..., 3] int64, levels [...] int64).

    Keys are int64 bit containers: level 10 sets bit 63, so the level nibble
    is extracted with a logical mask, never an arithmetic shift.
    """
    keys = keys.to(torch.int64)
    level = (keys >> _LEVEL_SHIFT) & 0xF
    body = keys & ((1 << _LEVEL_SHIFT) - 1)
    x = _compact3(body)
    y = _compact3(body >> 1)
    z = _compact3(body >> 2)
    return torch.stack([x, y, z], dim=-1), level


class MortonIndex:
    """Sorted Morton buffer with vectorized searchsorted membership/lookup."""

    def __init__(self, keys: torch.Tensor):
        self.keys, self.order = torch.sort(keys.to(torch.int64))

    def query(self, keys: torch.Tensor) -> torch.Tensor:
        """Index into the ORIGINAL (unsorted) array, -1 when absent."""
        keys = keys.to(torch.int64)
        pos = torch.searchsorted(self.keys, keys)
        pos_clamped = pos.clamp(max=self.keys.numel() - 1)
        hit = self.keys[pos_clamped] == keys
        return torch.where(hit, self.order[pos_clamped], torch.full_like(pos, -1))

    def __len__(self) -> int:
        return self.keys.numel()