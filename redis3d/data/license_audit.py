"""Commercial-safe license allowlist and provenance validator (roadfinal.md §2.6, §5.2, §7 Phase 3).

Enforces strict provenance filters: commercial clients reject models trained on
CC-BY-NC or restricted open weights. Dropping flagged assets prior to .wov shard
compilation guarantees 100% indemnifiable commercial IP.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


class LicenseAuditError(ValueError):
    """Raised when an asset violates commercial license compliance or provenance."""

    pass


# Canonical commercial-safe allowlist
COMMERCIAL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "CC0",
        "CC-BY-4.0",
        "Procedural-Owned",
        "BSD-2-Clause",
        "BSD-3-Clause",
    }
)

# Extended lookup set of approved commercial licenses and aliases
APPROVED_COMMERCIAL_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "MIT LICENSE",
        "APACHE-2.0",
        "APACHE 2.0",
        "APACHE-2",
        "APACHE2",
        "APACHE LICENSE 2.0",
        "CC0",
        "CC0-1.0",
        "CC0 1.0",
        "CC0 1.0 UNIVERSAL",
        "PUBLIC DOMAIN",
        "PUBLIC-DOMAIN",
        "CC-BY-4.0",
        "CC-BY 4.0",
        "CC BY 4.0",
        "CC-BY",
        "CC-BY-3.0",
        "CC-BY 3.0",
        "PROCEDURAL-OWNED",
        "PROCEDURAL",
        "PROPRIETARY-OWNED",
        "COMMERCIAL-CLEAN",
        "AXION-OWNED",
        "REDIS3D-OWNED",
        "BSD-2-CLAUSE",
        "BSD 2-CLAUSE",
        "BSD-2",
        "BSD2",
        "BSD-3-CLAUSE",
        "BSD 3-CLAUSE",
        "BSD-3",
        "BSD3",
        "BSD",
    }
)

# Strict forbidden substrings & keywords (rejection triggers)
FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "-NC",
    "NC-",
    "NON-COMMERCIAL",
    "NONCOMMERCIAL",
    "NON COMMERCIAL",
    "-ND",
    "ND-",
    "NO-DERIVATIVES",
    "NODERIVATIVES",
    "GPL",
    "AGPL",
    "LGPL",
    "EUPL",
    "SSPL",
    "COMMONS CLAUSE",
    "PROPRIETARY",
    "RESTRICTED",
    "ALL RIGHTS RESERVED",
    "UNKNOWN",
    "UNLICENSED",
    "PROPRIETARY-RESTRICTED",
    "EULA",
    "RESEARCH ONLY",
    "ACADEMIC ONLY",
    "PERSONAL USE ONLY",
)

CONTAMINATED_PATTERNS: tuple[str, ...] = FORBIDDEN_KEYWORDS

# Canonical normalization map
_CANONICAL_MAP: dict[str, str] = {
    "MIT": "MIT",
    "APACHE-2.0": "Apache-2.0",
    "APACHE 2.0": "Apache-2.0",
    "APACHE-2": "Apache-2.0",
    "APACHE2": "Apache-2.0",
    "APACHE LICENSE 2.0": "Apache-2.0",
    "CC0": "CC0",
    "CC0-1.0": "CC0",
    "CC0 1.0": "CC0",
    "PUBLIC DOMAIN": "CC0",
    "PUBLIC-DOMAIN": "CC0",
    "CC-BY-4.0": "CC-BY-4.0",
    "CC-BY 4.0": "CC-BY-4.0",
    "CC BY 4.0": "CC-BY-4.0",
    "CC-BY": "CC-BY-4.0",
    "PROCEDURAL-OWNED": "Procedural-Owned",
    "PROCEDURAL": "Procedural-Owned",
    "AXION-OWNED": "Procedural-Owned",
    "REDIS3D-OWNED": "Procedural-Owned",
    "COMMERCIAL-CLEAN": "Procedural-Owned",
    "BSD-2-CLAUSE": "BSD-2-Clause",
    "BSD 2-CLAUSE": "BSD-2-Clause",
    "BSD-2": "BSD-2-Clause",
    "BSD2": "BSD-2-Clause",
    "BSD-3-CLAUSE": "BSD-3-Clause",
    "BSD 3-CLAUSE": "BSD-3-Clause",
    "BSD-3": "BSD-3-Clause",
    "BSD3": "BSD-3-Clause",
    "BSD": "BSD-3-Clause",
}

_TAINTED_PROVENANCE_KEYWORDS: frozenset[str] = frozenset(
    {
        "unknown",
        "unverified",
        "unspecified",
        "none",
        "n/a",
        "scraped",
        "scraped-without-license",
        "pirated",
        "torrent",
        "leaked",
        "unauthorized",
        "ripped",
    }
)


def normalize_license(license_str: str | None) -> str:
    """Normalize license tag string for deterministic audit matching."""
    if not license_str or not isinstance(license_str, str):
        return "UNKNOWN"
    norm = license_str.strip().upper()
    norm = norm.replace("_", "-")
    return norm


def normalize_license_tag(raw_tag: str | None) -> str | None:
    """Return canonical allowlist tag for license_str, or None if unrecognized."""
    norm = normalize_license(raw_tag)
    if is_contaminated_license(norm):
        return None
    if norm in _CANONICAL_MAP:
        return _CANONICAL_MAP[norm]
    for k, v in _CANONICAL_MAP.items():
        if norm == k or norm.startswith(f"{k} ") or norm.endswith(f" {k}"):
            return v
    return None


def is_contaminated_license(license_str: str | None) -> bool:
    """Return True if license_str contains any copyleft, non-commercial, or forbidden keywords."""
    norm = normalize_license(license_str)
    if norm == "UNKNOWN":
        return True
    for bad in FORBIDDEN_KEYWORDS:
        if bad in norm:
            return True
    # Word-boundary check for standalone "NC"
    tokens = set(re.split(r"[\s\-_/]+", norm))
    if "NC" in tokens or "NONCOMMERCIAL" in tokens:
        return True
    return False


def is_license_allowed(license_str: str | None) -> bool:
    """Return True if license_str strictly qualifies for commercial use."""
    if is_contaminated_license(license_str):
        return False

    norm = normalize_license(license_str)

    # Check against allowlist
    if norm in APPROVED_COMMERCIAL_LICENSES:
        return True

    # Check prefix / suffix variants (e.g. "MIT LICENSE", "CC0-1.0 UNIVERSAL")
    for approved in APPROVED_COMMERCIAL_LICENSES:
        if norm == approved or norm.startswith(f"{approved} ") or norm.endswith(f" {approved}"):
            return True

    return False


def is_commercial_safe(license_tag: str | None) -> bool:
    """Alias to is_license_allowed."""
    return is_license_allowed(license_tag)


def audit_provenance(manifest: dict[str, Any], require_present: bool = False) -> tuple[bool, str]:
    """Audit provenance tracking to ensure clear, verified legal chain of custody.

    Returns (is_valid, reason).
    """
    if not isinstance(manifest, dict):
        return False, f"Manifest must be a dict, got {type(manifest).__name__}"

    prov = (
        manifest.get("provenance")
        or manifest.get("source")
        or manifest.get("origin")
        or manifest.get("legal_chain")
        or manifest.get("dataset")
        or manifest.get("generator")
    )

    if prov is None:
        if require_present:
            return False, "Missing required 'provenance' / 'source' field in manifest"
        return True, "No explicit provenance provided"

    if isinstance(prov, str):
        prov_clean = prov.strip().lower()
        if not prov_clean:
            if require_present:
                return False, "Empty provenance string"
            return True, "Empty provenance string"
        if prov_clean in _TAINTED_PROVENANCE_KEYWORDS:
            return False, f"Provenance explicitly marked as tainted: '{prov}'"
        for kw in _TAINTED_PROVENANCE_KEYWORDS:
            if kw in prov_clean.split():
                return False, f"Provenance contains prohibited keyword '{kw}': '{prov}'"
        return True, "Valid provenance string"

    if isinstance(prov, dict):
        if not prov:
            if require_present:
                return False, "Empty provenance dict"
            return True, "Empty provenance dict"
        valid_keys = {"source", "dataset", "author", "repo", "url", "pipeline", "generator"}
        found_keys = set(prov.keys()) & valid_keys
        if require_present and not found_keys:
            return False, f"Provenance dict lacks valid attribution keys (found: {list(prov.keys())})"

        for k, v in prov.items():
            if isinstance(v, str):
                v_clean = v.strip().lower()
                if v_clean in _TAINTED_PROVENANCE_KEYWORDS:
                    return False, f"Provenance key '{k}' contains tainted value '{v}'"
        return True, "Valid provenance dict"

    return False, f"Invalid provenance type: {type(prov).__name__}"


def audit_asset_license(
    manifest: dict[str, Any],
    strict: bool = False,
    raise_on_fail: bool = False,
    require_provenance: bool = False,
) -> bool:
    """Audit single asset dictionary for commercial license validity and provenance.

    Args:
        manifest: Asset dictionary or manifest entry.
        strict: If True, raises LicenseAuditError upon failure.
        raise_on_fail: Alias for strict.
        require_provenance: If True, requires explicit, valid provenance tracking.

    Returns:
        True if approved, False if rejected.

    Raises:
        LicenseAuditError: If (strict or raise_on_fail) and audit fails.
    """
    should_raise = strict or raise_on_fail
    asset_id = str(manifest.get("asset_id", manifest.get("id", "<unnamed>")))

    # 1. Provenance check
    prov_ok, prov_reason = audit_provenance(manifest, require_present=require_provenance)
    if not prov_ok:
        if should_raise:
            raise LicenseAuditError(f"Asset '{asset_id}' failed provenance audit: {prov_reason}")
        return False

    # 2. Extract license tag
    tag = manifest.get("license_tag") or manifest.get("license") or manifest.get("license_type")
    if tag is None and "metadata" in manifest and isinstance(manifest["metadata"], dict):
        tag = manifest["metadata"].get("license_tag") or manifest["metadata"].get("license")

    if tag is None:
        if should_raise:
            raise LicenseAuditError(f"Asset '{asset_id}' missing license tag.")
        return False

    # 3. Check license allowlist and contamination
    allowed = is_license_allowed(str(tag))
    if not allowed:
        if should_raise:
            raise LicenseAuditError(
                f"Asset '{asset_id}' rejected due to non-commercial or contaminated license: '{tag}'."
            )
        return False

    return True


def enforce_provenance(manifest: dict[str, Any]) -> None:
    """Enforce provenance and license check before asset is accepted into training corpus.

    Raises LicenseAuditError if validation fails.
    """
    audit_asset_license(manifest, strict=True, require_provenance=True)


def filter_allowed_assets(
    assets: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split asset collection into allowed (commercial-safe) and rejected lists."""
    allowed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for asset in assets:
        if audit_asset_license(asset, strict=False):
            allowed.append(asset)
        else:
            rejected.append(asset)
    return allowed, rejected


def audit_manifest_file(
    path: Path | str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit an entire manifest file (JSON or JSONL format).

    Args:
        path: Path to manifest JSON/JSONL file.

    Returns:
        (approved, rejected): Tuple of asset manifest lists.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return [], []

    items: list[dict[str, Any]] = []

    # Attempt JSON parse
    if text.startswith("[") or text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("assets", data.get("entries", [data]))
        except json.JSONDecodeError:
            items = []

    if not items:
        # Fallback to JSONL
        for line in text.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                rec = json.loads(s)
                if isinstance(rec, dict):
                    items.append(rec)
            except json.JSONDecodeError as err:
                raise LicenseAuditError(f"Malformed JSONL line in {p}: {err}") from err

    return filter_allowed_assets(items)


def audit_manifest(manifest_or_path: dict | list | Path | str) -> dict[str, Any]:
    """Audit a manifest JSON file or loaded manifest object and return summary dictionary."""
    if isinstance(manifest_or_path, (str, Path)):
        p = Path(manifest_or_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = manifest_or_path

    if isinstance(data, dict):
        entries = data.get("assets", data.get("entries", [data]))
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unsupported manifest format: {type(data)}")

    total = len(entries)
    allowed_count = 0
    rejected_entries = []

    for item in entries:
        if not isinstance(item, dict):
            continue
        asset_id = item.get("asset_id", item.get("id", "unknown"))
        if audit_asset_license(item):
            allowed_count += 1
        else:
            rejected_entries.append(
                {
                    "asset_id": asset_id,
                    "license": item.get("license_tag") or item.get("license", "missing"),
                }
            )

    return {
        "total": total,
        "allowed_count": allowed_count,
        "rejected_count": len(rejected_entries),
        "rejected_entries": rejected_entries,
        "is_100pct_commercial": (allowed_count == total),
    }
