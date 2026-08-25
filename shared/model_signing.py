"""HMAC-signed pickle save/load for trained models.

pickle.load() on attacker-controlled bytes is a well-known, reliable
remote-code-execution vector (a crafted object's __reduce__ runs
arbitrary code the moment it's unpickled — no different in spirit from
sympify() running eval() during parsing). Every tool that loads a model
takes a caller-supplied model_path, and resolve_path() does not
restrict that to a fixed models directory, so this module signs the
pickled bytes at save time with a server-local secret and verifies the
signature before ever unpickling. A file this server didn't write
itself can never pass verification, regardless of what path it's
loaded from.

The key is persisted at ~/.mcp_ml_signing_key (mode 0600) rather than
kept in memory only, because basic/medium/advanced each also ship
their own standalone server.py for local stdio use — a model trained
via one tier's process and loaded via a separately-launched tier's
process must still verify.

MCP_ML_SIGNING_KEY_FILE overrides that path. In a container the home
directory is part of the image, so a rebuild issued a brand new key
and every model already written to the mounted output volume stopped
verifying — "signature is invalid", which reads as tampering, for
files this server had written itself a day earlier. The deployment
points this at the same persisted volume as .oauth-state.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
import secrets
from pathlib import Path
from typing import Any, BinaryIO


def _key_path() -> Path:
    """Where the signing key lives; env-overridable so it can be persisted."""
    override = os.environ.get("MCP_ML_SIGNING_KEY_FILE", "").strip()
    return Path(override) if override else Path.home() / ".mcp_ml_signing_key"


_KEY_LEN = 32
_SIG_LEN = 32  # SHA-256 digest size


class ModelIntegrityError(ValueError):
    """Raised when a model file's signature is missing or invalid."""


def _get_signing_key() -> bytes:
    path = _key_path()
    try:
        data = path.read_bytes()
        if len(data) == _KEY_LEN:
            return data
    except OSError:
        pass
    key = secrets.token_bytes(_KEY_LEN)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return key


def is_signed_by_us(path: Path) -> bool:
    """Whether this server can verify — and therefore load — this model file.

    list_models advertised four models that every loading tool in the fleet
    refused, and nothing in the listing said so. Checking costs one HMAC over
    bytes already on disk.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return False
    sig, data = blob[:_SIG_LEN], blob[_SIG_LEN:]
    if len(sig) != _SIG_LEN:
        return False
    expected = hmac.new(_get_signing_key(), data, hashlib.sha256).digest()
    return hmac.compare_digest(sig, expected)


def dump_signed(payload: Any, fh: BinaryIO) -> None:
    """Pickle `payload`, prefixing the bytes with an HMAC-SHA256 signature."""
    data = pickle.dumps(payload)
    sig = hmac.new(_get_signing_key(), data, hashlib.sha256).digest()
    fh.write(sig)
    fh.write(data)


def load_signed(fh: BinaryIO) -> Any:
    """Verify the HMAC signature, then unpickle.

    Raises:
        ModelIntegrityError: signature missing, truncated, or invalid.
    """
    blob = fh.read()
    sig, data = blob[:_SIG_LEN], blob[_SIG_LEN:]
    if len(sig) != _SIG_LEN:
        raise ModelIntegrityError(
            "Model file is missing its integrity signature — refusing to unpickle. "
            "This file was not written by this server's train/export tools."
        )
    expected = hmac.new(_get_signing_key(), data, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise ModelIntegrityError(
            "Model file signature is invalid — refusing to unpickle. It was "
            "signed with a different key, so it was written by another "
            "machine, or by this server before its signing key changed — "
            "retrain it, or load it where it was made. (A modified file "
            "fails the same way, which is the point of the check.)"
        )
    return pickle.loads(data)


__all__ = ["ModelIntegrityError", "dump_signed", "is_signed_by_us", "load_signed"]
