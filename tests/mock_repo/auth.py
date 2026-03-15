"""
tests/mock_repo/auth.py
Authentication module used as test data for the RAG pipeline.
"""

import hashlib
import secrets
from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """Represents an authenticated user."""
    username: str
    email: str
    password_hash: str
    is_active: bool = True


def hash_password(password: str) -> str:
    """Hash a plaintext password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def authenticate(username: str, password: str, user_store: dict) -> Optional[User]:
    """
    Authenticate a user by username and password.

    Args:
        username:   The user's login name.
        password:   The plaintext password to verify.
        user_store: Dict mapping usernames to User objects.

    Returns:
        The User object if authentication succeeds, else None.
    """
    user = user_store.get(username)
    if user is None:
        return None
    if not user.is_active:
        return None
    if user.password_hash != hash_password(password):
        return None
    return user


def generate_token(username: str) -> str:
    """Generate a secure random session token for the given user."""
    return f"{username}:{secrets.token_hex(32)}"


def login(username: str, password: str, user_store: dict) -> Optional[str]:
    """
    Full login flow: authenticate then generate a session token.

    Returns:
        A session token string on success, or None on failure.
    """
    user = authenticate(username, password, user_store)
    if user:
        return generate_token(username)
    return None
