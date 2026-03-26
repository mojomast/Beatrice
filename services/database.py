"""
ussynet.database — SQLite persistence layer for channel services.

Manages users, registered channels, per-channel access lists, and bans.
Modeled after Undernet's X/CService database design with Dancer heritage.

Schema:
    users     — registered service accounts
    channels  — registered IRC channels
    access    — per-channel access entries (user ↔ channel, level 0-500)
    bans      — per-channel hostmask bans with optional expiry

All stdlib, no external deps.  Python 3.13+.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("ussynet.database")

# ---------------------------------------------------------------------------
# Flag constants
# ---------------------------------------------------------------------------

# User flags (bitfield)
USER_FLAG_ADMIN = 1
USER_FLAG_SUSPENDED = 2

# Channel flags (bitfield)
CHAN_FLAG_SUSPENDED = 1

# Access-level boundaries (Undernet X style)
ACCESS_MIN = 0
ACCESS_MAX = 500
ACCESS_OWNER = 500

# Allowed automode values
AUTOMODE_VALUES = ("none", "voice", "op")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_password(password: str, salt: str) -> str:
    """SHA-256 hash of password with salt: H(salt + password)."""
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _generate_salt() -> str:
    """16 random bytes as hex → 32-char salt string."""
    return os.urandom(16).hex()


def _normalize_channel(name: str) -> str:
    """Lowercase channel name for consistent storage/lookups."""
    return name.lower()


def _now_iso() -> str:
    """Current UTC timestamp in ISO-8601 format for SQLite."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _row_to_dict(row: sqlite3.Row | None) -> Optional[dict]:
    """Convert a sqlite3.Row to a plain dict, or None if row is None."""
    if row is None:
        return None
    return dict(row)


# ---------------------------------------------------------------------------
# SQL DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT    UNIQUE NOT NULL COLLATE NOCASE,
    password        TEXT    NOT NULL,
    salt            TEXT    NOT NULL,
    email           TEXT,
    registered_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen       TIMESTAMP,
    last_hostmask   TEXT,
    flags           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS channels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT    UNIQUE NOT NULL,
    registered_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    registered_by   INTEGER REFERENCES users(id),
    description     TEXT    DEFAULT '',
    url             TEXT    DEFAULT '',
    topic           TEXT    DEFAULT '',
    mode_lock       TEXT    DEFAULT '',
    autotopic       INTEGER DEFAULT 0,
    flags           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS access (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id      INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    level           INTEGER NOT NULL,
    automode        TEXT    DEFAULT 'none' CHECK(automode IN ('none','voice','op')),
    added_by        INTEGER REFERENCES users(id),
    added_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    modified_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(channel_id, user_id)
);

CREATE TABLE IF NOT EXISTS bans (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id      INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    mask            TEXT    NOT NULL,
    reason          TEXT    DEFAULT '',
    set_by          INTEGER REFERENCES users(id),
    set_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at      TIMESTAMP,
    level           INTEGER DEFAULT 75
);

-- Speed up common lookups
CREATE INDEX IF NOT EXISTS idx_access_channel ON access(channel_id);
CREATE INDEX IF NOT EXISTS idx_access_user    ON access(user_id);
CREATE INDEX IF NOT EXISTS idx_bans_channel   ON bans(channel_id);
CREATE INDEX IF NOT EXISTS idx_bans_expires   ON bans(expires_at);

CREATE TABLE IF NOT EXISTS vhosts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pattern         TEXT    UNIQUE NOT NULL COLLATE NOCASE,
    description     TEXT    DEFAULT '',
    added_by        INTEGER REFERENCES users(id),
    added_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active       INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS user_vhosts (
    user_id         INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    vhost_id        INTEGER NOT NULL REFERENCES vhosts(id) ON DELETE CASCADE,
    set_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_vhosts_vhost ON user_vhosts(vhost_id);

CREATE TABLE IF NOT EXISTS auto_vhosts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    hostmask        TEXT    UNIQUE NOT NULL,
    vhost           TEXT    NOT NULL,
    added_by        INTEGER REFERENCES users(id),
    added_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class Database:
    """SQLite database manager for dancussy channel services.

    Usage::

        db = Database("dancussy.db")
        db.connect()
        uid = db.create_user("mojo", "hunter2", email="mojo@ussy.host")
        cid = db.register_channel("#ussycode", uid, description="main dev channel")
        db.close()
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the database, enable WAL mode, and create tables."""
        logger.info("Connecting to database: %s", self.db_path)
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        self.conn.row_factory = sqlite3.Row
        # WAL mode for better concurrent read/write performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Enforce foreign key constraints (off by default in SQLite)
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA_SQL)
        self.conn.commit()
        logger.info("Database ready — tables ensured.")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed.")

    def _cursor(self) -> sqlite3.Cursor:
        """Return a cursor, raising RuntimeError if not connected."""
        if self.conn is None:
            raise RuntimeError("Database is not connected — call connect() first")
        return self.conn.cursor()

    # ==================================================================
    # USER METHODS
    # ==================================================================

    def create_user(self, username: str, password: str, email: str | None = None) -> int:
        """Register a new user account.

        Args:
            username: Desired login name (case-insensitive uniqueness).
            password: Plaintext password — will be salted + hashed.
            email:    Optional contact email.

        Returns:
            The new user's id.

        Raises:
            ValueError: If the username is already taken.
        """
        salt = _generate_salt()
        pw_hash = _hash_password(password, salt)
        cur = self._cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password, salt, email) VALUES (?, ?, ?, ?)",
                (username, pw_hash, salt, email),
            )
            self.conn.commit()
            user_id = cur.lastrowid
            logger.info("Created user %r (id=%d)", username, user_id)
            return user_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Username {username!r} is already registered")

    def authenticate(self, username: str, password: str) -> Optional[dict]:
        """Verify credentials and return the user dict, or None on failure.

        Also returns None if the user account is suspended.
        """
        cur = self._cursor()
        cur.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,))
        row = cur.fetchone()
        if row is None:
            return None
        user = dict(row)
        expected = _hash_password(password, user["salt"])
        if user["password"] != expected:
            return None
        # Deny suspended users
        if user["flags"] & USER_FLAG_SUSPENDED:
            logger.warning("Auth attempt for suspended user %r", username)
            return None
        return user

    def get_user(self, username: str) -> Optional[dict]:
        """Look up a user by username (case-insensitive)."""
        cur = self._cursor()
        cur.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,))
        return _row_to_dict(cur.fetchone())

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Look up a user by numeric id."""
        cur = self._cursor()
        cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return _row_to_dict(cur.fetchone())

    def update_last_seen(self, user_id: int, hostmask: str) -> None:
        """Record the last time and hostmask we saw this user."""
        cur = self._cursor()
        cur.execute(
            "UPDATE users SET last_seen = ?, last_hostmask = ? WHERE id = ?",
            (_now_iso(), hostmask, user_id),
        )
        self.conn.commit()

    def is_admin(self, user_id: int) -> bool:
        """Return True if user has the admin flag set."""
        user = self.get_user_by_id(user_id)
        if user is None:
            return False
        return bool(user["flags"] & USER_FLAG_ADMIN)

    def set_admin(self, user_id: int, is_admin: bool) -> None:
        """Set or clear the admin flag on a user."""
        cur = self._cursor()
        if is_admin:
            cur.execute(
                "UPDATE users SET flags = flags | ? WHERE id = ?",
                (USER_FLAG_ADMIN, user_id),
            )
        else:
            cur.execute(
                "UPDATE users SET flags = flags & ~? WHERE id = ?",
                (USER_FLAG_ADMIN, user_id),
            )
        self.conn.commit()

    def set_suspended(self, user_id: int, suspended: bool) -> None:
        """Suspend or unsuspend a user account."""
        cur = self._cursor()
        if suspended:
            cur.execute(
                "UPDATE users SET flags = flags | ? WHERE id = ?",
                (USER_FLAG_SUSPENDED, user_id),
            )
        else:
            cur.execute(
                "UPDATE users SET flags = flags & ~? WHERE id = ?",
                (USER_FLAG_SUSPENDED, user_id),
            )
        self.conn.commit()

    # ==================================================================
    # CHANNEL METHODS
    # ==================================================================

    def register_channel(self, name: str, user_id: int, description: str = "") -> int:
        """Register an IRC channel with the services bot.

        The registering user is automatically added at level 500 (owner).

        Args:
            name:        Channel name (e.g. ``#ussycode``).
            user_id:     The registering user's id.
            description: Optional channel description.

        Returns:
            The new channel's id.

        Raises:
            ValueError: If the channel is already registered.
        """
        norm = _normalize_channel(name)
        cur = self._cursor()
        try:
            cur.execute(
                "INSERT INTO channels (name, registered_by, description) VALUES (?, ?, ?)",
                (norm, user_id, description),
            )
            channel_id = cur.lastrowid
            # Auto-add registrant as channel owner (500)
            cur.execute(
                "INSERT INTO access (channel_id, user_id, level, automode, added_by) "
                "VALUES (?, ?, ?, 'op', ?)",
                (channel_id, user_id, ACCESS_OWNER, user_id),
            )
            self.conn.commit()
            logger.info("Registered channel %s (id=%d) by user_id=%d", norm, channel_id, user_id)
            return channel_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Channel {norm!r} is already registered")

    def unregister_channel(self, name: str) -> None:
        """Drop a channel registration and all associated access/ban entries.

        ON DELETE CASCADE handles the access and bans tables.
        """
        norm = _normalize_channel(name)
        cur = self._cursor()
        cur.execute("DELETE FROM channels WHERE name = ?", (norm,))
        self.conn.commit()
        logger.info("Unregistered channel %s", norm)

    def get_channel(self, name: str) -> Optional[dict]:
        """Look up a channel by name (case-insensitive)."""
        norm = _normalize_channel(name)
        cur = self._cursor()
        cur.execute("SELECT * FROM channels WHERE name = ?", (norm,))
        return _row_to_dict(cur.fetchone())

    def get_registered_channels(self) -> list[dict]:
        """Return all registered channels."""
        cur = self._cursor()
        cur.execute("SELECT * FROM channels ORDER BY name")
        return [dict(row) for row in cur.fetchall()]

    def update_channel(self, name: str, **kwargs) -> None:
        """Update one or more fields on a registered channel.

        Allowed keyword args: description, url, topic, mode_lock, autotopic, flags.

        Raises:
            ValueError: If an invalid field name is passed.
            KeyError:   If no matching channel exists.
        """
        allowed = {"description", "url", "topic", "mode_lock", "autotopic", "flags"}
        bad = set(kwargs) - allowed
        if bad:
            raise ValueError(f"Invalid channel fields: {bad}")
        if not kwargs:
            return

        norm = _normalize_channel(name)
        # Build SET clause dynamically
        set_parts = [f"{col} = ?" for col in kwargs]
        values = list(kwargs.values())
        values.append(norm)

        cur = self._cursor()
        cur.execute(
            f"UPDATE channels SET {', '.join(set_parts)} WHERE name = ?",
            values,
        )
        if cur.rowcount == 0:
            raise KeyError(f"Channel {norm!r} is not registered")
        self.conn.commit()

    # ==================================================================
    # ACCESS METHODS
    # ==================================================================

    def _resolve_channel_id(self, channel_name: str) -> int:
        """Get channel id by name, raising ValueError if not found."""
        ch = self.get_channel(channel_name)
        if ch is None:
            raise ValueError(f"Channel {channel_name!r} is not registered")
        return ch["id"]

    def _resolve_user_id(self, username: str) -> int:
        """Get user id by username, raising ValueError if not found."""
        user = self.get_user(username)
        if user is None:
            raise ValueError(f"User {username!r} does not exist")
        return user["id"]

    def add_access(
        self,
        channel_name: str,
        username: str,
        level: int,
        added_by_id: int,
        automode: str = "none",
    ) -> bool:
        """Add a user to a channel's access list.

        Args:
            channel_name: The channel name.
            username:     The user to add.
            level:        Access level (0-500). Must be strictly less than
                          the adder's level on the channel.
            added_by_id:  User id of the person adding access.
            automode:     One of ``'none'``, ``'voice'``, ``'op'``.

        Returns:
            True on success.

        Raises:
            ValueError: If level is out of range, automode is invalid,
                        the user already has access, or the adder's level
                        is insufficient.
        """
        if not (ACCESS_MIN <= level <= ACCESS_MAX):
            raise ValueError(f"Access level must be {ACCESS_MIN}-{ACCESS_MAX}, got {level}")
        if automode not in AUTOMODE_VALUES:
            raise ValueError(f"automode must be one of {AUTOMODE_VALUES}, got {automode!r}")

        channel_id = self._resolve_channel_id(channel_name)
        user_id = self._resolve_user_id(username)

        # Verify adder has sufficient level
        adder_access = self._get_access_raw(channel_id, added_by_id)
        adder_level = adder_access["level"] if adder_access else 0
        if level >= adder_level:
            raise ValueError(
                f"Cannot add level {level} — your level ({adder_level}) must be higher"
            )

        cur = self._cursor()
        try:
            cur.execute(
                "INSERT INTO access (channel_id, user_id, level, automode, added_by) "
                "VALUES (?, ?, ?, ?, ?)",
                (channel_id, user_id, level, automode, added_by_id),
            )
            self.conn.commit()
            logger.info(
                "Added access: %s on %s level %d (by user_id=%d)",
                username, channel_name, level, added_by_id,
            )
            return True
        except sqlite3.IntegrityError:
            raise ValueError(
                f"User {username!r} already has access on {channel_name}"
            )

    def remove_access(self, channel_name: str, username: str) -> None:
        """Remove a user from a channel's access list."""
        channel_id = self._resolve_channel_id(channel_name)
        user_id = self._resolve_user_id(username)
        cur = self._cursor()
        cur.execute(
            "DELETE FROM access WHERE channel_id = ? AND user_id = ?",
            (channel_id, user_id),
        )
        self.conn.commit()

    def modify_access(
        self,
        channel_name: str,
        username: str,
        level: int | None = None,
        automode: str | None = None,
    ) -> None:
        """Modify an existing access entry (level and/or automode).

        Raises:
            ValueError: On invalid level/automode or missing access entry.
        """
        channel_id = self._resolve_channel_id(channel_name)
        user_id = self._resolve_user_id(username)

        updates: dict[str, object] = {"modified_at": _now_iso()}
        if level is not None:
            if not (ACCESS_MIN <= level <= ACCESS_MAX):
                raise ValueError(f"Access level must be {ACCESS_MIN}-{ACCESS_MAX}, got {level}")
            updates["level"] = level
        if automode is not None:
            if automode not in AUTOMODE_VALUES:
                raise ValueError(f"automode must be one of {AUTOMODE_VALUES}, got {automode!r}")
            updates["automode"] = automode

        set_parts = [f"{col} = ?" for col in updates]
        values = list(updates.values())
        values.extend([channel_id, user_id])

        cur = self._cursor()
        cur.execute(
            f"UPDATE access SET {', '.join(set_parts)} "
            "WHERE channel_id = ? AND user_id = ?",
            values,
        )
        if cur.rowcount == 0:
            raise ValueError(
                f"No access entry for {username!r} on {channel_name}"
            )
        self.conn.commit()

    def _get_access_raw(self, channel_id: int, user_id: int) -> Optional[dict]:
        """Internal: fetch an access row by channel_id + user_id."""
        cur = self._cursor()
        cur.execute(
            "SELECT * FROM access WHERE channel_id = ? AND user_id = ?",
            (channel_id, user_id),
        )
        return _row_to_dict(cur.fetchone())

    def get_access(self, channel_name: str, username: str) -> Optional[dict]:
        """Get a specific access entry, including the username.

        Returns None if no entry exists.
        """
        channel_id = self._resolve_channel_id(channel_name)
        user_id = self._resolve_user_id(username)
        cur = self._cursor()
        cur.execute(
            "SELECT a.*, u.username FROM access a "
            "JOIN users u ON u.id = a.user_id "
            "WHERE a.channel_id = ? AND a.user_id = ?",
            (channel_id, user_id),
        )
        return _row_to_dict(cur.fetchone())

    def get_access_list(self, channel_name: str) -> list[dict]:
        """Return all access entries for a channel, sorted by level descending.

        Each dict includes the ``username`` field from the users table.
        """
        channel_id = self._resolve_channel_id(channel_name)
        cur = self._cursor()
        cur.execute(
            "SELECT a.*, u.username FROM access a "
            "JOIN users u ON u.id = a.user_id "
            "WHERE a.channel_id = ? "
            "ORDER BY a.level DESC",
            (channel_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_user_channels(self, username: str) -> list[dict]:
        """Return all channels a user has access on, with channel name and level."""
        user_id = self._resolve_user_id(username)
        cur = self._cursor()
        cur.execute(
            "SELECT c.name, a.level, a.automode, a.channel_id "
            "FROM access a "
            "JOIN channels c ON c.id = a.channel_id "
            "WHERE a.user_id = ? "
            "ORDER BY c.name",
            (user_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ==================================================================
    # BAN METHODS
    # ==================================================================

    def add_ban(
        self,
        channel_name: str,
        mask: str,
        reason: str,
        set_by_id: int,
        duration: int = 0,
        level: int = 75,
    ) -> int:
        """Add a hostmask ban to a channel.

        Args:
            channel_name: Channel name.
            mask:         Ban mask (e.g. ``*!*@bad.host``).
            reason:       Human-readable reason.
            set_by_id:    User id of the person setting the ban.
            duration:     Duration in seconds; 0 means permanent.
            level:        Ban level (default 75).

        Returns:
            The new ban's id.
        """
        channel_id = self._resolve_channel_id(channel_name)
        expires_at: str | None = None
        if duration > 0:
            ts = datetime.now(timezone.utc).timestamp() + duration
            expires_at = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        cur = self._cursor()
        cur.execute(
            "INSERT INTO bans (channel_id, mask, reason, set_by, expires_at, level) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (channel_id, mask, reason, set_by_id, expires_at, level),
        )
        self.conn.commit()
        ban_id = cur.lastrowid
        logger.info(
            "Added ban #%d on %s: %s (expires=%s)",
            ban_id, channel_name, mask, expires_at or "never",
        )
        return ban_id

    def remove_ban(self, ban_id: int) -> None:
        """Remove a ban by its id."""
        cur = self._cursor()
        cur.execute("DELETE FROM bans WHERE id = ?", (ban_id,))
        self.conn.commit()

    def remove_ban_by_mask(self, channel_name: str, mask: str) -> None:
        """Remove all bans matching an exact mask on a channel."""
        channel_id = self._resolve_channel_id(channel_name)
        cur = self._cursor()
        cur.execute(
            "DELETE FROM bans WHERE channel_id = ? AND mask = ?",
            (channel_id, mask),
        )
        self.conn.commit()

    def get_bans(self, channel_name: str) -> list[dict]:
        """Return all bans for a channel, including the setter's username."""
        channel_id = self._resolve_channel_id(channel_name)
        cur = self._cursor()
        cur.execute(
            "SELECT b.*, u.username AS set_by_username "
            "FROM bans b "
            "LEFT JOIN users u ON u.id = b.set_by "
            "WHERE b.channel_id = ? "
            "ORDER BY b.set_at DESC",
            (channel_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_matching_ban(self, channel_name: str, hostmask: str) -> Optional[dict]:
        """Check if a hostmask matches any active (non-expired) ban on a channel.

        Uses ``fnmatch.fnmatch`` for glob-style mask matching.
        Returns the first matching ban dict, or None.
        """
        channel_id = self._resolve_channel_id(channel_name)
        cur = self._cursor()
        now = _now_iso()
        # Fetch active bans: not expired (expires_at is NULL or in the future)
        cur.execute(
            "SELECT b.*, u.username AS set_by_username "
            "FROM bans b "
            "LEFT JOIN users u ON u.id = b.set_by "
            "WHERE b.channel_id = ? AND (b.expires_at IS NULL OR b.expires_at > ?) "
            "ORDER BY b.level DESC",
            (channel_id, now),
        )
        hostmask_lower = hostmask.lower()
        for row in cur.fetchall():
            ban = dict(row)
            if fnmatch.fnmatch(hostmask_lower, ban["mask"].lower()):
                return ban
        return None

    def cleanup_expired_bans(self) -> int:
        """Remove all bans whose expiry time has passed.

        Returns:
            Number of bans removed.
        """
        now = _now_iso()
        cur = self._cursor()
        cur.execute(
            "DELETE FROM bans WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        removed = cur.rowcount
        self.conn.commit()
        if removed:
            logger.info("Cleaned up %d expired ban(s)", removed)
        return removed

    # ==================================================================
    # VHOST METHODS
    # ==================================================================

    def add_vhost(self, pattern: str, description: str = "", added_by_id: int | None = None) -> int:
        """Add an available vhost pattern that users can choose from.

        Args:
            pattern:      The vhost string (e.g. ``user.ussy.host``).
            description:  Optional human-readable description.
            added_by_id:  Admin user id who added this vhost.

        Returns:
            The new vhost's id.

        Raises:
            ValueError: If the vhost pattern already exists.
        """
        cur = self._cursor()
        try:
            cur.execute(
                "INSERT INTO vhosts (pattern, description, added_by) VALUES (?, ?, ?)",
                (pattern, description, added_by_id),
            )
            self.conn.commit()
            vhost_id = cur.lastrowid
            logger.info("Added vhost %r (id=%d)", pattern, vhost_id)
            return vhost_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Vhost {pattern!r} already exists")

    def remove_vhost(self, pattern: str) -> None:
        """Remove a vhost pattern. Also clears any user assignments to it."""
        cur = self._cursor()
        cur.execute("SELECT id FROM vhosts WHERE pattern = ? COLLATE NOCASE", (pattern,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Vhost {pattern!r} does not exist")
        vhost_id = row["id"]
        # Remove user assignments first (CASCADE would handle it, but be explicit)
        cur.execute("DELETE FROM user_vhosts WHERE vhost_id = ?", (vhost_id,))
        cur.execute("DELETE FROM vhosts WHERE id = ?", (vhost_id,))
        self.conn.commit()
        logger.info("Removed vhost %r (id=%d)", pattern, vhost_id)

    def get_vhost(self, pattern: str) -> Optional[dict]:
        """Look up a vhost by pattern (case-insensitive)."""
        cur = self._cursor()
        cur.execute("SELECT * FROM vhosts WHERE pattern = ? COLLATE NOCASE", (pattern,))
        return _row_to_dict(cur.fetchone())

    def get_vhost_by_id(self, vhost_id: int) -> Optional[dict]:
        """Look up a vhost by id."""
        cur = self._cursor()
        cur.execute("SELECT * FROM vhosts WHERE id = ?", (vhost_id,))
        return _row_to_dict(cur.fetchone())

    def list_vhosts(self, active_only: bool = True) -> list[dict]:
        """Return all available vhost patterns.

        Args:
            active_only: If True, only return active (non-disabled) vhosts.
        """
        cur = self._cursor()
        if active_only:
            cur.execute("SELECT * FROM vhosts WHERE is_active = 1 ORDER BY pattern")
        else:
            cur.execute("SELECT * FROM vhosts ORDER BY pattern")
        return [dict(row) for row in cur.fetchall()]

    def set_user_vhost(self, user_id: int, vhost_id: int) -> None:
        """Assign a vhost to a user (replaces any existing assignment).

        Raises:
            ValueError: If the vhost doesn't exist or is inactive.
        """
        vhost = self.get_vhost_by_id(vhost_id)
        if vhost is None:
            raise ValueError("Vhost does not exist")
        if not vhost["is_active"]:
            raise ValueError("Vhost is currently disabled")

        cur = self._cursor()
        cur.execute(
            "INSERT OR REPLACE INTO user_vhosts (user_id, vhost_id, set_at) "
            "VALUES (?, ?, ?)",
            (user_id, vhost_id, _now_iso()),
        )
        self.conn.commit()
        logger.info("Set vhost for user_id=%d to vhost_id=%d (%s)", user_id, vhost_id, vhost["pattern"])

    def clear_user_vhost(self, user_id: int) -> None:
        """Remove the vhost assignment for a user."""
        cur = self._cursor()
        cur.execute("DELETE FROM user_vhosts WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def get_user_vhost(self, user_id: int) -> Optional[dict]:
        """Get the vhost currently assigned to a user.

        Returns the vhost dict (from the vhosts table) or None.
        """
        cur = self._cursor()
        cur.execute(
            "SELECT v.* FROM vhosts v "
            "JOIN user_vhosts uv ON uv.vhost_id = v.id "
            "WHERE uv.user_id = ?",
            (user_id,),
        )
        return _row_to_dict(cur.fetchone())

    def toggle_vhost(self, pattern: str, active: bool) -> None:
        """Enable or disable a vhost pattern.

        Raises:
            ValueError: If the vhost doesn't exist.
        """
        cur = self._cursor()
        cur.execute(
            "UPDATE vhosts SET is_active = ? WHERE pattern = ? COLLATE NOCASE",
            (1 if active else 0, pattern),
        )
        if cur.rowcount == 0:
            raise ValueError(f"Vhost {pattern!r} does not exist")
        self.conn.commit()

    # ==================================================================
    # AUTO-VHOST METHODS
    # ==================================================================

    def add_auto_vhost(self, hostmask: str, vhost: str, added_by_id: int | None = None) -> int:
        """Add an automatic vhost mapping.

        When a user matching *hostmask* (glob pattern like ``*@185.255.121.49``
        or ``ussybot!*@*``) connects/joins, the bot will CHGHOST them to *vhost*.

        Returns:
            The new auto_vhost id.

        Raises:
            ValueError: If the hostmask already has an auto-vhost.
        """
        cur = self._cursor()
        try:
            cur.execute(
                "INSERT INTO auto_vhosts (hostmask, vhost, added_by) VALUES (?, ?, ?)",
                (hostmask, vhost, added_by_id),
            )
            self.conn.commit()
            av_id = cur.lastrowid
            logger.info("Added auto-vhost: %s -> %s (id=%d)", hostmask, vhost, av_id)
            return av_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Auto-vhost for {hostmask!r} already exists")

    def remove_auto_vhost(self, hostmask: str) -> None:
        """Remove an auto-vhost mapping by hostmask."""
        cur = self._cursor()
        cur.execute("DELETE FROM auto_vhosts WHERE hostmask = ?", (hostmask,))
        if cur.rowcount == 0:
            raise ValueError(f"No auto-vhost for {hostmask!r}")
        self.conn.commit()
        logger.info("Removed auto-vhost for %s", hostmask)

    def list_auto_vhosts(self) -> list[dict]:
        """Return all auto-vhost mappings."""
        cur = self._cursor()
        cur.execute("SELECT * FROM auto_vhosts ORDER BY hostmask")
        return [dict(row) for row in cur.fetchall()]

    def match_auto_vhost(self, hostmask: str) -> Optional[str]:
        """Check if a hostmask matches any auto-vhost rule.

        Uses ``fnmatch`` for glob matching. Returns the vhost string
        to apply, or None if no match.
        """
        cur = self._cursor()
        cur.execute("SELECT hostmask, vhost FROM auto_vhosts")
        hostmask_lower = hostmask.lower()
        for row in cur.fetchall():
            if fnmatch.fnmatch(hostmask_lower, row["hostmask"].lower()):
                return row["vhost"]
        return None

    # ==================================================================
    # CONVENIENCE / INTROSPECTION
    # ==================================================================

    def __repr__(self) -> str:
        state = "connected" if self.conn else "disconnected"
        return f"<Database path={self.db_path!r} {state}>"
