# Operator Guide

## IRC Awareness

Beatrice tracks:

- joined and known channels
- per-channel user lists from `NAMES`
- channel topics from `TOPIC` and topic numerics
- recent nick changes
- `WHOIS` lookups on demand

This state is exposed to the model in private conversations and partially folded into prompt context for channels and private messages.

## Typed Memory And Profiles

Durable memory records are stored as:

- `scope`
- `kind` (`fact`, `note`, `observation`, `summary`)
- optional `subject`
- `content`
- timestamps

Profiles are stored per `scope + subject` and are updated either explicitly through tools or conservatively from clear first-person statements in conversation.

## Safe Web Fetch

Web fetch is restricted to public `http` and `https` targets.

- no embedded credentials
- no localhost or private IP targets
- no non-default ports
- limited redirects
- no HTTPS-to-HTTP downgrade
- text and JSON content only
- bounded body size

Fetched content is treated as untrusted data, not instructions.

## Runtime Changes And Approval Queue

Privileged autonomous actions are queued instead of executed directly.

- view queue: `!bot approvals`
- approve: `!bot approve <id> <password>`
- reject: `!bot reject <id> <password>`

Rules:

- approvals must come from a private message
- if `BOT_ADMIN_NICKS` is set, only those nicks may approve or reject
- approvals expire after `BOT_APPROVAL_TIMEOUT_SECONDS`

Direct human admin commands like `!bot set ...` and `!bot save runtime ...` still work as explicit operator actions.

## Files And Data

- runtime overrides: `BOT_RUNTIME_FILE`
- memory database: `BOT_MEMORY_DB_FILE`
- audit log: `BOT_AUDIT_LOG_FILE`
- secrets: `bot/secrets.json`

Mount the `data/` directory in Docker deployments if you want these files to persist.

## Audit Trail

The audit log is append-only JSONL and records:

- approval requests
- approvals
- rejections
- dangerous action execution results

This provides a minimal operator trail without introducing another service.
