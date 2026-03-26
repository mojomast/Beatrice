# Beatrice IRC Bot

Beatrice is an OpenRouter-backed IRC bot with short public replies, richer private help, typed durable memory, safe web fetch, IRC environment awareness, and an admin approval flow for dangerous autonomous actions.

## What She Can Do

- answer in channels when addressed or when a channel is explicitly opted into chat mode
- answer private messages conversationally
- inspect IRC state such as server, joined channels, users, topics, nick changes, and `WHOIS`
- fetch and summarize safe public web pages and JSON APIs
- store typed memories (`fact`, `note`, `observation`, `summary`) and per-subject profiles
- request runtime behavior changes, which must be approved by a human admin before execution

## How To Talk To Her

- channel prompt: `Beatrice: explain DNS`
- command prompt: `!bot ask explain DNS`
- private message: just send a message directly to `Beatrice`
- ask her to join channel chat from private message: `talk in #ussycode`
- ask her to post in a channel from private message: `say hello in #ussycode`

## Persistence

- runtime overrides are stored in `BOT_RUNTIME_FILE`
- durable memories and profiles are stored in `BOT_MEMORY_DB_FILE`
- approval and dangerous-action events are appended to `BOT_AUDIT_LOG_FILE`
- API secrets are stored separately in `bot/secrets.json`
- rolling live chat context remains process-local and is rebuilt during runtime

## Admin Approval Flow

Dangerous autonomous actions do not execute immediately.

1. Beatrice queues the action and returns an approval ID.
2. A human admin approves it in a private message with `!bot approve <id> <password>`.
3. A human admin can reject it with `!bot reject <id> <password>`.

You can inspect pending actions with `!bot approvals`.

If `BOT_ADMIN_NICKS` is configured, only those nicks may approve or reject actions.

## Environment

Start from `.env.example` and configure at least:

- `OPENROUTER_API_KEY`
- `IRC_SERVER`
- `IRC_NICK`
- `BOT_ADMIN_PASSWORD`
- `BOT_ADMIN_NICKS` for stricter approval control

Persist the `data/` directory in Docker if you want runtime settings, memory, and audit logs to survive restarts.
