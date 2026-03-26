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

Managed child-bot changes also use this queue when requested in normal conversation, including create, update, start, stop, enable, disable, and remove operations.

## Managed Child Bots

Beatrice can orchestrate simple child chatbots that run in their own processes.

- child bots are chat-only and do not get tool access, web access, GitHub access, or approval powers
- Beatrice stores child bot definitions in `BOT_CHILD_BOTS_FILE`
- Beatrice stores child runtime state in `BOT_CHILD_STATE_FILE`
- each child gets its own runtime, memory, audit log, and logs under `BOT_CHILD_DATA_DIR/<id>/`

### Response Modes

- `addressed_only`: reply only when directly addressed in channel or via private message
- `ambient`: reply naturally to strong conversation openings such as questions, topic overlap, or explicit invitations

You can change this with direct commands like:

- `!bot child update id=helper response_mode=addressed_only`
- `!bot child update id=helper response_mode=ambient`

### Child Commands

- `!bot child list`
- `!bot child create id=<id> nick=<nick> channels=#chan prompt="..." response_mode=addressed_only`
- `!bot child update id=<id> prompt="..." response_mode=ambient`
- `!bot child start <id>`
- `!bot child stop <id>`
- `!bot child enable <id>`
- `!bot child disable <id>`
- `!bot child remove <id>`

### Natural-Language Admin Requests

Admins can also ask conversationally for child bots, for example:

- `Beatrice, spin up 5 helper bots for #ussycode with very different personalities`
- `Beatrice, make 3 greeter bots that only respond when talked to`

Beatrice converts these into approval-gated child-bot change requests. Batch-created bots are intentionally varied with distinct voice archetypes and response habits so they do not all act the same.

## Files And Data

- runtime overrides: `BOT_RUNTIME_FILE`
- memory database: `BOT_MEMORY_DB_FILE`
- audit log: `BOT_AUDIT_LOG_FILE`
- child registry: `BOT_CHILD_BOTS_FILE`
- child runtime state: `BOT_CHILD_STATE_FILE`
- child data root: `BOT_CHILD_DATA_DIR`
- secrets: `bot/secrets.json`

Mount the `data/` directory in Docker deployments if you want these files to persist.

## Audit Trail

The audit log is append-only JSONL and records:

- approval requests
- approvals
- rejections
- dangerous action execution results

This provides a minimal operator trail without introducing another service.
