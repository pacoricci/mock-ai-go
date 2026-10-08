# Responses implementation

The adapter lives in `internal/openai`; see the [API contract](openai-api.md)
for public behavior.

| Files | Responsibility |
| --- | --- |
| `responses.go`, `responses_request.go` | Decode, validate, resolve the model, and dispatch generation. |
| `responses_context.go`, `responses_continuation.go` | Prepare context, validate tool results, and construct next state. |
| `responses_resource.go`, `responses_errors.go` | Encode resources, usage, and errors. |
| `responses_store.go` | Bounded in-memory continuation storage. |
| `responses_sse.go` | Events, terminal delivery, and failed-write cleanup. |
| `function_tools.go` | Shared tool normalization and call/result validation. |
| `structured_output.go` | Shared format validation and structured generation binding. |

## Generation

`chat.Model` provides `PlanTools`, `Complete`, `Stream`, and `StreamTools`.
Tool streaming consumes a completed plan, preserves IDs/arguments/usage, splits
at UTF-8 boundaries, and respects cancellation and synchronous backpressure.
Chat translates its wire format in `chat_tools.go` and `chat_tools_sse.go`;
Responses owns its event lifecycle and continuation store.

## Context invariants

Each context starts from the previous `FinalContext`, or empty state. Append
order affects the deterministic digest:

| Context | Append order | Purpose |
| --- | --- | --- |
| Planning | Instructions, input. | Tool selection; `PlanTools` hashes tool configuration separately. |
| Generation | Tool configuration, instructions, input. | Content and input usage. |
| Carry | Tool configuration, input, actual output. | Next turn's state. |

Instructions must not carry forward. Tool configuration contributes to the digest
but does not supply tool definitions to later calls. Preserve request-validation
versus internal-error classification in `prepareResponseContexts`.

## Structured text and storage

`internal/jsonschema` compiles immutable, bounded schemas. The optional
`chat.StructuredModel` capability plans a full document before applying the
output ceiling; streaming borrows that plan. `textGeneration` binds it to the
shared orchestration. Validate formats even when tool selection chooses calls;
models without the capability reject structured requests with HTTP 400.

Truncation produces a normal terminal incomplete result. Its actual output
contributes to stored context; the format is not inherited. Snapshots retain
digests and estimates, not text. Store complete/incomplete state before sending
the terminal SSE event, and remove it if delivery fails. Cancellation and failed
streams must never leave a continuable snapshot.

HTTP continuation and SSE tests cover instruction lifetime, tool accounting,
JSON/SSE agreement, and terminal-write failure.
