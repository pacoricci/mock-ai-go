# Anthropic Messages

`POST /v1/messages` supports text and deterministic client tools in JSON or SSE.
Use SDK base URL `http://localhost:8080`, a dummy key, and header
`anthropic-version: 2023-06-01`. Missing/other versions and nonempty
`anthropic-beta` return 400. Authentication headers are ignored.

## Request

| Field | Mock behavior |
| --- | --- |
| `model` | Required registered model ID, normally `mock-chat-model`; unknown IDs return 404. No automatic Claude aliases. |
| `max_tokens` | Required positive integer within `MAX_COMPLETION_TOKENS`. Zero/cache warming is unsupported. |
| `messages` | Nonempty history of `user` and `assistant` messages. Must start and end with `user`; assistant prefill is unsupported. Consecutive messages with the same role are combined into one turn. |
| `content` | Nonempty text string or nonempty list of supported blocks: `text`, assistant `tool_use`, user `tool_result`. Text blocks must be nonempty. |
| `system` | Optional nonempty string or nonempty list of text blocks. Participates in deterministic context and usage. |
| `stream` | Boolean, default false; true selects SSE. |
| `tools` | Unique client-tool names, descriptions, and object `input_schema` definitions. Names use 1-64 ASCII letters, digits, underscores, or hyphens. `type` may be omitted or `custom`. |
| `tool_choice` | `auto` (default), `none`, `any`, or `tool` with a matching `name`. `disable_parallel_tool_use` is nested here. |
| `temperature`, `top_p`, `top_k` | Accepted without affecting generation; validated as 0-1, 0-1, and nonnegative integer respectively. |
| `metadata.user_id` | Accepted without affecting generation or usage. |
| `stop_sequences` | Omitted or empty only; nonempty sequences are unsupported. |


Unknown fields and unsupported blocks return 400. Thinking, images, documents,
citations, caching, structured output, server tools, toolsets, and beta features
are unsupported. `strict: true` is rejected; tool schemas are best-effort.
There is no `seed` field. Plain-text length is sampled independently for each
generation. Tool plans remain reproducible for identical normalized context,
tools, model, and budget; message IDs and timing are independent.

## Text and tools

Text samples 128–512 mock words by default, capped by `max_tokens`. Limits below
128 produce exactly that many words. It ends with `end_turn` when output is
shorter than the limit, or `max_tokens` when the limit is reached.
Stop sequences, refusals, and reasoning are not simulated.
JSON and SSE use the same sampling rules; separate requests may produce different
lengths. Usage counts the words actually emitted and is not tokenizer output.
See [configuration](configuration.md#text-length) to adjust or disable sampling.

`none` selects text, `any` requires tools, `tool` selects the named tool, and
`auto` deterministically chooses text or tools. After user tool results, `auto`
always selects text to terminate tool loops. Parallel selection defaults on;
disabling it limits output to one call. Each tool is selected at most once.

Tool output contains `tool_use` blocks with IDs, names, and object-valued `input`,
and ends with `tool_use`. Functions are never executed. If the whole plan exceeds
`max_tokens`, the request returns 400 before streaming instead of truncating JSON.

Resend the full history each turn. Follow an assistant tool turn immediately with
a user turn containing one `tool_result` per `tool_use_id`, before any text.
Results may be reordered; missing, unknown, or repeated IDs are rejected.
Result content accepts text/text blocks and may be omitted or empty. `is_error`
affects context. There is no conversation store.

## Streaming and errors

```text
message_start
  content_block_start
  content_block_delta (text_delta or input_json_delta, repeated)
  content_block_stop
message_delta
message_stop
```

Events have matching JSON `type` fields and contiguous block indices. Tool
fragments are UTF-8-safe; concatenate them to recover `input`. Initial usage has
zero output tokens; `message_delta` carries cumulative usage and the stop reason.
`stop_sequence` is null. No `[DONE]` or periodic ping is emitted.
[Shared pacing](configuration.md) applies; writes respect backpressure and
cancellation. JSON responses are unpaced.

Every response has `request-id`. Error bodies have this shape:

```json
{"type":"error","error":{"type":"invalid_request_error","message":"…"},"request_id":"…"}
```

| Status | Error type | Trigger |
| --- | --- | --- |
| 400 | `invalid_request_error` | Invalid/unsupported request, version, or token budget. |
| 404 | `not_found_error` | Unknown model or unsupported Messages subpath. |
| 405 | `invalid_request_error` | Wrong method, with `Allow: POST`. |
| 413 | `request_too_large` | Body exceeds the mock's 1 MiB limit. |
| 500 | `api_error` | Model failure before streaming starts. |


Validation occurs before SSE headers. After streaming starts, model failures emit
`event: error` without `message_stop`; HTTP status remains 200. Cancellation and
write failure stop generation without a success event. Request/message IDs are
unique within a handler lifetime; tool IDs are deterministic.

Count-token, batch, and other Messages subpaths return 404. Bedrock, Vertex,
and `ssebench` support for Messages are unavailable.
See [compatibility tests](sdk-compatibility.md) for SDK checks.
