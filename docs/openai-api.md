# OpenAI-compatible APIs

Use `http://localhost:8080/v1` as the SDK base URL, a dummy key, and
`mock-chat-model`. See [OpenAPI](openapi.yaml) for request fields and errors.
Plain text defaults to 128–512 words, capped by the request's output-token limit.
Chat Completions reproduces length and content for the same request with an explicit
`seed`; JSON and SSE agree. Without a seed, repeated generations may differ in
length and content. Responses does not expose a seed field.
See [text length configuration](configuration.md#text-length) for fixed-size mode.

## Chat Completions

```sh
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"seed":42,"stream":true,"stream_options":{"include_usage":true}}'
```

Omit `stream` for JSON. SSE ends with a finish choice, optional usage in an
empty-choices chunk, and `[DONE]`.

Chat completion IDs are opaque and include a random handler namespace, so
separate processes and restarts do not reuse IDs. Every chunk of one stream
shares its response ID. The namespace does not affect generated text or pacing.

Tool responses contain `message.content: null`, `message.tool_calls`, and
`finish_reason: "tool_calls"`. Replay the assistant message followed by one
`{"role":"tool","tool_call_id":"…","content":"…"}` per result. Content must
be a string. Results may be reordered after their calls; unknown IDs, duplicate
calls/results, and results preceding calls are rejected. Use `tool_choice:
"none"` for a text follow-up. Legacy `functions` and `function_call` are rejected.

Streaming calls begin with an index, ID, type, and name. Concatenate subsequent
`delta.tool_calls` argument fragments by index before parsing JSON.

## Responses

```sh
curl http://localhost:8080/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-chat-model","input":"Hello","max_output_tokens":16}'
```

Add `"stream": true` and use `curl -N` for SSE. Explicit `max_output_tokens`
must be at least 16.

`input` accepts a string or ordered messages (`user`, `system`, `developer`,
`assistant`), `function_call`, `function_call_output`, and supplied `reasoning`
items. Message content accepts strings or text parts; non-text parts are ignored.
`instructions` applies only to the current generation and usage. `metadata` is
returned but does not affect continuation.

Reasoning generation, multimodal generation, sampling controls, background mode,
prompt caching, and service-tier selection are not simulated. Responses report
foreground execution and the `default` tier. Compaction, retrieval, and WebSocket
transport are unsupported.

### Process-local continuation

Pass `previous_response_id` with the same model to continue a complete or terminal
incomplete Response. `store` defaults to true. With `store: false`, resend the
relevant history; the returned ID cannot be continued.

State is lost on restart and is not shared between replicas. Snapshots contain
context digests, usage estimates, and emitted call IDs, not conversation text.
Tool configuration contributes to context; instructions do not carry forward.
Failed, cancelled, or interrupted streams are never stored. A failed write of
`response.completed` or `response.incomplete` removes the pending snapshot.

The store uses an absolute TTL and FIFO eviction. Reads do not renew expiry.
Entry and logical-byte limits cover snapshots, not in-flight requests; see
[configuration](configuration.md).

| HTTP 400 code | Cause |
| --- | --- |
| `invalid_value` | Malformed ID or model mismatch. |
| `previous_response_not_found` | Unstored, expired, evicted, or restart-invalidated ID. |

Both identify `previous_response_id` as the parameter. IDs are opaque
`resp_mock_` values followed by 32 lowercase hexadecimal characters.

## Function calling

The mock returns calls but never executes them. For Chat Completions, add:

```json
{
  "tool_choice": "required",
  "parallel_tool_calls": false,
  "tools": [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {
      "type": "object",
      "properties": {"location": {"type": "string"}},
      "required": ["location"]
    }
  }}]
}
```

For Responses, place `name` and `parameters` directly beside `type`, without
the `function` wrapper.

| `tool_choice` | Behavior |
| --- | --- |
| `none` | Text. |
| `auto` | Text without tools; deterministic 50/50 text-or-calls selection with tools. |
| `required` | Calls; requires at least one tool. |
| Chat `{"type":"function","function":{"name":"get_weather"}}` | Restricts selection to that function. |

Parallel selection defaults on and emits distinct calls. Setting
`parallel_tool_calls: false` limits selection to one; Responses also accepts
`max_tool_calls` as a cap.

Arguments support objects, required properties, primitives, enums, constants,
arrays, and simple `oneOf`/`anyOf` branches. Advanced constraints are best-effort:
`strict: true` does not enforce them; Responses returns `strict: false`.
[Structured text](structured-output.md) uses a separate, validated schema subset.

JSON and SSE share the call plan. Arguments stream sequentially in UTF-8-safe
fragments, about four bytes per configured token. Responses emits
`response.function_call_arguments.delta` and `.done`. Pacing and chunk size do
not change call IDs, arguments, or usage.

## Simulated usage

Usage is not tokenizer output. Text output counts generated words; structured
text counts approximately four UTF-8 bytes per token. Tools count function names,
complete arguments, and call overhead, independently of chunk boundaries.

Plain Chat input uses byte length plus message overhead. Tool histories use
per-item estimates including tool configuration, call IDs, and results. Responses
accumulates those estimates across continuations, including prior output but
excluding previous instructions. Cached and generated reasoning tokens are zero.
