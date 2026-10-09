# Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `mock-ai-go` | Startup name. |
| `HTTP_ADDR` | `:8080` | HTTP listen address. |
| `PPROF_ADDR` | empty | Private pprof/metrics listener, e.g. `127.0.0.1:6060`. |
| `MODEL_ID` | `mock-chat-model` | Model for Chat, Responses, and Messages. |
| `EMBEDDING_MODEL_ID` | `mock-embedding-model` | Separate model ID for embeddings. |
| `EMBEDDING_DIMENSIONS` | `1536` | Default and maximum requested embedding length, from 1 through 4096. |
| `DEFAULT_COMPLETION_TOKENS` | `32` | Budget when a request omits its output-token limit, for structured output or plain text with length sampling disabled. |
| `MAX_COMPLETION_TOKENS` | `4096` | Maximum generated words per request. |
| `TEXT_MIN_TOKENS` | `128` | Minimum sampled plain-text word count, capped by the request's output-token limit. |
| `TEXT_MAX_TOKENS` | `512` | Maximum sampled plain-text word count. |
| `SSE_TOKENS_PER_CHUNK` | `4` | Generated words in each SSE text delta. |
| `SSE_DELAY` | `100ms` | Base delay between SSE content chunks. |
| `SSE_DELAY_JITTER` | `20ms` | Uniform jitter applied to each content gap; negative results are clamped to zero. |
| `SSE_INITIAL_DELAY` | `800ms` | Delay before response headers and the first SSE event. |
| `RESPONSE_STORE_TTL` | `1h` | Absolute lifetime of a stored terminal Response; reads do not renew it. |
| `RESPONSE_STORE_MAX_ENTRIES` | `10000` | Maximum number of process-local Response snapshots. |
| `RESPONSE_STORE_MAX_BYTES` | `67108864` | Maximum logical bytes across process-local Response snapshots. |

Response-store settings must be positive and well formed or startup fails.

## Text length

Plain text samples a uniform word count from the inclusive range
`TEXT_MIN_TOKENS`–`TEXT_MAX_TOKENS`, by default 128–512 words (mean 320).
Request limits (`max_completion_tokens`, its legacy alias `max_tokens`, or
Responses `max_output_tokens`) cap the upper bound. A limit below the configured
minimum produces exactly that many words. Anthropic requires `max_tokens` and
applies the same ceiling. Limits above `MAX_COMPLETION_TOKENS` are rejected.

Without a request limit, plain text uses the full configured range.
With an explicit Chat Completions `seed`, identical requests reproduce both
length and content, including across JSON and SSE. Without a seed, each
generation independently samples its length, so identical requests may differ.
Responses and Anthropic do not expose a seed field.

Both bounds must be positive, ordered, and at most `MAX_COMPLETION_TOKENS`.
Set both to zero to disable sampling and restore fixed output sizes: the request
limit, or `DEFAULT_COMPLETION_TOKENS` when omitted. Equal positive bounds give a
fixed word count capped by the request limit. Structured output and tool planning
keep their existing budget and deterministic generation semantics.

## Streaming

Defaults give an 800 ms initial wait and 80–120 ms content gaps. Timing is
independent of content and decorrelated across concurrent streams. For fixed gaps,
set jitter and initial delay to `0s`. To disable pacing:

```sh
SSE_INITIAL_DELAY=0s SSE_DELAY=0s SSE_DELAY_JITTER=0s make run
```

JSON responses are unpaced. See [timing estimates](benchmarks.md#timing-and-metrics).

## Docker

`make docker-run` builds and runs the non-root image on `127.0.0.1:8080`.
`make docker-up` uses Compose; set `HTTP_HOST_PORT=8081` to change its host port.
Both publish the port on the host loopback interface only.
Pass server variables with `docker run --env` or Compose `environment`.

## Network access

The mock does not authenticate requests; dummy API keys are for SDK compatibility
only. Native runs listen on all interfaces by default (`HTTP_ADDR=:8080`); use
`HTTP_ADDR=127.0.0.1:8080 make run` to restrict access to the local machine.
Use the service in a trusted test environment. Keep the optional diagnostics
listener on loopback as well.
