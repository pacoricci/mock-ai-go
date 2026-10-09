# Compatibility tests

Official SDK tests use local HTTP/SSE servers and dummy keys; no provider access
is needed. Their dependencies live in separate Go modules under `tests/`.

```sh
make check                 # Service and both SDK suites, plus vet.
make lint
make test-openai-sdk       # OpenAI only.
make test-anthropic-sdk    # Anthropic only.
```

Root `go test ./...` covers only the service module. Run `go test -race ./...`
inside either SDK module for its race checks. The first run downloads dependencies.

## Covered contracts

| Suite | Pinned SDK | Coverage |
| --- | --- | --- |
| `tests/openai-sdk` | `openai-go/v3` v3.68.0 | Chat, Responses, tools, structured text, embeddings, and typed errors. |
| `tests/anthropic-sdk` | `anthropic-sdk-go` v1.76.0 | Messages, tools, replay, typed errors, and cancellation. |

Both check JSON/SSE content and usage agreement, event ordering, Unicode argument
fragments, parallel and named tools, replay with reordered results, concurrent
histories, and failure/cancellation behavior.
SDK fixtures use fixed-size text generation. Service HTTP tests additionally
cover sampled word counts and request ceilings across all three text APIs in
JSON and SSE, seeded Chat agreement, and Anthropic terminal reasons.

OpenAI tests additionally cover token boundaries, schema validation, continuation
across JSON/SSE, branching, store expiry/eviction/model isolation, stateless
responses, and interrupted streams that must not persist. Embedding tests cover
batch determinism, dimensions, encodings, usage, and invalid inputs. Base64 uses
a raw response override because the pinned SDK's embedding field is numeric.

These checks establish the documented subset with these SDK versions, not full
provider parity or compatibility with other clients. To update an SDK, change its
nested module, run `go mod tidy` there, rerun checks, and commit `go.mod` and
`go.sum` together.

## Open Responses acceptance suite

The Bun suite is pinned in the `third_party/openresponses-compliance` submodule.
Install it once:

```sh
git submodule update --init --recursive
cd third_party/openresponses-compliance
bun install --ignore-scripts
```

With the server running, execute from the repository root:

```sh
make openresponses-compliance
```

The default filter covers `basic-response`, `assistant-phase`,
`response-output-phase-schema`, `streaming-response`, `system-prompt`,
`tool-calling`, `image-input`, and `multi-turn`. Image input is ignored by this
text-only mock. Compaction and seven WebSocket tests are excluded.
Override `OPENRESPONSES_FILTER`, `OPENRESPONSES_BASE_URL`,
`OPENRESPONSES_API_KEY`, or `OPENRESPONSES_MODEL` as needed.
