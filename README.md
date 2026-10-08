# mock-ai-go

A lightweight Go server for testing LLM clients without calling a provider.
Mocks OpenAI-compatible Chat Completions, Responses, and embeddings, plus
Anthropic Messages. The service uses only the Go standard library.

Supports deterministic output, JSON and SSE, function calling, structured
output, and configurable streaming delays. Content and usage are synthetic;
embeddings carry no semantic similarity, and functions are never executed.

## Quick start

With Go 1.26 or later:

```sh
make run
```

Or with Docker:

```sh
make docker-up
```

In another terminal:

```sh
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}]}'
```

Add `"stream": true` and use `curl -N` for SSE.

## Connect your client

| API | Base URL | Default model |
| --- | --- | --- |
| OpenAI Chat Completions / Responses | `http://localhost:8080/v1` | `mock-chat-model` |
| OpenAI embeddings | `http://localhost:8080/v1` | `mock-embedding-model` |
| Anthropic Messages | `http://localhost:8080` | `mock-chat-model` |

Use a dummy API key. Supported features and limitations are documented below;
this mock does not implement the full provider APIs.

## Documentation

- [OpenAI APIs](docs/openai-api.md): requests, function calls, continuation, and usage.
- [Anthropic Messages](docs/anthropic-messages.md), [embeddings](docs/embeddings.md), and [structured output](docs/structured-output.md): supported contracts.
- [Configuration](docs/configuration.md): environment variables, streaming delays, and Docker.
- [Benchmarks](docs/benchmarks.md): HTTP/SSE load testing with `ssebench` and Go microbenchmarks.
- [Compatibility tests](docs/sdk-compatibility.md): official Go SDKs and Open Responses acceptance tests.
- [OpenAPI specification](docs/openapi.yaml), also available through [Swagger UI](http://localhost:8080/docs) while the server runs.

## Development

```sh
make check  # Tests and vet, including SDK suites.
make lint
make build  # Writes bin/mock-ai-go.
```

See the [Responses implementation guide](docs/responses-implementation.md) for
request flow and conversation-state invariants.

## License

[MIT](LICENSE). Third-party components retain their own licenses.
