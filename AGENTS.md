## Project overview

`mock-ai-go` is a small Go service that mocks LLM provider APIs:
OpenAI-compatible Chat Completions, Responses, and embeddings, plus Anthropic
Messages. It supports JSON and SSE responses, deterministic pseudo-random
output, function calling, structured output, configurable stream pacing,
private runtime diagnostics, and a separate black-box SSE load generator.

Keep the service lightweight. Add a package or dependency only when it has a
clear responsibility that the standard library or an existing package does not
already cover.

---

## Code Style

- Follow [Go Code Review Comments](https://go.dev/wiki/CodeReviewComments)
  and the formatting/style section of
  [Go: Best Practices for Production Environments](https://peter.bourgon.org/go-in-production/#formatting-and-style).
- State your assumptions.
- Interface contracts: when ownership or lifetime semantics are important,
  document it at the interface definition, not just in the implementation.
- All exposed objects must have a doc comment.
- All comments must start with a capital letter and end with a full stop.
- Run `make lint` before submitting. The project uses `golangci-lint` including
  `gocritic` rules such as `emptyStringTest` — fix linter findings rather than
  suppressing them with `//nolint` unless there is a clear false-positive.
- Use `//nolint:linter1[,linter2,...]` sparingly; prefer fixing the code.

---

## Tests

- Bug fixes require a test that reproduces the bug.
- New behaviour or exported API changes require unit or e2e tests.
- Tests should attempt to mirror realistic data and/or behaviour.
- Use only exported APIs in tests where possible — this keeps tests closer to
  real library usage and simplifies review.

---

## Documentation and scope

- When changing documented behaviour, update any relevant text in the docs/ directory.
  Check whether the OpenAPI spec also needs updating.
