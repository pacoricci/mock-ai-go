# Embeddings

`POST /v1/embeddings` returns deterministic synthetic vectors with no semantic
similarity. Use them to test SDKs, ingestion, and vector storage.

```sh
curl http://localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"mock-embedding-model","input":["Hello café","Second document"],"dimensions":256}'
```

## Request

| Field | Accepted values |
| --- | --- |
| `model` | Configured embedding model ID; unknown IDs return 404. |
| `input` | Nonempty string or nonempty array of nonempty strings. Whitespace-only strings are accepted. |
| `dimensions` | 1 through configured `EMBEDDING_DIMENSIONS`; omitted/null uses that setting. |
| `encoding_format` | `float` (default) or `base64`; null uses the default. |
| `user` | String or null; ignored. |

Token arrays, mixed arrays, null items, and unknown fields (including `seed` and
`stream`) return 400. Responses are JSON and unpaced. Errors use the OpenAI
envelope: `model_not_found` for lookup, `invalid_value` for parameter validation.
Malformed/oversized bodies return 400; wrong methods return 405.

## Bounds

[Configuration](configuration.md) defines the model and maximum dimensions.
Empty settings use defaults; whitespace-only IDs or dimensions outside 1–4096
fail startup. Programmatic `config.Config` also accepts empty ID/zero dimensions
for defaults.

| Per-request limit | Value |
| --- | --- |
| JSON body | 1 MiB. |
| Inputs | 2048 strings. |
| Input length | 32768 UTF-8 bytes each. |
| Total components | 1048576 (`input count × dimensions`). |

A bad item rejects the entire batch. These bounds apply per request, not globally.

## Output

`data` preserves input order with zero-based indices. Vectors have unit L2 norm
within float32 tolerance. Base64 encodes the same float32 components as float
JSON, in little-endian byte order.

SHA-256 generation depends on model ID, exact input bytes, and dimensions.
Batch position, concurrency, restarts, `user`, and encoding do not affect it.
Text is not normalized. Smaller dimensions renormalize components, so shorter
vectors are not prefixes of longer ones. Golden tests pin the algorithm.

`usage.prompt_tokens` and `usage.total_tokens` equal the sum of
`ceil(UTF-8 bytes / 4)` per input. These are estimates, not tokenizer counts.
See [SDK tests](sdk-compatibility.md) for encoding coverage.
