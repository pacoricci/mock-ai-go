# Structured output

Chat Completions uses `response_format`; Responses uses `text.format`.
Both accept `text` (default), `json_object`, or `json_schema`. Output is synthetic
and returned as a string in the normal text field, including over SSE.

## Formats

Add this to a Chat request:

```json
{
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "order",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
        "additionalProperties": false
      }
    }
  }
}
```

For Responses, put `type`, `name`, `strict`, and `schema` directly inside
`text.format`, without the `json_schema` wrapper.
Names use 1–64 ASCII letters, digits, underscores, or hyphens; `description` is
optional and `strict` defaults to false. Responses echoes the format and ignores
`text.verbosity`.

`{"type":"json_object"}` generates an object with a string `message` and integer
`value`. It does not interpret prompt instructions or require the word JSON.

## Supported schema subset

The root must resolve to a non-nullable object. Supported features are:

- Objects with `properties`, `required`, and boolean `additionalProperties`.
  All declared properties are generated, including optional ones. No additional
  properties are synthesized. Property order follows the supplied schema.
- Arrays with a single `items` schema and `minItems`/`maxItems`. The generator
  chooses one item unless a bound requires a different count.
- Strings, integers, numbers, booleans, and null. A type may also be a two-element
  array combining one type with `null`.
- Primitive `enum` and `const` values, including Unicode strings. Every listed
  value must satisfy the declared type and bounds; a constant must belong to
  the enum when both are present.
- `minimum`/`maximum`, `minLength`/`maxLength` (Unicode character counts), and
  nested `anyOf` branches. The generator chooses one supported branch.
- Acyclic local references beneath `#/$defs/`, including JSON Pointer escapes.
  Definitions are checked even when unused. Remote and recursive refs are rejected.
- Annotations `title`, `description`, `default`, `examples`, and `$schema` are
  accepted as metadata; defaults and examples do not select generated values.

With `strict: true`, every object must declare `additionalProperties: false`
and list every property in `required`. Nullable types model optional values.
Both strict and non-strict successful, complete outputs satisfy this subset.
Unsupported constraints are rejected in both modes, rather than silently ignored.
Examples include `pattern`, `format`, `oneOf`, `allOf`, `not`, conditionals,
`multipleOf`, `uniqueItems`, and schema-valued `additionalProperties`.
Validation siblings of `$ref` and `anyOf` are unsupported.

Resource bounds are 64 KiB of encoded schema, depth 32, 4096 compiler node visits
(including expanded references), 256 properties per object, 256 array items,
4096 characters per string, 256 enum values or union branches, numeric bounds
and enum values within +/-1e12, and 1 MiB of generated JSON. Bounds outside
these limits, contradictory constraints, unknown required properties, malformed
schemas, and unsupported features return HTTP 400 before SSE starts. Errors
identify the schema parameter and include a JSON Pointer diagnostic.

## Budgets and streaming

Output limits are ceilings: a complete document may finish early. Chat accepts
at least one token; Responses requires at least 16. Limits above configured
capacity are rejected.

Generation builds the document, then emits at most `4 × budget` UTF-8 bytes,
backing up to a character boundary. Usage is `ceil(emitted bytes / 4)`; prompt
usage includes the format. Increasing the ceiling preserves the prefix for the
same context, format, model, and Chat seed. Pacing/chunk size do not affect it.
With tools, this applies to the text branch; selection also depends on budget.

| Result | Chat | Responses |
| --- | --- | --- |
| Full document fits | `finish_reason: "stop"` | `status: "completed"` |
| Truncated document | `finish_reason: "length"` | `status: "incomplete"`, reason `max_output_tokens` |

Truncation returns HTTP 200 and may leave invalid partial JSON, even in strict
mode. The document is never repaired; the outer response remains valid.

JSON and reassembled SSE text/usage agree. Truncated Chat sends a `length` choice,
optional usage, then `[DONE]`. Responses sends text/part/item done events,
`response.incomplete`, and `[DONE]`, with an incomplete message and null
`completed_at`; it does not send `response.completed`.

## Tools and continuation

Formats affect assistant text, not tool arguments, and are validated even if
tools are selected. Supply the format again after tool results or on continuation;
it is not inherited or carried into the next digest.

Terminal incomplete Responses are continuable when stored. Their partial output
contributes to context, but the next generation creates a new document rather
than appending missing bytes. Cancellation, failures, and undelivered terminal
events are not stored. See [continuation](openai-api.md#process-local-continuation).

Anthropic structured output, full JSON Schema, semantic extraction, and refusals
are unsupported. See [SDK coverage](sdk-compatibility.md).
