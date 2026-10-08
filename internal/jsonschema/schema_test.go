package jsonschema_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/jsonschema"
)

const orderSchema = `{
 "type":"object","additionalProperties":false,"required":["id","lines","note","state"],
 "properties":{
   "id":{"type":"integer","minimum":4.2,"maximum":7.8},
   "lines":{"type":"array","minItems":2,"maxItems":3,"items":{"$ref":"#/$defs/line"}},
   "note":{"type":["string","null"],"minLength":2,"maxLength":8},
   "state":{"anyOf":[{"type":"string","enum":["ready","München 世界"]},{"type":"null"}]}
 },
 "$defs":{"line":{"type":"object","additionalProperties":false,"required":["sku","price"],"properties":{
   "sku":{"type":"string","const":"SKU-42"},"price":{"type":"number","minimum":1.5,"maximum":2.5}
 }}}
}`

func TestGenerateSupportedSchema(t *testing.T) {
	t.Parallel()
	schema, err := jsonschema.Compile([]byte(orderSchema), true)
	if err != nil {
		t.Fatal(err)
	}
	for seed := range uint64(30) {
		state := seed
		raw, err := schema.Generate(context.Background(), func() uint64 { state = state*6364136223846793005 + 1; return state })
		if err != nil {
			t.Fatal(err)
		}
		var result struct {
			ID    int `json:"id"`
			Lines []struct {
				SKU   string  `json:"sku"`
				Price float64 `json:"price"`
			} `json:"lines"`
			Note  *string `json:"note"`
			State *string `json:"state"`
		}
		decoder := json.NewDecoder(strings.NewReader(string(raw)))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&result); err != nil {
			t.Fatal(err)
		}
		if result.ID < 5 || result.ID > 7 || len(result.Lines) < 2 || len(result.Lines) > 3 {
			t.Fatalf("bounds violated: %s", raw)
		}
		for _, line := range result.Lines {
			if line.SKU != "SKU-42" || line.Price < 1.5 || line.Price > 2.5 {
				t.Fatalf("line violates schema: %s", raw)
			}
		}
		if result.Note != nil && (len(*result.Note) < 2 || len(*result.Note) > 8) {
			t.Fatalf("note violates schema: %s", raw)
		}
		if result.State != nil && *result.State != "ready" && *result.State != "München 世界" {
			t.Fatalf("state violates schema: %s", raw)
		}
		if !strings.HasPrefix(string(raw), `{"id":`) || strings.Index(string(raw), `"lines":`) > strings.Index(string(raw), `"note":`) {
			t.Fatalf("schema property order lost: %s", raw)
		}
	}
}

func TestRejectUnsupportedAndContradictorySchemas(t *testing.T) {
	t.Parallel()
	cases := map[string]string{
		"null required entry":    `{"type":"object","properties":{"":{"type":"string"}},"required":[null],"additionalProperties":false}`,
		"invalid pointer escape": `{"type":"object","properties":{"x":{"$ref":"#/$defs/a~2"}},"required":["x"],"additionalProperties":false,"$defs":{"a~2":{"type":"string"}}}`,
		"missing type":           `{}`,
		"root array":             `{"type":"array","items":{"type":"string"}}`,
		"strict properties":      `{"type":"object","properties":{"x":{"type":"string"}},"additionalProperties":false}`,
		"strict additional":      `{"type":"object","required":[],"properties":{}}`,
		"unknown required":       `{"type":"object","required":["x"],"additionalProperties":false}`,
		"recursive":              `{"type":"object","required":["x"],"additionalProperties":false,"properties":{"x":{"$ref":"#/$defs/node"}},"$defs":{"node":{"$ref":"#/$defs/node"}}}`,
	}
	for name, property := range map[string]string{
		"duplicate enum":              `{"type":"number","enum":[1,1.0]}`,
		"const conceals invalid enum": `{"type":"string","const":"valid","enum":["valid",42]}`,
		"pattern":                     `{"type":"string","pattern":"[0-9]+"}`,
		"format":                      `{"type":"string","format":"date-time"}`,
		"string bounds":               `{"type":"string","minLength":4,"maxLength":3}`,
		"size limit":                  `{"type":"array","minItems":257,"items":{"type":"null"}}`,
		"numeric bounds":              `{"type":"integer","minimum":5,"maximum":4}`,
		"empty integer interval":      `{"type":"integer","minimum":0.1,"maximum":0.9}`,
		"enum type":                   `{"type":"integer","enum":["oops"]}`,
		"enum bounds":                 `{"type":"integer","minimum":2,"enum":[1,2]}`,
		"const enum conflict":         `{"type":"string","enum":["a"],"const":"b"}`,
		"empty enum":                  `{"type":"string","enum":[]}`,
		"null enum":                   `{"type":"string","enum":[null]}`,
		"no items":                    `{"type":"array"}`,
		"missing ref":                 `{"$ref":"#/$defs/missing"}`,
		"external ref":                `{"$ref":"https://example.com/schema"}`,
		"ref sibling":                 `{"$ref":"#/$defs/x","type":"string"}`,
		"anyOf sibling":               `{"anyOf":[{"type":"string"}],"minLength":2}`,
		"wrong keyword":               `{"type":"string","minimum":2}`,
		"null bound":                  `{"type":"integer","minimum":null}`,
		"duplicate property":          `{"type":"object","properties":{"a":{"type":"string"},"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`,
	} {
		cases[name] = fmt.Sprintf(`{"type":"object","required":["x"],"additionalProperties":false,"properties":{"x":%s}}`, property)
	}
	for name, raw := range cases {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			_, err := jsonschema.Compile([]byte(raw), true)
			var schemaError *jsonschema.Error
			if !errors.As(err, &schemaError) || !strings.HasPrefix(schemaError.Path, "#") {
				t.Fatalf("expected schema error, got %v", err)
			}
		})
	}
}

func TestGenerationLimitsAndCancellation(t *testing.T) {
	t.Parallel()
	property := `{"type":"string","minLength":4096}`
	for range 3 {
		property = `{"type":"array","minItems":256,"items":` + property + `}`
	}
	schema, err := jsonschema.Compile([]byte(`{"type":"object","properties":{"x":`+property+`}}`), false)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := schema.Generate(context.Background(), func() uint64 { return 1 }); err == nil {
		t.Fatal("oversized output succeeded")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := schema.Generate(ctx, func() uint64 { return 1 }); !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled generation: %v", err)
	}
	if _, err := jsonschema.Compile([]byte(strings.Repeat(" ", 65537)), false); err == nil {
		t.Fatal("oversized schema succeeded")
	}
}

func TestNonStrictOptionalAndScalarConstants(t *testing.T) {
	t.Parallel()
	schema, err := jsonschema.Compile([]byte(`{"type":"object","properties":{"empty":{"type":"string","maxLength":0},"count":{"type":"integer","enum":[10000],"const":10000},"flag":{"type":"boolean","const":false},"nothing":{"type":["string","null"],"const":null}}}`), false)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := schema.Generate(context.Background(), func() uint64 { return 1 })
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"empty":"","count":10000,"flag":false,"nothing":null}` {
		t.Fatalf("unexpected constants: %s", raw)
	}
}

func TestSchemaOwnershipAndEscapedReferences(t *testing.T) {
	t.Parallel()
	raw := []byte(`{"type":"object","properties":{"first":{"$ref":"#/$defs/a~1b~0c"}},"$defs":{"a/b~c":{"type":"string","const":"owned"}}}`)
	schema, err := jsonschema.Compile(raw, false)
	if err != nil {
		t.Fatal(err)
	}
	clear(raw)
	for range 4 {
		t.Run("concurrent generation", func(t *testing.T) {
			t.Parallel()
			result, err := schema.Generate(context.Background(), func() uint64 { return 1 })
			if err != nil || string(result) != `{"first":"owned"}` {
				t.Fatalf("schema retained caller storage: %s, %v", result, err)
			}
		})
	}
	if _, err := jsonschema.Compile([]byte(`{"type":"object","$defs":{"unused":{"type":"string","pattern":".*"}}}`), false); err == nil {
		t.Fatal("unsupported unused definition accepted")
	}
}

func FuzzCompileAndGenerate(f *testing.F) {
	f.Add(orderSchema, true)
	f.Add(`{"type":"object","properties":{"x":{"type":"string","enum":["世界",null]}}}`, false)
	f.Add(`{"type":"object","$defs":{"recursive":{"$ref":"#/$defs/recursive"}}}`, false)
	f.Fuzz(func(t *testing.T, raw string, strict bool) {
		schema, err := jsonschema.Compile([]byte(raw), strict)
		if err != nil {
			return
		}
		output, err := schema.Generate(context.Background(), func() uint64 { return 7 })
		if err == nil && !json.Valid(output) {
			t.Fatalf("successful generation produced invalid JSON: %q", output)
		}
	})
}
