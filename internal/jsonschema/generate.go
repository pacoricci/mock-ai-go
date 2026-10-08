package jsonschema

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

func (n *node) readValues(fields map[string]json.RawMessage, path string) error {
	constant, hasConstant := fields["const"]
	enum, hasEnum := fields["enum"]
	if !hasConstant && !hasEnum {
		return nil
	}
	if n.kind == "object" || n.kind == "array" {
		return schemaError(path, "const and enum support primitive values only")
	}
	if hasEnum {
		if json.Unmarshal(enum, &n.values) != nil || len(n.values) == 0 || len(n.values) > 256 {
			return schemaError(pointer(path, "enum"), "expected 1 to 256 values")
		}
		seen := make(map[any]bool)
		for _, raw := range n.values {
			if !n.acceptsValue(raw, fields) {
				return schemaError(pointer(path, "enum"), "value violates type or bounds")
			}
			var value any
			_ = json.Unmarshal(raw, &value)
			if seen[value] {
				return schemaError(pointer(path, "enum"), "duplicate value")
			}
			seen[value] = true
		}
	}
	if hasConstant {
		if !n.acceptsValue(constant, fields) {
			return schemaError(pointer(path, "const"), "value violates type or bounds")
		}
		if hasEnum {
			var value any
			_ = json.Unmarshal(constant, &value)
			found := false
			for _, raw := range n.values {
				var candidate any
				_ = json.Unmarshal(raw, &candidate)
				// Only primitive values are allowed, so decoded values are comparable.
				if candidate == value {
					found = true
				}
			}
			if !found {
				return schemaError(pointer(path, "const"), "constant is absent from enum")
			}
		}
		n.values = []json.RawMessage{constant}
	}
	return nil
}

func (n *node) acceptsValue(raw json.RawMessage, fields map[string]json.RawMessage) bool {
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return false
	}
	if value == nil {
		return n.kind == "null" || n.nullable
	}
	switch v := value.(type) {
	case string:
		size := utf8.RuneCountInString(v)
		return n.kind == "string" && size >= n.minSize && size <= n.maxSize
	case bool:
		return n.kind == "boolean"
	case float64:
		if n.kind != "integer" && n.kind != "number" || math.Abs(v) > 1e12 {
			return false
		}
		if n.kind == "integer" && math.Trunc(v) != v {
			return false
		}
		return (fields["minimum"] == nil || v >= n.minimum) && (fields["maximum"] == nil || v <= n.maximum)
	default:
		return false
	}
}

// Generate returns a fresh JSON document in schema property order. next is owned
// by the caller, is called synchronously, and must yield deterministic random bits.
// Generation is cancellable and bounded to one MiB of encoded output.
func (s *Schema) Generate(ctx context.Context, next func() uint64) ([]byte, error) {
	if s == nil || s.root == nil || next == nil {
		return nil, schemaError("#", "missing compiled schema or random source")
	}
	g := generator{ctx: ctx, next: next}
	if err := g.value(s.root); err != nil {
		return nil, err
	}
	return g.output, nil
}

type generator struct {
	ctx    context.Context
	next   func() uint64
	output []byte
}

func (g *generator) append(value string) error {
	if len(value) > maxOutputBytes-len(g.output) {
		return schemaError("#", "generated document exceeds 1 MiB")
	}
	g.output = append(g.output, value...)
	return nil
}

func (g *generator) value(n *node) error {
	if err := g.ctx.Err(); err != nil {
		return err
	}
	if len(n.values) > 0 {
		return g.append(string(n.values[g.next()%uint64(len(n.values))]))
	}
	if len(n.branches) > 0 {
		return g.value(n.branches[g.next()%uint64(len(n.branches))])
	}
	if n.kind == "null" || n.nullable && g.next()%2 == 0 {
		return g.append("null")
	}
	switch n.kind {
	case "object":
		return g.object(n)
	case "array":
		return g.array(n)
	case "string":
		value := fmt.Sprintf("mock_%016x", g.next())
		if len(value) < n.minSize {
			value += strings.Repeat("x", n.minSize-len(value))
		}
		value = value[:min(len(value), n.maxSize)]
		encoded, _ := json.Marshal(value)
		return g.append(string(encoded))
	case "integer":
		value := int64(n.minimum) + int64(g.next()%uint64(n.maximum-n.minimum+1))
		return g.append(strconv.FormatInt(value, 10))
	case "number":
		fraction := float64(g.next()%10001) / 10000
		value := min(n.maximum, max(n.minimum, n.minimum+(n.maximum-n.minimum)*fraction))
		return g.append(strconv.FormatFloat(value, 'g', -1, 64))
	case "boolean":
		return g.append(strconv.FormatBool(g.next()%2 == 0))
	default:
		return schemaError("#", "unsupported compiled type")
	}
}

func (g *generator) object(n *node) error {
	if err := g.append("{"); err != nil {
		return err
	}
	for index, p := range n.properties {
		if index > 0 {
			if err := g.append(","); err != nil {
				return err
			}
		}
		key, _ := json.Marshal(p.name)
		if err := g.append(string(key) + ":"); err != nil {
			return err
		}
		if err := g.value(p.node); err != nil {
			return err
		}
	}
	return g.append("}")
}

func (g *generator) array(n *node) error {
	if err := g.append("["); err != nil {
		return err
	}
	count := min(n.maxSize, max(1, n.minSize))
	for index := range count {
		if index > 0 {
			if err := g.append(","); err != nil {
				return err
			}
		}
		if err := g.value(n.items); err != nil {
			return err
		}
	}
	return g.append("]")
}
