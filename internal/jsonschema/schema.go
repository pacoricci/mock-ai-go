// Package jsonschema compiles a bounded JSON Schema subset for deterministic mock output.
package jsonschema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
)

const (
	maxSchemaBytes = 64 << 10
	maxNodes       = 4096
	maxDepth       = 32
	maxItems       = 256
	maxString      = 4096
	maxOutputBytes = 1 << 20
)

// Error identifies an invalid or unsupported schema at a JSON Pointer path.
type Error struct {
	Path    string
	Message string
}

// Error returns the schema path and diagnostic message.
func (e *Error) Error() string { return e.Path + ": " + e.Message }

// Schema is an immutable compiled schema, safe for concurrent generation.
type Schema struct {
	root *node
}

type property struct {
	name string
	node *node
}

type node struct {
	kind       string
	nullable   bool
	properties []property
	items      *node
	branches   []*node
	values     []json.RawMessage
	minimum    float64
	maximum    float64
	minSize    int
	maxSize    int
}

type compiler struct {
	root   json.RawMessage
	strict bool
	nodes  int
	active map[string]bool
}

// Compile copies raw into an immutable plan. Strict schemas require every object
// property to be required and additionalProperties to be false. Recursive refs,
// unsupported constraints, and schemas exceeding the documented bounds fail.
func Compile(raw json.RawMessage, strict bool) (*Schema, error) {
	if len(raw) > maxSchemaBytes {
		return nil, schemaError("#", "schema exceeds 64 KiB")
	}
	c := compiler{root: bytes.Clone(raw), strict: strict, active: make(map[string]bool)}
	n, err := c.compile(c.root, "#", 0)
	if err != nil {
		return nil, err
	}
	if n.kind != "object" || n.nullable || len(n.branches) != 0 {
		return nil, schemaError("#", "root must be an object")
	}
	return &Schema{root: n}, nil
}

func schemaError(path, message string) error { return &Error{Path: path, Message: message} }

func pointer(path, key string) string {
	return path + "/" + strings.ReplaceAll(strings.ReplaceAll(key, "~", "~0"), "/", "~1")
}

func (c *compiler) compile(raw json.RawMessage, path string, depth int) (*node, error) {
	c.nodes++
	if depth > maxDepth || c.nodes > maxNodes {
		return nil, schemaError(path, "schema exceeds depth 32 or 4096 expanded nodes")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil || fields == nil {
		return nil, schemaError(path, "expected a schema object")
	}
	for key := range fields {
		switch key {
		case "type", "properties", "required", "additionalProperties", "items", "enum", "const",
			"anyOf", "$ref", "$defs", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
			"title", "description", "default", "examples", "$schema":
		default:
			return nil, schemaError(pointer(path, key), "unsupported keyword")
		}
	}
	if rawDefs, exists := fields["$defs"]; exists {
		var definitions map[string]json.RawMessage
		if json.Unmarshal(rawDefs, &definitions) != nil || definitions == nil {
			return nil, schemaError(pointer(path, "$defs"), "expected an object")
		}
		for name, definition := range definitions {
			if _, err := c.compile(definition, pointer(pointer(path, "$defs"), name), depth+1); err != nil {
				return nil, err
			}
		}
	}
	if rawRef, exists := fields["$ref"]; exists {
		return c.reference(fields, rawRef, path, depth)
	}
	if branches, exists := fields["anyOf"]; exists {
		return c.union(fields, branches, path, depth)
	}
	n := &node{}
	if err := n.readType(fields["type"], path); err != nil {
		return nil, err
	}
	if err := checkKeywords(fields, n.kind, path); err != nil {
		return nil, err
	}
	var err error
	switch n.kind {
	case "object":
		err = c.object(n, fields, path, depth)
	case "array":
		n.minSize, n.maxSize, err = sizeBounds(fields, "minItems", "maxItems", maxItems, path)
		if err == nil {
			n.items, err = c.compile(fields["items"], pointer(path, "items"), depth+1)
		}
	case "string":
		n.minSize, n.maxSize, err = sizeBounds(fields, "minLength", "maxLength", maxString, path)
	case "integer", "number":
		err = n.numberBounds(fields, path)
	}
	if err != nil {
		return nil, err
	}
	if err := n.readValues(fields, path); err != nil {
		return nil, err
	}
	return n, nil
}

func (n *node) readType(raw json.RawMessage, path string) error {
	if err := json.Unmarshal(raw, &n.kind); err != nil {
		var kinds []string
		if err := json.Unmarshal(raw, &kinds); err != nil || len(kinds) != 2 || !slices.Contains(kinds, "null") || kinds[0] == kinds[1] {
			return schemaError(pointer(path, "type"), "expected a type or one type plus null")
		}
		n.nullable = true
		n.kind = kinds[0]
		if n.kind == "null" {
			n.kind = kinds[1]
		}
	}
	switch n.kind {
	case "object", "array", "string", "integer", "number", "boolean", "null":
		return nil
	default:
		return schemaError(pointer(path, "type"), "unsupported or missing type")
	}
}

func checkKeywords(fields map[string]json.RawMessage, kind, path string) error {
	for key := range fields {
		var expected string
		switch key {
		case "properties", "required", "additionalProperties":
			expected = "object"
		case "items", "minItems", "maxItems":
			expected = "array"
		case "minLength", "maxLength":
			expected = "string"
		case "minimum", "maximum":
			if kind == "integer" {
				continue
			}
			expected = "number"
		}
		if expected != "" && kind != expected {
			return schemaError(pointer(path, key), "keyword does not match type")
		}
	}
	return nil
}

func (c *compiler) object(n *node, fields map[string]json.RawMessage, path string, depth int) error {
	var additional bool
	if raw, exists := fields["additionalProperties"]; exists {
		if string(raw) == "null" || json.Unmarshal(raw, &additional) != nil {
			return schemaError(pointer(path, "additionalProperties"), "expected a boolean")
		}
	} else {
		additional = true
	}
	if c.strict && additional {
		return schemaError(pointer(path, "additionalProperties"), "strict objects require false")
	}
	var required []string
	if raw, exists := fields["required"]; exists {
		var names []json.RawMessage
		if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) || json.Unmarshal(raw, &names) != nil {
			return schemaError(pointer(path, "required"), "expected an array of property names")
		}
		for _, name := range names {
			var text string
			if bytes.Equal(bytes.TrimSpace(name), []byte("null")) || json.Unmarshal(name, &text) != nil {
				return schemaError(pointer(path, "required"), "property names must be strings")
			}
			required = append(required, text)
		}
	}
	properties := fields["properties"]
	if properties == nil {
		properties = json.RawMessage(`{}`)
	}
	decoder := json.NewDecoder(bytes.NewReader(properties))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return schemaError(pointer(path, "properties"), "expected an object")
	}
	seen := make(map[string]bool)
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return schemaError(path, "invalid properties")
		}
		name, ok := token.(string)
		if !ok || seen[name] || len(seen) >= 256 {
			return schemaError(pointer(path, "properties"), "duplicate property or more than 256 properties")
		}
		seen[name] = true
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return schemaError(path, "invalid property schema")
		}
		child, err := c.compile(raw, pointer(pointer(path, "properties"), name), depth+1)
		if err != nil {
			return err
		}
		if c.strict && !slices.Contains(required, name) {
			return schemaError(pointer(path, "required"), "strict objects require every property")
		}
		n.properties = append(n.properties, property{name: name, node: child})
	}
	seenRequired := make(map[string]bool)
	for _, name := range required {
		if !seen[name] || seenRequired[name] {
			return schemaError(pointer(path, "required"), "unknown or duplicate property")
		}
		seenRequired[name] = true
	}
	return nil
}

func sizeBounds(fields map[string]json.RawMessage, minKey, maxKey string, limit int, path string) (int, int, error) {
	minimum, maximum := 0, limit
	for key, target := range map[string]*int{minKey: &minimum, maxKey: &maximum} {
		if raw, exists := fields[key]; exists {
			if string(raw) == "null" || json.Unmarshal(raw, target) != nil || *target < 0 || *target > limit {
				return 0, 0, schemaError(pointer(path, key), fmt.Sprintf("expected integer between 0 and %d", limit))
			}
		}
	}
	if minimum > maximum {
		return 0, 0, schemaError(path, "minimum exceeds maximum")
	}
	return minimum, maximum, nil
}

func (n *node) numberBounds(fields map[string]json.RawMessage, path string) error {
	n.minimum, n.maximum = 0, 999
	for key, target := range map[string]*float64{"minimum": &n.minimum, "maximum": &n.maximum} {
		if raw, exists := fields[key]; exists {
			if string(raw) == "null" || json.Unmarshal(raw, target) != nil || math.Abs(*target) > 1e12 {
				return schemaError(pointer(path, key), "expected a finite number within +/-1e12")
			}
		}
	}
	if _, exists := fields["minimum"]; !exists {
		n.minimum = min(n.minimum, n.maximum)
	}
	if _, exists := fields["maximum"]; !exists {
		n.maximum = max(n.maximum, n.minimum)
	}
	if n.kind == "integer" {
		n.minimum, n.maximum = math.Ceil(n.minimum), math.Floor(n.maximum)
	}
	if n.minimum > n.maximum {
		return schemaError(path, "numeric bounds have no supported value")
	}
	return nil
}

func (c *compiler) reference(fields map[string]json.RawMessage, raw json.RawMessage, path string, depth int) (*node, error) {
	if err := standalone(fields, "$ref", path); err != nil {
		return nil, err
	}
	var ref string
	if json.Unmarshal(raw, &ref) != nil || !strings.HasPrefix(ref, "#/$defs/") {
		return nil, schemaError(pointer(path, "$ref"), "only local #/$defs/ references are supported")
	}
	if c.active[ref] {
		return nil, schemaError(pointer(path, "$ref"), "recursive references are unsupported")
	}
	resolved := c.root
	for _, part := range strings.Split(strings.TrimPrefix(ref, "#/"), "/") {
		var object map[string]json.RawMessage
		for index := 0; index < len(part); index++ {
			if part[index] == '~' {
				if index+1 == len(part) || part[index+1] != '0' && part[index+1] != '1' {
					return nil, schemaError(pointer(path, "$ref"), "invalid JSON Pointer escape")
				}
				index++
			}
		}
		part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
		if json.Unmarshal(resolved, &object) != nil || object[part] == nil {
			return nil, schemaError(pointer(path, "$ref"), "reference target does not exist")
		}
		resolved = object[part]
	}
	c.active[ref] = true
	n, err := c.compile(resolved, ref, depth+1)
	delete(c.active, ref)
	return n, err
}

func (c *compiler) union(fields map[string]json.RawMessage, raw json.RawMessage, path string, depth int) (*node, error) {
	if err := standalone(fields, "anyOf", path); err != nil {
		return nil, err
	}
	var branches []json.RawMessage
	if json.Unmarshal(raw, &branches) != nil || len(branches) == 0 || len(branches) > 256 {
		return nil, schemaError(pointer(path, "anyOf"), "expected 1 to 256 branches")
	}
	n := &node{}
	for index, rawBranch := range branches {
		branch, err := c.compile(rawBranch, fmt.Sprintf("%s/anyOf/%d", path, index), depth+1)
		if err != nil {
			return nil, err
		}
		n.branches = append(n.branches, branch)
	}
	return n, nil
}

func standalone(fields map[string]json.RawMessage, keyword, path string) error {
	for key := range fields {
		switch key {
		case keyword, "$defs", "title", "description", "default", "examples", "$schema":
		default:
			return schemaError(pointer(path, key), "constraints alongside "+keyword+" are unsupported")
		}
	}
	return nil
}
