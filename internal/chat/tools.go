package chat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

const (
	toolSeedSalt                   uint64 = 0xa0761d6478bd642f
	responsesToolSeedDomain               = "mock-ai-go/responses-tool-plan/v1"
	maxSynthesizedDepth                   = 8
	maxSynthesizedArrayItems              = 4
	maxSynthesizedObjectProperties        = 32
	maxSynthesizedStringLength            = 64
	toolCallTokenOverhead                 = 3
)

var (
	// ErrInvalidToolChoice indicates an unsupported tool selection mode.
	ErrInvalidToolChoice = errors.New("invalid tool choice")
	// ErrRequiredToolMissing indicates that required tool selection has no tools.
	ErrRequiredToolMissing = errors.New("required tool choice has no tools")
	// ErrInvalidMaxToolCalls indicates a negative per-request tool call limit.
	ErrInvalidMaxToolCalls = errors.New("max tool calls must not be negative")
)

// PlanTools deterministically selects text generation or synthetic function calls.
func (m *MockModel) PlanTools(ctx context.Context, request ToolRequest) (ToolPlan, error) {
	if err := ctx.Err(); err != nil {
		return ToolPlan{}, fmt.Errorf("plan chat tools: %w", err)
	}
	if request.MaxToolCalls < 0 {
		return ToolPlan{}, ErrInvalidMaxToolCalls
	}
	if request.Choice == "" {
		request.Choice = ToolChoiceAuto
	}
	if request.Choice != ToolChoiceNone && request.Choice != ToolChoiceAuto && request.Choice != ToolChoiceRequired {
		return ToolPlan{}, fmt.Errorf("%w: %q", ErrInvalidToolChoice, request.Choice)
	}
	if request.Choice == ToolChoiceRequired && len(request.Tools) == 0 {
		return ToolPlan{}, ErrRequiredToolMissing
	}
	if request.Choice == ToolChoiceNone || len(request.Tools) == 0 {
		return ToolPlan{UseText: true}, nil
	}

	completionTokens, err := m.completionTokens(request.Request)
	if err != nil {
		return ToolPlan{}, err
	}
	random, err := m.newToolRandom(request, completionTokens)
	if err != nil {
		return ToolPlan{}, err
	}
	if request.Choice == ToolChoiceAuto && random.next()&1 != 0 {
		return ToolPlan{UseText: true}, nil
	}

	limit := len(request.Tools)
	if !request.ParallelToolCalls {
		limit = 1
	}
	if request.MaxToolCalls > 0 {
		limit = min(limit, request.MaxToolCalls)
	}
	callCount := 1
	if request.ParallelToolCalls && limit > 1 {
		callCount += int(uniformUint64n(&random, uint64(limit)))
	}

	indices := make([]int, len(request.Tools))
	for index := range indices {
		indices[index] = index
	}
	calls := make([]ToolCall, 0, callCount)
	completionUsageTokens := 0
	for index := range callCount {
		if err := ctx.Err(); err != nil {
			return ToolPlan{}, fmt.Errorf("plan chat tools: %w", err)
		}
		selected := index + int(uniformUint64n(&random, uint64(len(indices)-index)))
		indices[index], indices[selected] = indices[selected], indices[index]
		tool := request.Tools[indices[index]]
		arguments := synthesizeToolArguments(tool.Parameters, &random)
		call := ToolCall{
			ID:        fmt.Sprintf("fc_mock_%d_%016x", index, random.next()),
			CallID:    fmt.Sprintf("call_mock_%d_%016x", index, random.next()),
			Name:      tool.Name,
			Arguments: arguments,
		}
		calls = append(calls, call)
		completionUsageTokens += toolCallTokenOverhead + estimateTextTokens(call.Name) + estimateTextTokens(call.Arguments)
	}

	usage, err := requestUsage(request.Request, completionUsageTokens)
	if err != nil {
		return ToolPlan{}, err
	}
	return ToolPlan{
		Calls: calls,
		Usage: usage,
	}, nil
}

func (m *MockModel) newToolRandom(request ToolRequest, completionTokens int) (xorshift64Star, error) {
	hash := m.requestSeed(request.Request, completionTokens)
	if request.ContextState != nil {
		hashString(&hash, responsesToolSeedDomain)
	}
	hashUint64(&hash, toolSeedSalt)
	hashString(&hash, string(request.Choice))
	if request.ParallelToolCalls {
		hashByte(&hash, 1)
	} else {
		hashByte(&hash, 0)
	}
	hashUint64(&hash, uint64(request.MaxToolCalls))
	for _, tool := range request.Tools {
		hashString(&hash, tool.Name)
		hashString(&hash, tool.Description)
		parameters, err := json.Marshal(tool.Parameters)
		if err != nil {
			return xorshift64Star{}, fmt.Errorf("canonicalize parameters for tool %q: %w", tool.Name, err)
		}
		hashString(&hash, string(parameters))
	}
	return newXorshift64Star(hash), nil
}

func synthesizeToolArguments(parameters map[string]any, random *xorshift64Star) string {
	arguments := make(map[string]any)
	if parameters != nil {
		if generated, ok := synthesizeSchemaValue(parameters, "argument", random, 0).(map[string]any); ok {
			arguments = generated
		}
	}
	encoded, err := json.Marshal(arguments)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

func synthesizeSchemaValue(schema map[string]any, name string, random *xorshift64Star, depth int) any {
	if depth >= maxSynthesizedDepth {
		return nil
	}
	if value, exists := schema["const"]; exists {
		return value
	}
	if values, ok := schema["enum"].([]any); ok && len(values) > 0 {
		return values[uniformUint64n(random, uint64(len(values)))]
	}
	if branch := chooseSchemaBranch(schema, "oneOf", random); branch != nil {
		return synthesizeSchemaValue(branch, name, random, depth+1)
	}
	if branch := chooseSchemaBranch(schema, "anyOf", random); branch != nil {
		return synthesizeSchemaValue(branch, name, random, depth+1)
	}

	schemaType := effectiveSchemaType(schema)
	switch schemaType {
	case "object":
		return synthesizeObject(schema, random, depth)
	case "array":
		return synthesizeArray(schema, name, random, depth)
	case "integer":
		return synthesizeInteger(schema, random)
	case "number":
		return synthesizeNumber(schema, random)
	case "boolean":
		return random.next()&1 == 0
	case "null":
		return nil
	default:
		return synthesizeString(schema, name, random)
	}
}

func chooseSchemaBranch(schema map[string]any, keyword string, random *xorshift64Star) map[string]any {
	branches, ok := schema[keyword].([]any)
	if !ok || len(branches) == 0 {
		return nil
	}
	start := int(uniformUint64n(random, uint64(len(branches))))
	for offset := range len(branches) {
		branch, ok := branches[(start+offset)%len(branches)].(map[string]any)
		if ok {
			return branch
		}
	}
	return nil
}

func effectiveSchemaType(schema map[string]any) string {
	if value, ok := schema["type"].(string); ok {
		return value
	}
	if values, ok := schema["type"].([]any); ok {
		for _, value := range values {
			if schemaType, ok := value.(string); ok && schemaType != "null" {
				return schemaType
			}
		}
		return "null"
	}
	if _, ok := schema["properties"]; ok {
		return "object"
	}
	if _, ok := schema["items"]; ok {
		return "array"
	}
	return "string"
}

func synthesizeObject(schema map[string]any, random *xorshift64Star, depth int) map[string]any {
	result := make(map[string]any)
	properties, _ := schema["properties"].(map[string]any)
	required, _ := schema["required"].([]any)
	for _, rawName := range required[:min(len(required), maxSynthesizedObjectProperties)] {
		name, ok := rawName.(string)
		if !ok {
			continue
		}
		propertySchema, _ := properties[name].(map[string]any)
		if propertySchema == nil {
			result[name] = nil
			continue
		}
		result[name] = synthesizeSchemaValue(propertySchema, name, random, depth+1)
	}
	return result
}

func synthesizeArray(schema map[string]any, name string, random *xorshift64Star, depth int) []any {
	count := 1
	if minimum, ok := jsonSchemaInt(schema["minItems"]); ok {
		count = max(count, minimum)
	}
	if maximum, ok := jsonSchemaInt(schema["maxItems"]); ok {
		count = min(count, maximum)
	}
	count = max(0, min(count, maxSynthesizedArrayItems))
	itemSchema, _ := schema["items"].(map[string]any)
	result := make([]any, 0, count)
	for index := range count {
		if itemSchema == nil {
			result = append(result, fmt.Sprintf("mock_%s_%d", sanitizeArgumentName(name), index))
			continue
		}
		result = append(result, synthesizeSchemaValue(itemSchema, name, random, depth+1))
	}
	return result
}

func synthesizeInteger(schema map[string]any, random *xorshift64Star) int64 {
	minimum, hasMinimum := jsonSchemaNumber(schema["minimum"])
	maximum, hasMaximum := jsonSchemaNumber(schema["maximum"])
	if !hasMinimum {
		minimum = 0
	}
	if !hasMaximum {
		maximum = minimum + 999
	}
	if maximum < minimum {
		maximum = minimum
	}
	span := uint64(min(maximum-minimum, 999)) + 1
	return int64(minimum) + int64(uniformUint64n(random, span))
}

func synthesizeNumber(schema map[string]any, random *xorshift64Star) float64 {
	minimum, hasMinimum := jsonSchemaNumber(schema["minimum"])
	maximum, hasMaximum := jsonSchemaNumber(schema["maximum"])
	if !hasMinimum {
		minimum = 0
	}
	if !hasMaximum {
		maximum = minimum + 100
	}
	if maximum < minimum {
		maximum = minimum
	}
	fraction := float64(uniformUint64n(random, 10_001)) / 10_000
	return minimum + (maximum-minimum)*fraction
}

func synthesizeString(schema map[string]any, name string, random *xorshift64Star) string {
	value := fmt.Sprintf("mock_%s_%04x", sanitizeArgumentName(name), random.next()&0xffff)
	minimum, _ := jsonSchemaInt(schema["minLength"])
	maximum, hasMaximum := jsonSchemaInt(schema["maxLength"])
	minimum = min(max(0, minimum), maxSynthesizedStringLength)
	for len(value) < minimum {
		value += "x"
	}
	if hasMaximum && maximum >= 0 && len(value) > maximum {
		value = value[:maximum]
	}
	if len(value) > maxSynthesizedStringLength {
		value = value[:maxSynthesizedStringLength]
	}
	return value
}

func sanitizeArgumentName(name string) string {
	var builder strings.Builder
	for _, character := range name {
		if (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '_' {
			builder.WriteRune(character)
			if builder.Len() == 16 {
				break
			}
		}
	}
	if builder.Len() == 0 {
		return "argument"
	}
	return strings.ToLower(builder.String())
}

func jsonSchemaInt(value any) (int, bool) {
	number, ok := jsonSchemaNumber(value)
	if !ok || number > float64(math.MaxInt) || number < float64(math.MinInt) {
		return 0, false
	}
	return int(number), true
}

func jsonSchemaNumber(value any) (float64, bool) {
	number, ok := value.(float64)
	if !ok || math.IsNaN(number) || math.IsInf(number, 0) {
		return 0, false
	}
	return number, true
}
