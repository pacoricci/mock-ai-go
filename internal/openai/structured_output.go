package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
	"github.com/pacoricci/mock-ai-go/internal/jsonschema"
)

type outputFormat struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type structuredFormat struct {
	format      outputFormat
	schema      *jsonschema.Schema
	fingerprint string
	param       string
}

func normalizeChatFormat(raw json.RawMessage) (structuredFormat, *responseRequestError) {
	if len(raw) == 0 {
		return plainFormat(), nil
	}
	var envelope struct {
		Type       string          `json:"type"`
		JSONSchema json.RawMessage `json:"json_schema"`
	}
	if err := decodeFormat(raw, &envelope); err != nil {
		return structuredFormat{}, formatError("response_format", err.Error())
	}
	if envelope.Type == "json_schema" {
		var format outputFormat
		if err := decodeFormat(envelope.JSONSchema, &format); err != nil {
			return structuredFormat{}, formatError("response_format.json_schema", err.Error())
		}
		if format.Type != "" {
			return structuredFormat{}, formatError("response_format.json_schema.type", "type belongs in response_format")
		}
		format.Type = envelope.Type
		return compileFormat(format, "response_format.json_schema")
	}
	if envelope.JSONSchema != nil {
		return structuredFormat{}, formatError("response_format.json_schema", "json_schema requires type json_schema")
	}
	return compileFormat(outputFormat{Type: envelope.Type}, "response_format")
}

func normalizeResponseFormat(raw json.RawMessage) (structuredFormat, *responseRequestError) {
	if len(raw) == 0 {
		return plainFormat(), nil
	}
	var config struct {
		Format    json.RawMessage `json:"format"`
		Verbosity json.RawMessage `json:"verbosity"`
	}
	if err := decodeFormat(raw, &config); err != nil {
		return structuredFormat{}, formatError("text", err.Error())
	}
	if len(config.Format) == 0 {
		return plainFormat(), nil
	}
	var format outputFormat
	if err := decodeFormat(config.Format, &format); err != nil {
		return structuredFormat{}, formatError("text.format", err.Error())
	}
	return compileFormat(format, "text.format")
}

func decodeFormat(raw json.RawMessage, target any) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || trimmed[0] != '{' {
		return errors.New("expected an object")
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.DisallowUnknownFields()
	return decoder.Decode(target)
}

func plainFormat() structuredFormat {
	return structuredFormat{format: outputFormat{Type: "text"}}
}

func formatError(param, message string) *responseRequestError {
	return newResponseRequestError(message, param, "invalid_value")
}

func compileFormat(format outputFormat, param string) (structuredFormat, *responseRequestError) {
	result := structuredFormat{format: format, param: param}
	switch format.Type {
	case "text", "json_object":
		if format.Name != "" || format.Description != "" || format.Schema != nil || format.Strict != nil {
			return result, formatError(param, "schema settings require type json_schema")
		}
	case "json_schema":
		if !validFormatName(format.Name) {
			return result, formatError(param+".name", "name requires 1-64 ASCII letters, digits, underscores or hyphens")
		}
		schema, err := jsonschema.Compile(format.Schema, format.Strict != nil && *format.Strict)
		if err != nil {
			return result, formatError(schemaErrorParameter(param, err), err.Error())
		}
		result.schema = schema
	default:
		return result, formatError(param+".type", "supported formats are text, json_object and json_schema")
	}
	encoded, err := json.Marshal(format)
	if err != nil {
		return result, formatError(param, err.Error())
	}
	result.fingerprint = string(encoded)
	return result, nil
}

func validFormatName(name string) bool {
	if name == "" || len(name) > 64 {
		return false
	}
	for _, c := range name {
		if c != '_' && c != '-' && (c < 'a' || c > 'z') && (c < 'A' || c > 'Z') && (c < '0' || c > '9') {
			return false
		}
	}
	return true
}

// textGeneration binds an immutable structured plan to one request while keeping
// the plain text model contract available for ordinary completions.
type textGeneration struct {
	chat.Model
	structured chat.StructuredModel
	plan       *chat.StructuredPlan
}

func prepareTextGeneration(ctx context.Context, model chat.Model, request chat.Request, format structuredFormat) (textGeneration, *responseRequestError, error) {
	generation := textGeneration{Model: model}
	if format.format.Type == "text" {
		return generation, nil, nil
	}
	capability, ok := model.(chat.StructuredModel)
	if !ok {
		return generation, formatError(format.param, "model does not support structured output"), nil
	}
	plan, err := capability.PlanStructured(ctx, chat.StructuredRequest{Request: request, Schema: format.schema, Format: format.fingerprint})
	if err != nil {
		var schemaError *jsonschema.Error
		if errors.As(err, &schemaError) {
			return generation, formatError(format.param+".schema", schemaError.Error()), nil
		}
		return generation, nil, err
	}
	generation.structured = capability
	generation.plan = &plan
	return generation, nil, nil
}

// Complete returns ordinary text or the prepared structured output.
func (g textGeneration) Complete(ctx context.Context, request chat.Request) (chat.Response, error) {
	if g.plan == nil {
		return g.Model.Complete(ctx, request)
	}
	if err := ctx.Err(); err != nil {
		return chat.Response{}, err
	}
	return chat.Response{Content: g.plan.Content, Usage: g.plan.Usage}, nil
}

// Stream emits ordinary text or the prepared structured output with model pacing.
func (g textGeneration) Stream(ctx context.Context, request chat.Request, emit chat.Emit) (chat.Usage, error) {
	if g.plan == nil {
		return g.Model.Stream(ctx, request, emit)
	}
	err := g.structured.StreamStructured(ctx, request, *g.plan, emit)
	return g.plan.Usage, err
}

func (g textGeneration) incomplete() bool { return g.plan != nil && g.plan.Incomplete }

func (g textGeneration) finishReason() string {
	if g.incomplete() {
		return "length"
	}
	return "stop"
}

func (g textGeneration) outputStatus() string {
	if g.incomplete() {
		return "incomplete"
	}
	return "completed"
}

// schemaErrorParameter converts a compiler JSON Pointer into a provider parameter
// path while leaving the full pointer available in the human-readable message.
func schemaErrorParameter(param string, err error) string {
	var schemaErr *jsonschema.Error
	if errors.As(err, &schemaErr) && schemaErr.Path != "#" {
		return fmt.Sprintf("%s.schema%s", param, strings.ReplaceAll(strings.TrimPrefix(schemaErr.Path, "#"), "/", "."))
	}
	return param + ".schema"
}
