package openai

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"

	"github.com/pacoricci/mock-ai-go/internal/embedding"
)

// NewEmbeddingsHandler serves POST /v1/embeddings using a stateless mock model.
func NewEmbeddingsHandler(model *embedding.MockModel) (http.Handler, error) {
	if model == nil {
		return nil, fmt.Errorf("embedding model is nil")
	}
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		createEmbeddings(w, r, model)
	})
	return mux, nil
}

type embeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	Dimensions     *int            `json:"dimensions"`
	EncodingFormat *string         `json:"encoding_format"`
	User           *string         `json:"user"`
}

type embeddingData struct {
	Object    string `json:"object"`
	Index     int    `json:"index"`
	Embedding any    `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type embeddingResponse struct {
	Object string          `json:"object"`
	Data   []embeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingUsage  `json:"usage"`
}

func createEmbeddings(w http.ResponseWriter, r *http.Request, model *embedding.MockModel) {
	fields, ok := decodeJSONRequest[map[string]json.RawMessage](w, r)
	if !ok {
		return
	}
	var request embeddingRequest
	for name, raw := range fields {
		var target any
		switch name {
		case "model":
			target = &request.Model
		case "input":
			target = &request.Input
		case "dimensions":
			target = &request.Dimensions
		case "encoding_format":
			target = &request.EncodingFormat
		case "user":
			target = &request.User
		default:
			writeEmbeddingValidation(w, name, "Unsupported embeddings parameter: "+name)
			return
		}
		if err := json.Unmarshal(raw, target); err != nil {
			writeEmbeddingValidation(w, name, "Invalid embeddings parameter type")
			return
		}
	}
	if request.Model == "" {
		writeEmbeddingValidation(w, "model", "model is required")
		return
	}
	if request.Model != model.ID() {
		writeError(w, http.StatusNotFound, fmt.Sprintf("The model %q does not exist", request.Model), "invalid_request_error", stringPointer("model"), stringPointer("model_not_found"))
		return
	}
	inputs, err := decodeEmbeddingInputs(request.Input)
	if err != nil {
		writeEmbeddingValidation(w, "input", err.Error())
		return
	}
	dimensions := model.Dimensions()
	if request.Dimensions != nil {
		dimensions = *request.Dimensions
	}
	format := "float"
	if request.EncodingFormat != nil {
		format = *request.EncodingFormat
	}
	if format != "float" && format != "base64" {
		writeEmbeddingValidation(w, "encoding_format", "encoding_format must be float or base64")
		return
	}
	result, err := model.Embed(r.Context(), inputs, dimensions)
	if err != nil {
		var validation *embedding.ValidationError
		if errors.As(err, &validation) {
			writeEmbeddingValidation(w, validation.Param, validation.Message)
		} else if r.Context().Err() == nil {
			writeError(w, http.StatusInternalServerError, "Embedding generation failed", "server_error", nil, nil)
		}
		return
	}
	response := embeddingResponse{
		Object: "list", Model: model.ID(), Data: make([]embeddingData, len(result.Vectors)),
		Usage: embeddingUsage{PromptTokens: result.PromptTokens, TotalTokens: result.PromptTokens},
	}
	for i, vector := range result.Vectors {
		var value any = vector
		if format == "base64" {
			data := make([]byte, len(vector)*4)
			for j, component := range vector {
				binary.LittleEndian.PutUint32(data[j*4:], math.Float32bits(component))
			}
			value = base64.StdEncoding.EncodeToString(data)
		}
		response.Data[i] = embeddingData{Object: "embedding", Index: i, Embedding: value}
	}
	writeJSON(w, response)
}

func decodeEmbeddingInputs(raw json.RawMessage) ([]string, error) {
	const message = "input must be a string or an array of strings; token arrays are unsupported"
	if len(raw) == 0 {
		return nil, errors.New(message)
	}
	if raw[0] == '"' {
		var value string
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, errors.New(message)
		}
		return []string{value}, nil
	}
	var items []json.RawMessage
	if raw[0] != '[' || json.Unmarshal(raw, &items) != nil {
		return nil, errors.New(message)
	}
	values := make([]string, len(items))
	for i, item := range items {
		if len(item) == 0 || item[0] != '"' || json.Unmarshal(item, &values[i]) != nil {
			return nil, errors.New(message)
		}
	}
	return values, nil
}

func writeEmbeddingValidation(w http.ResponseWriter, param, message string) {
	writeError(w, http.StatusBadRequest, message, "invalid_request_error", stringPointer(param), stringPointer("invalid_value"))
}
