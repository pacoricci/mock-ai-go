package openai_test

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/embedding"
	"github.com/pacoricci/mock-ai-go/internal/openai"
)

func embeddingsHandler(t *testing.T) http.Handler {
	t.Helper()
	model, err := embedding.NewMockModel("embeddings", 64)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := openai.NewEmbeddingsHandler(model)
	if err != nil {
		t.Fatal(err)
	}
	return handler
}

func postEmbedding(handler http.Handler, body string) *httptest.ResponseRecorder {
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/embeddings", strings.NewReader(body)))
	return recorder
}

func TestEmbeddingsFormats(t *testing.T) {
	t.Parallel()
	handler := embeddingsHandler(t)
	var reference [][]float32
	for _, format := range []string{"float", "base64"} {
		response := postEmbedding(handler, fmt.Sprintf(`{"model":"embeddings","input":["café","世界","café"],"dimensions":17,"encoding_format":%q,"user":"test-user"}`, format))
		if response.Code != 200 || response.Header().Get("Content-Type") != "application/json" {
			t.Fatal(response.Code, response.Body.String())
		}
		var result struct {
			Object, Model string
			Data          []struct {
				Object    string
				Index     int
				Embedding json.RawMessage
			}
			Usage struct {
				PromptTokens int `json:"prompt_tokens"`
				TotalTokens  int `json:"total_tokens"`
			}
		}
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		if result.Object != "list" || result.Model != "embeddings" || len(result.Data) != 3 || result.Usage.PromptTokens != 6 || result.Usage.TotalTokens != 6 {
			t.Fatalf("response = %+v", result)
		}
		vectors := make([][]float32, len(result.Data))
		for i, data := range result.Data {
			if data.Object != "embedding" || data.Index != i {
				t.Fatal("wrong item metadata")
			}
			if format == "float" {
				if err := json.Unmarshal(data.Embedding, &vectors[i]); err != nil {
					t.Fatal(err)
				}
			} else {
				var encoded string
				if err := json.Unmarshal(data.Embedding, &encoded); err != nil {
					t.Fatal(err)
				}
				raw, err := base64.StdEncoding.DecodeString(encoded)
				if err != nil {
					t.Fatal(err)
				}
				if len(raw) != 17*4 {
					t.Fatalf("binary length = %d", len(raw))
				}
				for j := 0; j < len(raw); j += 4 {
					vectors[i] = append(vectors[i], math.Float32frombits(binary.LittleEndian.Uint32(raw[j:])))
				}
			}
			if len(vectors[i]) != 17 {
				t.Fatal("wrong vector length")
			}
		}
		if reference == nil {
			reference = vectors
		} else if !reflect.DeepEqual(reference, vectors) {
			t.Fatal("float and base64 differ")
		}
	}
	response := postEmbedding(handler, `{"model":"embeddings","input":"hello"}`)
	var result struct {
		Data []struct{ Embedding []float32 }
	}
	if response.Code != 200 {
		t.Fatal(response.Body.String())
	}
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if len(result.Data) != 1 || len(result.Data[0].Embedding) != 64 {
		t.Fatal("wrong defaults")
	}
}

func TestEmbeddingsRejectInvalidRequests(t *testing.T) {
	t.Parallel()
	handler := embeddingsHandler(t)
	for _, test := range []struct {
		name, body, param string
		status            int
	}{
		{"malformed", `{`, "", 400},
		{"trailing JSON", `{} {}`, "", 400},
		{"null body", `null`, "model", 400},
		{"array body", `[]`, "", 400},
		{"missing model", `{"input":"hello"}`, "model", 400},
		{"unknown model", `{"model":"chat","input":"hello"}`, "model", 404},
		{"missing input", `{"model":"embeddings"}`, "input", 400},
		{"null input", `{"model":"embeddings","input":null}`, "input", 400},
		{"empty string", `{"model":"embeddings","input":""}`, "input", 400},
		{"empty batch", `{"model":"embeddings","input":[]}`, "input", 400},
		{"null item", `{"model":"embeddings","input":["valid",null]}`, "input", 400},
		{"empty item", `{"model":"embeddings","input":["valid",""]}`, "input", 400},
		{"tokens", `{"model":"embeddings","input":[1,2]}`, "input", 400},
		{"token batches", `{"model":"embeddings","input":[[1,2]]}`, "input", 400},
		{"mixed", `{"model":"embeddings","input":["hello",1]}`, "input", 400},
		{"zero dimensions", `{"model":"embeddings","input":"hello","dimensions":0}`, "dimensions", 400},
		{"negative dimensions", `{"model":"embeddings","input":"hello","dimensions":-1}`, "dimensions", 400},
		{"excess dimensions", `{"model":"embeddings","input":"hello","dimensions":65}`, "dimensions", 400},
		{"fractional dimensions", `{"model":"embeddings","input":"hello","dimensions":1.5}`, "dimensions", 400},
		{"invalid format", `{"model":"embeddings","input":"hello","encoding_format":""}`, "encoding_format", 400},
		{"stream", `{"model":"embeddings","input":"hello","stream":true}`, "stream", 400},
		{"seed", `{"model":"embeddings","input":"hello","seed":42}`, "seed", 400},
		{"long input", fmt.Sprintf(`{"model":"embeddings","input":%q}`, strings.Repeat("x", embedding.MaxInputBytes+1)), "input", 400},
		{"body limit", fmt.Sprintf(`{"model":"embeddings","input":%q}`, strings.Repeat("x", 1<<20)), "", 400},
	} {
		t.Run(test.name, func(t *testing.T) {
			response := postEmbedding(handler, test.body)
			if response.Code != test.status {
				t.Fatal(response.Code, response.Body.String())
			}
			var result struct{ Error struct{ Type, Param string } }
			if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
				t.Fatal(err)
			}
			if result.Error.Type != "invalid_request_error" || result.Error.Param != test.param {
				t.Fatal(response.Body.String())
			}
		})
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/embeddings", nil))
	if recorder.Code != 405 {
		t.Fatal(recorder.Code)
	}
	if _, err := openai.NewEmbeddingsHandler(nil); err == nil {
		t.Fatal("accepted nil model")
	}
}

func TestEmbeddingNullableDefaultsAndIgnoredUser(t *testing.T) {
	t.Parallel()
	handler := embeddingsHandler(t)
	first := postEmbedding(handler, `{"model":"embeddings","input":"hello"}`)
	second := postEmbedding(handler, `{"model":"embeddings","input":["hello"],"dimensions":null,"encoding_format":null,"user":"other-user"}`)
	if first.Code != 200 || second.Code != 200 || first.Body.String() != second.Body.String() {
		t.Fatal("nullable defaults, input wrapping, or user changed the response")
	}
}

func TestEmbeddingHTTPBatchLimits(t *testing.T) {
	t.Parallel()
	model, err := embedding.NewMockModel("embeddings", embedding.MaxDimensions)
	if err != nil {
		t.Fatal(err)
	}
	handler, err := openai.NewEmbeddingsHandler(model)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct{ count, dimensions int }{
		{embedding.MaxBatchSize + 1, 1},
		{embedding.MaxValues/embedding.MaxDimensions + 1, embedding.MaxDimensions},
	} {
		inputs := make([]string, test.count)
		for i := range inputs {
			inputs[i] = "document"
		}
		body, err := json.Marshal(map[string]any{"model": "embeddings", "input": inputs, "dimensions": test.dimensions})
		if err != nil {
			t.Fatal(err)
		}
		response := postEmbedding(handler, string(body))
		if response.Code != 400 || !strings.Contains(response.Body.String(), `"param":"input"`) {
			t.Fatal(response.Code, response.Body.String())
		}
	}
}
