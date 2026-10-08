package sdk_test

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"io"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/pacoricci/mock-ai-go/internal/app"
	"github.com/pacoricci/mock-ai-go/internal/config"
)

func newEmbeddingClient(t *testing.T) openai.Client {
	t.Helper()
	server, err := app.Start(io.Discard, config.Config{
		HTTPAddress: "127.0.0.1:0", ModelID: "chat",
		DefaultCompletionTokens: 8, MaxCompletionTokens: 64, StreamTokensPerChunk: 2,
		ResponseStoreTTL: time.Hour, ResponseStoreMaxEntries: 10, ResponseStoreMaxBytes: 1 << 20,
		EmbeddingModelID: "sdk-embedding", EmbeddingDimensions: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			t.Error(err)
		}
	})
	return openai.NewClient(option.WithBaseURL("http://"+server.HTTPAddress()+"/v1/"), option.WithAPIKey("dummy"), option.WithMaxRetries(0))
}

func TestEmbeddingsThroughApplication(t *testing.T) {
	t.Parallel()
	client := newEmbeddingClient(t)
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	params := openai.EmbeddingNewParams{
		Model: "sdk-embedding",
		Input: openai.EmbeddingNewParamsInputUnion{OfString: openai.String("Hello café 世界")},
	}
	single, err := client.Embeddings.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if single.Object != "list" || single.Model != "sdk-embedding" || len(single.Data) != 1 || len(single.Data[0].Embedding) != 32 || single.Usage.PromptTokens != 5 || single.Usage.TotalTokens != 5 {
		t.Fatalf("response = %+v", single)
	}
	params.Input = openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: []string{"other", "Hello café 世界"}}
	batch, err := client.Embeddings.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(batch.Data) != 2 || !reflect.DeepEqual(batch.Data[1].Embedding, single.Data[0].Embedding) {
		t.Fatal("batch mismatch")
	}
	for i, item := range batch.Data {
		if item.Index != int64(i) || item.Object != "embedding" {
			t.Fatalf("item = %+v", item)
		}
	}
	params.Dimensions = openai.Int(7)
	params.EncodingFormat = openai.EmbeddingNewParamsEncodingFormatFloat
	shortened, err := client.Embeddings.New(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(shortened.Data[0].Embedding) != 7 {
		t.Fatal("dimensions ignored")
	}
	params.EncodingFormat = openai.EmbeddingNewParamsEncodingFormatBase64
	// This SDK version types embeddings as float arrays, so decode base64 via its raw-body option.
	var encoded struct{ Data []struct{ Embedding string } }
	if _, err := client.Embeddings.New(ctx, params, option.WithResponseBodyInto(&encoded)); err != nil {
		t.Fatal(err)
	}
	if len(encoded.Data) != len(shortened.Data) {
		t.Fatal("base64 item count")
	}
	for i, item := range encoded.Data {
		raw, err := base64.StdEncoding.DecodeString(item.Embedding)
		if err != nil {
			t.Fatal(err)
		}
		if len(raw) != 7*4 {
			t.Fatal("base64 vector size")
		}
		for j, value := range shortened.Data[i].Embedding {
			if binary.LittleEndian.Uint32(raw[j*4:]) != math.Float32bits(float32(value)) {
				t.Fatal("base64 differs from float")
			}
		}
	}
}

func TestEmbeddingSDKErrors(t *testing.T) {
	t.Parallel()
	client := newEmbeddingClient(t)
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	for _, test := range []struct {
		name, model string
		input       openai.EmbeddingNewParamsInputUnion
		dimensions  int64
		status      int
		param       string
	}{
		{"unknown model", "unknown", openai.EmbeddingNewParamsInputUnion{OfString: openai.String("hello")}, 16, 404, "model"},
		{"dimensions", "sdk-embedding", openai.EmbeddingNewParamsInputUnion{OfString: openai.String("hello")}, 33, 400, "dimensions"},
		{"tokens", "sdk-embedding", openai.EmbeddingNewParamsInputUnion{OfArrayOfTokens: []int64{12, 34}}, 16, 400, "input"},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := client.Embeddings.New(ctx, openai.EmbeddingNewParams{Model: test.model, Input: test.input, Dimensions: openai.Int(test.dimensions)})
			var apiError *openai.Error
			if !errors.As(err, &apiError) || apiError.StatusCode != test.status || apiError.Param != test.param || apiError.Type != "invalid_request_error" {
				t.Fatalf("error = %v", err)
			}
		})
	}
}
