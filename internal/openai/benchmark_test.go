package openai

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func BenchmarkCreateChatCompletionJSON(b *testing.B) {
	handler := newBenchmarkHandler(b)
	const body = `{"model":"mock-chat-model","messages":[{"role":"user","content":"benchmark prompt"}],"max_completion_tokens":128,"seed":42}`

	b.ReportAllocs()
	for range b.N {
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK {
			b.Fatalf("status = %d", recorder.Code)
		}
	}
}

func BenchmarkCreateChatCompletionSSE(b *testing.B) {
	handler := newBenchmarkHandler(b)
	const body = `{"model":"mock-chat-model","messages":[{"role":"user","content":"benchmark prompt"}],"max_completion_tokens":128,"seed":42,"stream":true,"stream_options":{"include_usage":true}}`

	b.ReportAllocs()
	for range b.N {
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusOK {
			b.Fatalf("status = %d", recorder.Code)
		}
	}
}

func BenchmarkSSEEncoder32Chunks(b *testing.B) {
	writer := benchmarkDiscardResponseWriter{header: make(http.Header)}
	controller := http.NewResponseController(&writer)
	metadata := streamMetadata{ID: "chatcmpl-mock-1", Created: 1_700_000_000, Model: "mock-chat-model"}
	content := []byte(" amber orbit cedar")
	usage := chatCompletionUsage{PromptTokens: 10, CompletionTokens: 128, TotalTokens: 138}

	b.ReportAllocs()
	for range b.N {
		encoder := newSSEEncoder(&writer, controller, metadata, true)
		if err := encoder.writeInitial(); err != nil {
			b.Fatal(err)
		}
		for range 32 {
			if err := encoder.writeContent(content); err != nil {
				b.Fatal(err)
			}
		}
		if err := encoder.writeFinish(); err != nil {
			b.Fatal(err)
		}
		if err := encoder.writeUsage(usage); err != nil {
			b.Fatal(err)
		}
		if err := encoder.writeDone(); err != nil {
			b.Fatal(err)
		}
	}
}

func newBenchmarkHandler(b *testing.B) *Handler {
	b.Helper()

	chatModel, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{
		DefaultCompletionTokens: 128,
		MaxCompletionTokens:     4096,
		StreamTokensPerChunk:    4,
	})
	if err != nil {
		b.Fatalf("NewMockModel() error = %v", err)
	}
	registry, err := chat.NewRegistry(chatModel)
	if err != nil {
		b.Fatalf("NewRegistry() error = %v", err)
	}
	handler, err := NewHandler(registry, testResponseStoreOptions())
	if err != nil {
		b.Fatalf("NewHandler() error = %v", err)
	}
	return handler
}

type benchmarkDiscardResponseWriter struct {
	header http.Header
}

func (w *benchmarkDiscardResponseWriter) Header() http.Header {
	return w.header
}

func (*benchmarkDiscardResponseWriter) Write(value []byte) (int, error) {
	return len(value), nil
}

func (*benchmarkDiscardResponseWriter) WriteHeader(int) {}

func (*benchmarkDiscardResponseWriter) FlushError() error {
	return nil
}
