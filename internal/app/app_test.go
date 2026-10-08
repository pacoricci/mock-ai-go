package app

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/config"
)

func TestStartExposesResolvedAddressesAndServesHTTP(t *testing.T) {
	server, err := Start(io.Discard, config.Config{
		Name:                    "test-app",
		HTTPAddress:             "127.0.0.1:0",
		PPROFAddress:            "127.0.0.1:0",
		ModelID:                 "mock-chat-model",
		DefaultCompletionTokens: 8,
		MaxCompletionTokens:     64,
		StreamTokensPerChunk:    2,
		ResponseStoreTTL:        time.Hour,
		ResponseStoreMaxEntries: 1_000,
		ResponseStoreMaxBytes:   8 << 20,
	})
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if strings.HasSuffix(server.HTTPAddress(), ":0") || strings.HasSuffix(server.DiagnosticsAddress(), ":0") {
		t.Fatalf("resolved addresses = %q, %q", server.HTTPAddress(), server.DiagnosticsAddress())
	}

	response, err := http.Get("http://" + server.HTTPAddress() + "/openapi.yaml")
	if err != nil {
		t.Fatalf("GET OpenAPI document: %v", err)
	}
	_ = response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("GET OpenAPI status = %d", response.StatusCode)
	}

	shutdownContext, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := server.Shutdown(shutdownContext); err != nil {
		t.Fatalf("Shutdown() error = %v", err)
	}
	select {
	case err := <-server.Done():
		if err != nil {
			t.Fatalf("Done() error = %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("server did not report shutdown")
	}
}

func TestRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	output := notifyingWriter{ready: make(chan struct{})}
	errors := make(chan error, 1)

	go func() {
		errors <- Run(ctx, &output, config.Config{
			Name:                    "test-app",
			HTTPAddress:             "127.0.0.1:0",
			ModelID:                 "mock-chat-model",
			DefaultCompletionTokens: 8,
			MaxCompletionTokens:     64,
			StreamTokensPerChunk:    2,
			ResponseStoreTTL:        time.Hour,
			ResponseStoreMaxEntries: 1_000,
			ResponseStoreMaxBytes:   8 << 20,
		})
	}()

	select {
	case <-output.ready:
		cancel()
	case <-time.After(3 * time.Second):
		t.Fatal("Run() did not start the HTTP server")
	}

	select {
	case err := <-errors:
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Run() did not stop the HTTP server")
	}

	if got := output.String(); !strings.HasPrefix(got, "test-app listening on 127.0.0.1:") {
		t.Fatalf("Run() output = %q", got)
	}
}

func TestRunWithDiagnostics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	output := notifyingWriter{ready: make(chan struct{})}
	errors := make(chan error, 1)

	go func() {
		errors <- Run(ctx, &output, config.Config{
			Name:                    "test-app",
			HTTPAddress:             "127.0.0.1:0",
			PPROFAddress:            "127.0.0.1:0",
			ModelID:                 "mock-chat-model",
			DefaultCompletionTokens: 8,
			MaxCompletionTokens:     64,
			StreamTokensPerChunk:    2,
			ResponseStoreTTL:        time.Hour,
			ResponseStoreMaxEntries: 1_000,
			ResponseStoreMaxBytes:   8 << 20,
		})
	}()

	select {
	case <-output.ready:
		deadline := time.After(3 * time.Second)
		for !strings.Contains(output.String(), "diagnostics listening on 127.0.0.1:") {
			select {
			case <-deadline:
				t.Fatal("Run() did not report the diagnostics server")
			case <-time.After(time.Millisecond):
			}
		}
		cancel()
	case <-time.After(3 * time.Second):
		t.Fatal("Run() did not start the servers")
	}

	select {
	case err := <-errors:
		if err != nil {
			t.Fatalf("Run() error = %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Run() did not stop the servers")
	}
}

type notifyingWriter struct {
	bytes.Buffer
	mu    sync.Mutex
	once  sync.Once
	ready chan struct{}
}

func (w *notifyingWriter) Write(value []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	written, err := w.Buffer.Write(value)
	w.once.Do(func() { close(w.ready) })
	return written, err
}

func (w *notifyingWriter) String() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.Buffer.String()
}
