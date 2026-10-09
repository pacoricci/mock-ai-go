// Package app coordinates the application lifecycle.
package app

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"time"

	"github.com/pacoricci/mock-ai-go/docs"
	"github.com/pacoricci/mock-ai-go/internal/anthropic"
	"github.com/pacoricci/mock-ai-go/internal/chat"
	"github.com/pacoricci/mock-ai-go/internal/config"
	"github.com/pacoricci/mock-ai-go/internal/diagnostics"
	"github.com/pacoricci/mock-ai-go/internal/embedding"
	"github.com/pacoricci/mock-ai-go/internal/openai"
)

// Server owns the application listeners and their lifecycle. Callers that use
// Start must call Shutdown and may observe unexpected termination through Done.
type Server struct {
	httpServer        *http.Server
	listener          net.Listener
	diagnosticsServer *diagnostics.Server
	done              chan error
}

// Start starts the application listeners and returns without waiting for them
// to terminate. Address methods return the resolved listener addresses, which
// is useful when the configuration requests an ephemeral port.
func Start(output io.Writer, cfg config.Config) (*Server, error) {
	chatModel, err := chat.NewMockModel(cfg.ModelID, chat.MockOptions{
		DefaultCompletionTokens: cfg.DefaultCompletionTokens,
		MaxCompletionTokens:     cfg.MaxCompletionTokens,
		MinTextTokens:           cfg.MinTextTokens,
		MaxTextTokens:           cfg.MaxTextTokens,
		StreamTokensPerChunk:    cfg.StreamTokensPerChunk,
		StreamDelay:             cfg.StreamDelay,
		StreamDelayJitter:       cfg.StreamDelayJitter,
		StreamInitialDelay:      cfg.StreamInitialDelay,
	})
	if err != nil {
		return nil, fmt.Errorf("create chat model: %w", err)
	}
	registry, err := chat.NewRegistry(chatModel)
	if err != nil {
		return nil, fmt.Errorf("create chat registry: %w", err)
	}
	handler, err := openai.NewHandler(registry, openai.ResponseStoreOptions{
		TTL:        cfg.ResponseStoreTTL,
		MaxEntries: cfg.ResponseStoreMaxEntries,
		MaxBytes:   cfg.ResponseStoreMaxBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("create OpenAI handler: %w", err)
	}
	anthropicHandler, err := anthropic.NewHandler(registry)
	if err != nil {
		return nil, fmt.Errorf("create Anthropic handler: %w", err)
	}

	embeddingModel, err := embedding.NewMockModel(cfg.EmbeddingModelID, cfg.EmbeddingDimensions)
	if err != nil {
		return nil, fmt.Errorf("create embedding model: %w", err)
	}
	embeddingHandler, err := openai.NewEmbeddingsHandler(embeddingModel)
	if err != nil {
		return nil, fmt.Errorf("create embeddings handler: %w", err)
	}

	listener, err := net.Listen("tcp", cfg.HTTPAddress)
	if err != nil {
		return nil, fmt.Errorf("listen on %s: %w", cfg.HTTPAddress, err)
	}

	var diagnosticsServer *diagnostics.Server
	if cfg.PPROFAddress != "" {
		diagnosticsServer, err = diagnostics.Start(cfg.PPROFAddress)
		if err != nil {
			_ = listener.Close()
			return nil, fmt.Errorf("listen for diagnostics on %s: %w", cfg.PPROFAddress, err)
		}
	}

	if _, err := fmt.Fprintf(output, "%s listening on %s\n", cfg.Name, listener.Addr()); err != nil {
		shutdownDiagnostics(diagnosticsServer)
		_ = listener.Close()
		return nil, fmt.Errorf("write startup message: %w", err)
	}
	if diagnosticsServer != nil {
		if _, err := fmt.Fprintf(output, "diagnostics listening on %s\n", diagnosticsServer.Addr()); err != nil {
			shutdownDiagnostics(diagnosticsServer)
			_ = listener.Close()
			return nil, fmt.Errorf("write diagnostics startup message: %w", err)
		}
	}

	httpServer := &http.Server{
		Handler:           newHTTPHandler(handler, anthropicHandler, embeddingHandler),
		ReadHeaderTimeout: 5 * time.Second,
	}
	server := &Server{
		httpServer:        httpServer,
		listener:          listener,
		diagnosticsServer: diagnosticsServer,
		done:              make(chan error, 1),
	}
	httpDone := make(chan error, 1)
	go func() {
		httpDone <- httpServer.Serve(listener)
	}()
	go server.observe(httpDone)
	return server, nil
}

// Run starts the application and blocks until its work is complete.
func Run(ctx context.Context, output io.Writer, cfg config.Config) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("application context: %w", err)
	}

	server, err := Start(output, cfg)
	if err != nil {
		return err
	}

	select {
	case err := <-server.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if shutdownErr := server.Shutdown(shutdownContext); shutdownErr != nil && err == nil {
			return fmt.Errorf("shutdown servers: %w", shutdownErr)
		}
		return err
	case <-ctx.Done():
		shutdownContext, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownContext); err != nil {
			return fmt.Errorf("shutdown servers: %w", err)
		}
		return nil
	}
}

// HTTPAddress returns the resolved public API listener address.
func (s *Server) HTTPAddress() string {
	return s.listener.Addr().String()
}

// DiagnosticsAddress returns the resolved private diagnostics listener address,
// or an empty string when diagnostics are disabled.
func (s *Server) DiagnosticsAddress() string {
	if s.diagnosticsServer == nil {
		return ""
	}
	return s.diagnosticsServer.Addr().String()
}

// Done reports the first unexpected listener termination. The channel produces
// exactly one value; a nil value means the server was shut down normally.
func (s *Server) Done() <-chan error {
	return s.done
}

// Shutdown gracefully stops every listener owned by the server.
func (s *Server) Shutdown(ctx context.Context) error {
	return errors.Join(
		s.httpServer.Shutdown(ctx),
		shutdownDiagnosticsWithContext(ctx, s.diagnosticsServer),
	)
}

func (s *Server) observe(httpDone <-chan error) {
	var diagnosticsDone <-chan error
	if s.diagnosticsServer != nil {
		diagnosticsDone = s.diagnosticsServer.Done()
	}

	select {
	case err := <-httpDone:
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		} else if err != nil {
			err = fmt.Errorf("serve HTTP: %w", err)
		}
		s.done <- err
	case err := <-diagnosticsDone:
		if err != nil {
			err = fmt.Errorf("serve diagnostics: %w", err)
		}
		s.done <- err
	}
}

func newHTTPHandler(openAIHandler, anthropicHandler, embeddingHandler http.Handler) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/v1/embeddings", embeddingHandler)
	mux.Handle("/v1/embeddings/", embeddingHandler)
	mux.Handle("/v1/messages", anthropicHandler)
	mux.Handle("/v1/messages/", anthropicHandler)
	mux.Handle("/v1/", openAIHandler)
	mux.Handle("/", docs.NewHandler())
	return mux
}

func shutdownDiagnostics(server *diagnostics.Server) {
	if server == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = server.Shutdown(ctx)
}

func shutdownDiagnosticsWithContext(ctx context.Context, server *diagnostics.Server) error {
	if server == nil {
		return nil
	}
	return server.Shutdown(ctx)
}
