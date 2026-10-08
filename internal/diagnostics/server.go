// Package diagnostics exposes optional process diagnostics on a private HTTP server.
package diagnostics

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/pprof"
	runtimemetrics "runtime/metrics"
	"time"
)

// Server is a running diagnostics HTTP server.
type Server struct {
	server   *http.Server
	listener net.Listener
	done     chan error
}

// Start starts a diagnostics server on address.
func Start(address string) (*Server, error) {
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return nil, err
	}

	server := &Server{
		server: &http.Server{
			Handler:           NewHandler(),
			ReadHeaderTimeout: 5 * time.Second,
		},
		listener: listener,
		done:     make(chan error, 1),
	}
	go func() {
		err := server.server.Serve(listener)
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		server.done <- err
	}()
	return server, nil
}

// Addr returns the bound listener address.
func (s *Server) Addr() net.Addr {
	return s.listener.Addr()
}

// Done reports the terminal server error, or nil after a graceful shutdown.
func (s *Server) Done() <-chan error {
	return s.done
}

// Shutdown gracefully stops the diagnostics server.
func (s *Server) Shutdown(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

// NewHandler returns a private mux with pprof and runtime metric endpoints.
func NewHandler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /debug/pprof/", pprof.Index)
	mux.HandleFunc("GET /debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("GET /debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("GET /debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("POST /debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("GET /debug/pprof/trace", pprof.Trace)
	for _, name := range []string{"allocs", "block", "goroutine", "heap", "mutex", "threadcreate"} {
		mux.Handle("GET /debug/pprof/"+name, pprof.Handler(name))
	}
	mux.HandleFunc("GET /debug/metrics", writeRuntimeMetrics)
	return mux
}

var runtimeMetricNames = []string{
	"/cpu/classes/gc/total:cpu-seconds",
	"/cpu/classes/user:cpu-seconds",
	"/gc/cycles/total:gc-cycles",
	"/gc/heap/allocs:bytes",
	"/gc/heap/objects:objects",
	"/memory/classes/heap/released:bytes",
	"/memory/classes/heap/objects:bytes",
	"/memory/classes/heap/stacks:bytes",
	"/memory/classes/total:bytes",
	"/sched/goroutines:goroutines",
}

type runtimeMetricSnapshot struct {
	Timestamp time.Time      `json:"timestamp"`
	Values    map[string]any `json:"values"`
}

func writeRuntimeMetrics(w http.ResponseWriter, _ *http.Request) {
	samples := make([]runtimemetrics.Sample, len(runtimeMetricNames))
	for index, name := range runtimeMetricNames {
		samples[index].Name = name
	}
	runtimemetrics.Read(samples)

	values := make(map[string]any, len(samples))
	for _, sample := range samples {
		switch sample.Value.Kind() {
		case runtimemetrics.KindUint64:
			values[sample.Name] = sample.Value.Uint64()
		case runtimemetrics.KindFloat64:
			values[sample.Name] = sample.Value.Float64()
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(runtimeMetricSnapshot{
		Timestamp: time.Now().UTC(),
		Values:    values,
	})
}
