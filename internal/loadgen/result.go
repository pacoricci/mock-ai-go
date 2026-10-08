package loadgen

import (
	"runtime"
	"runtime/debug"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Result is the machine-readable outcome of a scenario.
type Result struct {
	Scenario    Scenario    `json:"scenario"`
	Environment Environment `json:"environment"`
	StartedAt   time.Time   `json:"started_at"`
	// FinishedAt marks the end of the workload, before final metrics collection.
	FinishedAt time.Time `json:"finished_at"`
	// ElapsedSeconds includes ramp-up and streaming, excluding metrics cleanup.
	ElapsedSeconds    float64          `json:"elapsed_seconds"`
	EventsPerSecond   float64          `json:"events_per_second"`
	BytesPerSecond    float64          `json:"bytes_per_second"`
	Attempted         uint64           `json:"attempted"`
	Connected         uint64           `json:"connected"`
	MaxActive         uint64           `json:"max_active"`
	Completed         uint64           `json:"completed"`
	Canceled          uint64           `json:"canceled"`
	Errors            uint64           `json:"errors"`
	Events            uint64           `json:"events"`
	Bytes             uint64           `json:"bytes"`
	UsageChunks       uint64           `json:"usage_chunks"`
	InvalidEvents     uint64           `json:"invalid_events"`
	InvalidUsage      uint64           `json:"invalid_usage"`
	OpenLatencyMillis LatencySummary   `json:"open_latency_ms"`
	ContentGapMillis  LatencySummary   `json:"content_gap_ms"`
	ServerMetrics     *MetricSummary   `json:"server_metrics,omitempty"`
	MemoryPerStream   *MemoryPerStream `json:"memory_per_active_stream,omitempty"`
	ErrorSamples      []string         `json:"error_samples,omitempty"`
}

// Environment identifies the executable and runtime used for a run.
type Environment struct {
	GoVersion    string `json:"go_version"`
	OS           string `json:"os"`
	Architecture string `json:"architecture"`
	GitCommit    string `json:"git_commit,omitempty"`
	GitModified  bool   `json:"git_modified,omitempty"`
}

// LatencySummary contains stream-opening latency percentiles in milliseconds.
type LatencySummary struct {
	Count uint64  `json:"count"`
	Min   float64 `json:"min"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	P99   float64 `json:"p99"`
	Max   float64 `json:"max"`
}

// MetricSummary contains periodic samples from the server diagnostics endpoint.
type MetricSummary struct {
	URL          string             `json:"url"`
	Samples      uint64             `json:"samples"`
	Start        map[string]float64 `json:"start"`
	End          map[string]float64 `json:"end"`
	Peak         map[string]float64 `json:"peak"`
	Delta        map[string]float64 `json:"delta"`
	ScrapeErrors uint64             `json:"scrape_errors"`
}

// MemoryPerStream is the increase from baseline to peak divided by max active streams.
type MemoryPerStream struct {
	HeapObjectBytes float64 `json:"heap_object_bytes"`
	StackBytes      float64 `json:"stack_bytes"`
	GoResidentBytes float64 `json:"go_resident_estimate_bytes"`
}

type counters struct {
	attempted     atomic.Uint64
	connected     atomic.Uint64
	active        atomic.Uint64
	maxActive     atomic.Uint64
	completed     atomic.Uint64
	canceled      atomic.Uint64
	errors        atomic.Uint64
	events        atomic.Uint64
	bytes         atomic.Uint64
	usageChunks   atomic.Uint64
	invalidEvents atomic.Uint64
	invalidUsage  atomic.Uint64

	mu            sync.Mutex
	openLatencies []time.Duration
	contentGaps   []time.Duration
	errorSamples  []string
}

func (c *counters) connectionOpened(latency time.Duration) {
	c.connected.Add(1)
	active := c.active.Add(1)
	for {
		maximum := c.maxActive.Load()
		if active <= maximum || c.maxActive.CompareAndSwap(maximum, active) {
			break
		}
	}
	c.mu.Lock()
	c.openLatencies = append(c.openLatencies, latency)
	c.mu.Unlock()
}

func (c *counters) connectionClosed() {
	c.active.Add(^uint64(0))
}

func (c *counters) addError(err error) {
	c.errors.Add(1)
	c.mu.Lock()
	if len(c.errorSamples) < 10 {
		c.errorSamples = append(c.errorSamples, err.Error())
	}
	c.mu.Unlock()
}

func (c *counters) addContentGaps(gaps []time.Duration) {
	if len(gaps) == 0 {
		return
	}
	c.mu.Lock()
	c.contentGaps = append(c.contentGaps, gaps...)
	c.mu.Unlock()
}

func (c *counters) result(scenario Scenario, startedAt, finishedAt time.Time, elapsed time.Duration, serverMetrics *MetricSummary) Result {
	c.mu.Lock()
	latencies := append([]time.Duration(nil), c.openLatencies...)
	contentGaps := append([]time.Duration(nil), c.contentGaps...)
	errorSamples := append([]string(nil), c.errorSamples...)
	c.mu.Unlock()

	result := Result{
		Scenario:          scenario,
		Environment:       currentEnvironment(),
		StartedAt:         startedAt,
		FinishedAt:        finishedAt,
		ElapsedSeconds:    elapsed.Seconds(),
		Attempted:         c.attempted.Load(),
		Connected:         c.connected.Load(),
		MaxActive:         c.maxActive.Load(),
		Completed:         c.completed.Load(),
		Canceled:          c.canceled.Load(),
		Errors:            c.errors.Load(),
		Events:            c.events.Load(),
		Bytes:             c.bytes.Load(),
		UsageChunks:       c.usageChunks.Load(),
		InvalidEvents:     c.invalidEvents.Load(),
		InvalidUsage:      c.invalidUsage.Load(),
		OpenLatencyMillis: summarizeLatencies(latencies),
		ContentGapMillis:  summarizeLatencies(contentGaps),
		ServerMetrics:     serverMetrics,
		ErrorSamples:      errorSamples,
	}
	if elapsed > 0 {
		result.EventsPerSecond = float64(result.Events) / elapsed.Seconds()
		result.BytesPerSecond = float64(result.Bytes) / elapsed.Seconds()
	}
	if serverMetrics != nil && result.MaxActive > 0 {
		streams := float64(result.MaxActive)
		result.MemoryPerStream = &MemoryPerStream{
			HeapObjectBytes: positiveDelta(serverMetrics, "/memory/classes/heap/objects:bytes") / streams,
			StackBytes:      positiveDelta(serverMetrics, "/memory/classes/heap/stacks:bytes") / streams,
			GoResidentBytes: positiveDelta(serverMetrics, derivedGoResidentMetric) / streams,
		}
	}
	return result
}

func positiveDelta(metrics *MetricSummary, name string) float64 {
	delta := metrics.Peak[name] - metrics.Start[name]
	if delta < 0 {
		return 0
	}
	return delta
}

func currentEnvironment() Environment {
	environment := Environment{
		GoVersion:    runtime.Version(),
		OS:           runtime.GOOS,
		Architecture: runtime.GOARCH,
	}
	if buildInfo, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range buildInfo.Settings {
			switch setting.Key {
			case "vcs.revision":
				environment.GitCommit = setting.Value
			case "vcs.modified":
				environment.GitModified = setting.Value == "true"
			}
		}
	}
	return environment
}

func summarizeLatencies(values []time.Duration) LatencySummary {
	if len(values) == 0 {
		return LatencySummary{}
	}
	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	return LatencySummary{
		Count: uint64(len(values)),
		Min:   milliseconds(values[0]),
		P50:   milliseconds(percentile(values, 0.50)),
		P95:   milliseconds(percentile(values, 0.95)),
		P99:   milliseconds(percentile(values, 0.99)),
		Max:   milliseconds(values[len(values)-1]),
	}
}

func percentile(values []time.Duration, quantile float64) time.Duration {
	index := int(float64(len(values)-1) * quantile)
	return values[index]
}

func milliseconds(value time.Duration) float64 {
	return float64(value) / float64(time.Millisecond)
}
