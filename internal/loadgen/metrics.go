package loadgen

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

const metricSampleInterval = time.Second

const derivedGoResidentMetric = "/derived/go-memory-resident-estimate:bytes"

type metricCollector struct {
	url    string
	client *http.Client

	mu      sync.Mutex
	samples []map[string]float64
	errors  uint64
}

func newMetricCollector(url string) *metricCollector {
	return &metricCollector{
		url:    url,
		client: &http.Client{Timeout: 2 * time.Second},
	}
}

func (c *metricCollector) initialSample(ctx context.Context) error {
	if c.url == "" {
		return nil
	}
	values, err := c.scrape(ctx)
	if err != nil {
		return fmt.Errorf("scrape initial server metrics: %w", err)
	}
	c.samples = append(c.samples, values)
	return nil
}

func (c *metricCollector) run(ctx context.Context, done chan<- struct{}) {
	defer close(done)
	if c.url == "" {
		return
	}
	ticker := time.NewTicker(metricSampleInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			c.sample(ctx)
		case <-ctx.Done():
			return
		}
	}
}

func (c *metricCollector) finalSample() {
	if c.url == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	c.sample(ctx)
}

func (c *metricCollector) sample(ctx context.Context) {
	values, err := c.scrape(ctx)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		c.errors++
		return
	}
	c.samples = append(c.samples, values)
}

func (c *metricCollector) scrape(ctx context.Context) (map[string]float64, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, c.url, nil)
	if err != nil {
		return nil, err
	}
	response, err := c.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status %s", response.Status)
	}
	var snapshot struct {
		Values map[string]float64 `json:"values"`
	}
	if err := json.NewDecoder(response.Body).Decode(&snapshot); err != nil {
		return nil, err
	}
	if len(snapshot.Values) == 0 {
		return nil, fmt.Errorf("empty metric snapshot")
	}
	if total, ok := snapshot.Values["/memory/classes/total:bytes"]; ok {
		released := snapshot.Values["/memory/classes/heap/released:bytes"]
		snapshot.Values[derivedGoResidentMetric] = total - released
	}
	return snapshot.Values, nil
}

func (c *metricCollector) summary() *MetricSummary {
	if c.url == "" {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.samples) == 0 {
		return &MetricSummary{URL: c.url, ScrapeErrors: c.errors}
	}

	start := cloneMetrics(c.samples[0])
	end := cloneMetrics(c.samples[len(c.samples)-1])
	peak := cloneMetrics(c.samples[0])
	for _, sample := range c.samples[1:] {
		for name, value := range sample {
			if value > peak[name] {
				peak[name] = value
			}
		}
	}
	delta := make(map[string]float64, len(end))
	for name, value := range end {
		delta[name] = value - start[name]
	}
	return &MetricSummary{
		URL:          c.url,
		Samples:      uint64(len(c.samples)),
		Start:        start,
		End:          end,
		Peak:         peak,
		Delta:        delta,
		ScrapeErrors: c.errors,
	}
}

func cloneMetrics(source map[string]float64) map[string]float64 {
	clone := make(map[string]float64, len(source))
	for name, value := range source {
		clone[name] = value
	}
	return clone
}
