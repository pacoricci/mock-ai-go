package loadgen

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

const maxSSEEventBytes = 1 << 20

// Run executes a scenario and waits for all streams to finish or be canceled.
func Run(ctx context.Context, scenario Scenario) (Result, error) {
	if err := scenario.Validate(); err != nil {
		return Result{}, err
	}

	body, err := marshalRequest(scenario)
	if err != nil {
		return Result{}, fmt.Errorf("encode request: %w", err)
	}

	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		ForceAttemptHTTP2:     true,
		MaxIdleConns:          scenario.Connections,
		MaxIdleConnsPerHost:   scenario.Connections,
		MaxConnsPerHost:       scenario.Connections,
		IdleConnTimeout:       30 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		DisableCompression:    true,
	}
	client := &http.Client{Transport: transport}
	defer transport.CloseIdleConnections()
	metricCollector := newMetricCollector(scenario.MetricsURL)
	if err := metricCollector.initialSample(ctx); err != nil {
		return Result{}, err
	}

	started := time.Now()
	startedAt := started.UTC()
	deadline := started.Add(time.Duration(scenario.RampUp) + time.Duration(scenario.Duration))
	runContext, cancel := context.WithDeadline(ctx, deadline)
	metricsDone := make(chan struct{})
	go metricCollector.run(runContext, metricsDone)

	var counts counters
	counts.openLatencies = make([]time.Duration, 0, scenario.Connections)
	var workers sync.WaitGroup

	for index := 0; index < scenario.Connections; index++ {
		if err := waitForRamp(runContext, started, time.Duration(scenario.RampUp), index, scenario.Connections); err != nil {
			break
		}
		counts.attempted.Add(1)
		workers.Add(1)
		go func() {
			defer workers.Done()
			runConnection(runContext, client, scenario, body, &counts)
		}()
	}

	workers.Wait()
	finished := time.Now()
	cancel()
	<-metricsDone
	metricCollector.finalSample()
	finishedAt := finished.UTC()
	return counts.result(scenario, startedAt, finishedAt, finished.Sub(started), metricCollector.summary()), nil
}

func marshalRequest(scenario Scenario) ([]byte, error) {
	if scenario.effectiveProtocol() == ProtocolResponses {
		request := map[string]any{
			"model":             scenario.Request.Model,
			"input":             scenario.Request.Prompt,
			"max_output_tokens": scenario.Request.MaxCompletionTokens,
			"stream":            true,
		}
		if scenario.Request.Store != nil {
			request["store"] = *scenario.Request.Store
		}
		return json.Marshal(request)
	}
	return json.Marshal(map[string]any{
		"model":                 scenario.Request.Model,
		"messages":              []map[string]string{{"role": "user", "content": scenario.Request.Prompt}},
		"max_completion_tokens": scenario.Request.MaxCompletionTokens,
		"seed":                  scenario.Request.Seed,
		"stream":                true,
		"stream_options":        map[string]bool{"include_usage": scenario.Request.IncludeUsage},
	})
}

func waitForRamp(ctx context.Context, startedAt time.Time, ramp time.Duration, index, connections int) error {
	if ramp == 0 || connections <= 1 {
		return ctx.Err()
	}
	target := startedAt.Add(time.Duration(index) * ramp / time.Duration(connections-1))
	delay := time.Until(target)
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return ctx.Err()
	case <-ctx.Done():
		return ctx.Err()
	}
}

func runConnection(ctx context.Context, client *http.Client, scenario Scenario, body []byte, counts *counters) {
	var contentTiming contentTiming
	defer contentTiming.commit(counts)

	connectionContext := ctx
	cancel := func() {}
	if lifetime := time.Duration(scenario.ConnectionLifetime); lifetime > 0 {
		connectionContext, cancel = context.WithTimeout(ctx, lifetime)
	}
	defer cancel()

	request, err := http.NewRequestWithContext(connectionContext, http.MethodPost, scenario.URL, bytes.NewReader(body))
	if err != nil {
		counts.addError(fmt.Errorf("create request: %w", err))
		return
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "text/event-stream")

	openedAt := time.Now()
	response, err := client.Do(request)
	if err != nil {
		if connectionContext.Err() != nil {
			counts.canceled.Add(1)
			return
		}
		counts.addError(fmt.Errorf("open stream: %w", err))
		return
	}
	defer func() { _ = response.Body.Close() }()

	if response.StatusCode != http.StatusOK {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		counts.addError(fmt.Errorf("open stream: status %s: %s", response.Status, strings.TrimSpace(string(message))))
		return
	}
	if contentType := response.Header.Get("Content-Type"); !strings.HasPrefix(contentType, "text/event-stream") {
		counts.addError(fmt.Errorf("open stream: unexpected Content-Type %q", contentType))
		return
	}

	counts.connectionOpened(time.Since(openedAt))
	defer counts.connectionClosed()

	reader := &countingReader{reader: response.Body, counts: counts}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 4096), maxSSEEventBytes)
	var eventName string
	validation := eventValidation{protocol: scenario.effectiveProtocol()}
	for scanner.Scan() {
		line := scanner.Bytes()
		if bytes.HasPrefix(line, []byte("event:")) {
			eventName = string(bytes.TrimSpace(bytes.TrimPrefix(line, []byte("event:"))))
			continue
		}
		if !bytes.HasPrefix(line, []byte("data:")) {
			if len(line) == 0 {
				eventName = ""
			}
			continue
		}
		payload := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))
		counts.events.Add(1)
		if bytes.Equal(payload, []byte("[DONE]")) {
			if validation.protocol == ProtocolResponses && !validation.terminalSeen {
				counts.invalidEvents.Add(1)
			}
			if validation.protocol == ProtocolChatCompletions || (validation.completed && !validation.failureSeen) {
				counts.completed.Add(1)
			}
			return
		}
		if validation.validate(eventName, payload, counts) {
			contentTiming.observe(time.Now())
		}
		eventName = ""
		if delay := time.Duration(scenario.ReadDelay); delay > 0 {
			if err := wait(connectionContext, delay); err != nil {
				counts.canceled.Add(1)
				return
			}
		}
	}

	if err := scanner.Err(); err != nil {
		if connectionContext.Err() != nil {
			counts.canceled.Add(1)
			return
		}
		counts.addError(fmt.Errorf("read stream: %w", err))
		return
	}
	if connectionContext.Err() != nil {
		counts.canceled.Add(1)
		return
	}
	counts.addError(errors.New("read stream: closed before [DONE]"))
}

type usageEvent struct {
	Choices []struct {
		Delta struct {
			Role    *string `json:"role"`
			Content *string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func validateChatEvent(payload []byte, counts *counters) bool {
	if !json.Valid(payload) {
		counts.invalidEvents.Add(1)
		return false
	}
	var event usageEvent
	if err := json.Unmarshal(payload, &event); err != nil {
		counts.invalidEvents.Add(1)
		return false
	}
	if event.Usage != nil {
		counts.usageChunks.Add(1)
		if event.Usage.PromptTokens < 0 || event.Usage.CompletionTokens < 0 ||
			event.Usage.TotalTokens != event.Usage.PromptTokens+event.Usage.CompletionTokens {
			counts.invalidUsage.Add(1)
		}
	}
	for _, choice := range event.Choices {
		if choice.Delta.Role == nil && choice.Delta.Content != nil {
			return true
		}
	}
	return false
}

type eventValidation struct {
	protocol     Protocol
	nextSequence int
	terminalSeen bool
	completed    bool
	failureSeen  bool
}

type responseEvent struct {
	Type           string  `json:"type"`
	SequenceNumber *int    `json:"sequence_number"`
	Delta          *string `json:"delta"`
	Code           string  `json:"code"`
	Message        string  `json:"message"`
	Response       *struct {
		Error *struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
		Usage *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	} `json:"response"`
}

func (v *eventValidation) validate(eventName string, payload []byte, counts *counters) bool {
	if v.protocol == ProtocolChatCompletions {
		return validateChatEvent(payload, counts)
	}
	if !json.Valid(payload) {
		counts.invalidEvents.Add(1)
		return false
	}
	var event responseEvent
	if err := json.Unmarshal(payload, &event); err != nil || !validResponseEventType(event.Type) ||
		eventName != event.Type || event.SequenceNumber == nil || *event.SequenceNumber != v.nextSequence {
		counts.invalidEvents.Add(1)
		return false
	}
	v.nextSequence++

	switch event.Type {
	case "response.completed":
		v.terminalSeen = true
		v.completed = true
		if event.Response == nil || event.Response.Usage == nil {
			counts.invalidUsage.Add(1)
			return false
		}
		usage := event.Response.Usage
		counts.usageChunks.Add(1)
		if usage.InputTokens < 0 || usage.OutputTokens < 0 || usage.TotalTokens != usage.InputTokens+usage.OutputTokens {
			counts.invalidUsage.Add(1)
		}
	case "response.failed", "response.incomplete":
		v.terminalSeen = true
		code, message := "", ""
		if event.Response != nil {
			if event.Response.Error != nil {
				code, message = event.Response.Error.Code, event.Response.Error.Message
			}
			if event.Response.IncompleteDetails != nil {
				message = event.Response.IncompleteDetails.Reason
			}
		}
		v.recordFailure(counts, event.Type, code, message)
	case "error":
		v.recordFailure(counts, event.Type, event.Code, event.Message)
	}
	return event.Type == "response.output_text.delta" && event.Delta != nil
}

func (v *eventValidation) recordFailure(counts *counters, eventType, code, message string) {
	if v.failureSeen {
		return
	}
	v.failureSeen = true
	counts.addError(fmt.Errorf("response stream: %s: %s", eventType, strings.TrimSpace(code+" "+message)))
}

func validResponseEventType(eventType string) bool {
	switch eventType {
	case "response.created", "response.in_progress", "response.output_item.added",
		"response.content_part.added", "response.output_text.delta", "response.output_text.done",
		"response.content_part.done", "response.output_item.done", "response.completed",
		"response.failed", "response.incomplete", "error":
		return true
	default:
		return false
	}
}

type contentTiming struct {
	last time.Time
	gaps []time.Duration
}

func (t *contentTiming) observe(now time.Time) {
	if !t.last.IsZero() {
		t.gaps = append(t.gaps, now.Sub(t.last))
	}
	t.last = now
}

func (t *contentTiming) commit(counts *counters) {
	counts.addContentGaps(t.gaps)
}

func wait(ctx context.Context, duration time.Duration) error {
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

type countingReader struct {
	reader io.Reader
	counts *counters
}

func (r *countingReader) Read(buffer []byte) (int, error) {
	read, err := r.reader.Read(buffer)
	r.counts.bytes.Add(uint64(read))
	return read, err
}
