package app

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/anthropic"
	"github.com/pacoricci/mock-ai-go/internal/chat"
	"github.com/pacoricci/mock-ai-go/internal/openai"
)

func TestSampledTextAcrossHTTPAPIs(t *testing.T) {
	t.Parallel()
	model, err := chat.NewMockModel("mock", chat.MockOptions{
		DefaultCompletionTokens: 32, MaxCompletionTokens: 4096,
		MinTextTokens: 128, MaxTextTokens: 512, StreamTokensPerChunk: 4,
	})
	if err != nil {
		t.Fatal(err)
	}
	registry, err := chat.NewRegistry(model)
	if err != nil {
		t.Fatal(err)
	}
	openAIHandler, err := openai.NewHandler(registry, openai.ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 1000, MaxBytes: 8 << 20,
	})
	if err != nil {
		t.Fatal(err)
	}
	anthropicHandler, err := anthropic.NewHandler(registry)
	if err != nil {
		t.Fatal(err)
	}
	handler := newHTTPHandler(openAIHandler, anthropicHandler, http.NotFoundHandler())
	for _, endpoint := range []string{"chat/completions", "responses", "messages"} {
		for _, limit := range []int{16, 200, 1024} {
			t.Run(fmt.Sprintf("%s/%d", endpoint, limit), func(t *testing.T) {
				base := fmt.Sprintf(`"model":"mock","messages":[{"role":"user","content":"Explain HTTP streaming"}],"max_completion_tokens":%d,"seed":42`, limit)
				switch endpoint {
				case "responses":
					base = fmt.Sprintf(`"model":"mock","input":"Explain HTTP streaming","max_output_tokens":%d`, limit)
				case "messages":
					base = fmt.Sprintf(`"model":"mock","messages":[{"role":"user","content":"Explain HTTP streaming"}],"max_tokens":%d`, limit)
				}
				var jsonText string
				for _, stream := range []bool{false, true} {
					body := "{" + base + "}"
					if stream {
						body = "{" + base + `,"stream":true`
						if endpoint == "chat/completions" {
							body += `,"stream_options":{"include_usage":true}`
						}
						body += "}"
					}
					request := httptest.NewRequest(http.MethodPost, "/v1/"+endpoint, strings.NewReader(body))
					request.Header.Set("anthropic-version", "2023-06-01")
					recorder := httptest.NewRecorder()
					handler.ServeHTTP(recorder, request)
					if recorder.Code != http.StatusOK {
						t.Fatalf("stream %v: status %d: %s", stream, recorder.Code, recorder.Body.String())
					}
					var text, reason string
					var tokens int
					if stream {
						var content strings.Builder
						for _, line := range strings.Split(recorder.Body.String(), "\n") {
							if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
								continue
							}
							var event lengthEvent
							if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
								t.Fatal(err)
							}
							switch endpoint {
							case "chat/completions":
								for _, choice := range event.Choices {
									content.WriteString(choice.Delta.Content)
								}
							case "responses":
								if event.Type == "response.output_text.delta" {
									var delta string
									if err := json.Unmarshal(event.Delta, &delta); err != nil {
										t.Fatal(err)
									}
									content.WriteString(delta)
								}
							case "messages":
								if len(event.Delta) > 0 {
									var delta struct {
										Text       string
										StopReason string `json:"stop_reason"`
									}
									if err := json.Unmarshal(event.Delta, &delta); err != nil {
										t.Fatal(err)
									}
									content.WriteString(delta.Text)
									if delta.StopReason != "" {
										reason = delta.StopReason
									}
								}
							}
							tokens = max(tokens, event.Usage.CompletionTokens, event.Usage.OutputTokens, event.Response.Usage.OutputTokens)
						}
						text = content.String()
					} else {
						var result lengthEvent
						if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
							t.Fatal(err)
						}
						for _, choice := range result.Choices {
							text += choice.Message.Content
						}
						for _, item := range result.Output {
							for _, part := range item.Content {
								text += part.Text
							}
						}
						for _, part := range result.Content {
							text += part.Text
						}
						tokens = max(result.Usage.CompletionTokens, result.Usage.OutputTokens)
						reason = result.StopReason
						jsonText = text
					}
					upper := min(512, limit)
					if count := len(strings.Fields(text)); count < min(128, upper) || count > upper || tokens != count {
						t.Fatalf("stream %v: word count %d, usage %d, want %d..%d", stream, count, tokens, min(128, upper), upper)
					}
					if endpoint == "messages" {
						wantReason := "end_turn"
						if tokens == limit {
							wantReason = "max_tokens"
						}
						if reason != wantReason {
							t.Fatalf("stream %v: reason %q, want %q", stream, reason, wantReason)
						}
					}
					if stream && endpoint == "chat/completions" && text != jsonText {
						t.Fatal("seeded JSON and SSE content differ")
					}
				}
			})
		}
	}
}

type lengthUsage struct {
	CompletionTokens int `json:"completion_tokens"`
	OutputTokens     int `json:"output_tokens"`
}

type lengthEvent struct {
	Type    string
	Choices []struct {
		Message struct{ Content string }
		Delta   struct{ Content string }
	}
	Content    []struct{ Text string }
	Output     []struct{ Content []struct{ Text string } }
	Delta      json.RawMessage
	Usage      lengthUsage
	Response   struct{ Usage lengthUsage }
	StopReason string `json:"stop_reason"`
}
