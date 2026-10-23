package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func TestChatIDsAreUniqueAcrossProcessesAndRestarts(t *testing.T) {
	t.Parallel()
	seen := make(map[string]bool)
	for process := range 3 {
		command := exec.Command(os.Args[0], "-test.run=^TestChatIDProcessHelper$")
		command.Env = append(os.Environ(), "MOCK_AI_IDENTITY_HELPER=1")
		output, err := command.Output()
		if err != nil {
			t.Fatalf("process %d: %v", process, err)
		}
		var ids []string
		if err := json.Unmarshal(output, &ids); err != nil {
			t.Fatalf("process %d JSON: %v", process, err)
		}
		if len(ids) != 16 {
			t.Fatalf("process %d returned %d IDs, want 16", process, len(ids))
		}
		for _, id := range ids {
			if seen[id] {
				t.Fatalf("response ID reused across processes or handlers: %q", id)
			}
			seen[id] = true
		}
	}
}

func TestChatIDProcessHelper(t *testing.T) {
	if os.Getenv("MOCK_AI_IDENTITY_HELPER") != "1" {
		return
	}
	model, err := chat.NewMockModel("mock-chat-model", chat.MockOptions{
		DefaultCompletionTokens: 4,
		MaxCompletionTokens:     8,
		StreamTokensPerChunk:    2,
	})
	if err != nil {
		t.Fatal(err)
	}
	ids := make([]string, 0, 16)
	// Each independent handler starts its sequence at one.
	for range 2 {
		handler := newTestHandler(t, model)
		for request := range 8 {
			stream := request%2 == 1
			fields := fmt.Sprintf(`"model":"mock-chat-model","messages":[{"role":"user","content":"Hello"}],"seed":42,"stream":%t`, stream)
			if stream {
				fields += `,"stream_options":{"include_usage":true}`
			}
			if request%4 >= 2 {
				fields += `,"tool_choice":"required","tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"const":"Rome"}},"required":["city"]}}}]`
			}
			body := "{" + fields + "}"
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
			if recorder.Code != http.StatusOK {
				t.Fatalf("status=%d", recorder.Code)
			}
			var response struct {
				ID string `json:"id"`
			}
			if stream {
				for _, event := range parseSSEEvents(t, recorder.Body.String()) {
					if event == "[DONE]" {
						continue
					}
					var chunk struct {
						ID string `json:"id"`
					}
					if err := json.Unmarshal([]byte(event), &chunk); err != nil {
						t.Fatal(err)
					}
					if response.ID != "" && response.ID != chunk.ID {
						t.Fatal("ID changed within one stream")
					}
					response.ID = chunk.ID
				}
			} else if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if !strings.HasPrefix(response.ID, "chatcmpl-") {
				t.Fatalf("invalid response ID: %q", response.ID)
			}
			ids = append(ids, response.ID)
		}
	}
	if err := json.NewEncoder(os.Stdout).Encode(ids); err != nil {
		t.Fatal(err)
	}
	os.Exit(0)
}
