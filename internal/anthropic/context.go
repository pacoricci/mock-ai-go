package anthropic

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

type textBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type toolUseBlock struct {
	Type  string         `json:"type"`
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

type toolResultBlock struct {
	Type      string          `json:"type"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

func contentBlocks(raw json.RawMessage) ([]json.RawMessage, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return nil, fmt.Errorf("content is required")
	}
	if raw[0] == '"' {
		var value string
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, err
		}
		block, err := json.Marshal(textBlock{Type: "text", Text: value})
		return []json.RawMessage{block}, err
	}
	var blocks []json.RawMessage
	if err := json.Unmarshal(raw, &blocks); err != nil || len(blocks) == 0 {
		return nil, fmt.Errorf("content must be a string or nonempty array of blocks")
	}
	return blocks, nil
}

func textParts(raw json.RawMessage) ([]chat.ContextContent, error) {
	blocks, err := contentBlocks(raw)
	if err != nil {
		return nil, err
	}
	parts := make([]chat.ContextContent, 0, len(blocks))
	for _, rawBlock := range blocks {
		var block textBlock
		if err := decodeStrict(rawBlock, &block); err != nil || block.Type != "text" || block.Text == "" {
			return nil, fmt.Errorf("only nonempty text blocks are supported here")
		}
		parts = append(parts, chat.ContextContent{Type: "text", Text: block.Text})
	}
	return parts, nil
}

func resultParts(raw json.RawMessage) ([]chat.ContextContent, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || bytes.Equal(raw, []byte(`""`)) {
		return nil, nil
	}
	if raw[0] == '[' {
		var blocks []json.RawMessage
		if err := json.Unmarshal(raw, &blocks); err == nil && len(blocks) == 0 {
			return nil, nil
		}
	}
	return textParts(raw)
}

type conversationTurn struct {
	role   string
	blocks []json.RawMessage
}

func conversationTurns(messages []inputMessage) ([]conversationTurn, error) {
	turns := make([]conversationTurn, 0, len(messages))
	for index, message := range messages {
		if message.Role != "user" && message.Role != "assistant" {
			return nil, fmt.Errorf("messages.%d.role must be user or assistant; use the top-level system field", index)
		}
		blocks, err := contentBlocks(message.Content)
		if err != nil {
			return nil, fmt.Errorf("messages.%d: %w", index, err)
		}
		if len(turns) > 0 && turns[len(turns)-1].role == message.Role {
			turns[len(turns)-1].blocks = append(turns[len(turns)-1].blocks, blocks...)
		} else {
			turns = append(turns, conversationTurn{role: message.Role, blocks: blocks})
		}
	}
	if turns[0].role != "user" || turns[len(turns)-1].role != "user" {
		return nil, fmt.Errorf("conversation must start and end with user; assistant prefill is unsupported")
	}
	return turns, nil
}

func contextItems(messages []inputMessage) ([]chat.ContextItem, bool, error) {
	turns, err := conversationTurns(messages)
	if err != nil {
		return nil, false, err
	}
	var items []chat.ContextItem
	seen := make(map[string]bool)
	pending := make(map[string]bool)
	lastHasResults := false
	for _, turn := range turns {
		textSeen := false
		lastHasResults = false
		for _, raw := range turn.blocks {
			var kind struct {
				Type string `json:"type"`
			}
			if err := json.Unmarshal(raw, &kind); err != nil {
				return nil, false, fmt.Errorf("invalid content block")
			}
			var item chat.ContextItem
			switch kind.Type {
			case "text":
				var block textBlock
				if err := decodeStrict(raw, &block); err != nil || block.Text == "" {
					return nil, false, fmt.Errorf("text blocks require nonempty text and no unsupported fields")
				}
				textSeen = true
				item = chat.ContextItem{Type: chat.ContextItemMessage, Role: turn.role, Content: []chat.ContextContent{{Type: "text", Text: block.Text}}}
			case "tool_use":
				var block toolUseBlock
				if err := decodeStrict(raw, &block); err != nil || turn.role != "assistant" || strings.TrimSpace(block.ID) == "" || !validToolName(block.Name) || block.Input == nil || seen[block.ID] {
					return nil, false, fmt.Errorf("assistant tool_use requires a unique ID, name, and object input")
				}
				seen[block.ID], pending[block.ID] = true, true
				encoded, err := json.Marshal(block.Input)
				if err != nil {
					return nil, false, err
				}
				item = chat.ContextItem{Type: chat.ContextItemFunctionCall, CallID: block.ID, Name: block.Name, Content: []chat.ContextContent{{Type: "arguments", Text: string(encoded)}}}
			case "tool_result":
				var block toolResultBlock
				if err := decodeStrict(raw, &block); err != nil || turn.role != "user" || textSeen || !pending[block.ToolUseID] {
					return nil, false, fmt.Errorf("tool_result must match an unresolved call in the preceding assistant turn and precede user text")
				}
				parts, err := resultParts(block.Content)
				if err != nil {
					return nil, false, fmt.Errorf("tool_result: %w", err)
				}
				parts = append(parts, chat.ContextContent{Type: "is_error", Text: strconv.FormatBool(block.IsError)})
				item = chat.ContextItem{Type: chat.ContextItemFunctionCallOutput, CallID: block.ToolUseID, Content: parts}
				delete(pending, block.ToolUseID)
				lastHasResults = true
			default:
				return nil, false, fmt.Errorf("unsupported content block type %q", kind.Type)
			}
			items = append(items, item)
		}
		if turn.role == "user" && len(pending) > 0 {
			return nil, false, fmt.Errorf("every tool_use requires a result in the following user turn")
		}
	}
	return items, lastHasResults, nil
}
