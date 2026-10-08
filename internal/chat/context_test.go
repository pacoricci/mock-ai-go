package chat

import "testing"

func TestContextStateAbsorbsOrderedConversationItems(t *testing.T) {
	t.Parallel()

	items := []ContextItem{
		{
			Type:    ContextItemMessage,
			Role:    "user",
			Content: []ContextContent{{Type: "text", Text: "Run the lookup."}},
		},
		{
			Type:    ContextItemFunctionCall,
			CallID:  "call_1",
			Name:    "lookup",
			Content: []ContextContent{{Type: "arguments", Text: `{"query":"amber"}`}},
		},
		{
			Type:    ContextItemFunctionCallOutput,
			CallID:  "call_1",
			Content: []ContextContent{{Type: "output", Text: `{"result":"found"}`}},
		},
		{
			Type: ContextItemReasoning,
			Content: []ContextContent{
				{Type: "summary_text", Text: "The lookup succeeded."},
				{Type: "encrypted_content", Text: "opaque-state"},
			},
		},
	}

	state, err := NewContextState().AbsorbAll(items)
	if err != nil {
		t.Fatalf("AbsorbAll() error = %v", err)
	}
	repeated, err := NewContextState().AbsorbAll(items)
	if err != nil {
		t.Fatalf("AbsorbAll() repeated error = %v", err)
	}
	if state != repeated {
		t.Fatalf("equal item sequences produced different states: %#v and %#v", state, repeated)
	}

	reordered := append([]ContextItem(nil), items...)
	reordered[1], reordered[2] = reordered[2], reordered[1]
	reorderedState, err := NewContextState().AbsorbAll(reordered)
	if err != nil {
		t.Fatalf("AbsorbAll() reordered error = %v", err)
	}
	if state.Digest() == reorderedState.Digest() {
		t.Fatalf("reordered items produced digest %016x", state.Digest())
	}

	changed := append([]ContextItem(nil), items...)
	changed[3] = items[3]
	changed[3].Content = append([]ContextContent(nil), items[3].Content...)
	changed[3].Content[0].Text = "The lookup failed."
	changedState, err := NewContextState().AbsorbAll(changed)
	if err != nil {
		t.Fatalf("AbsorbAll() changed reasoning error = %v", err)
	}
	if state.Digest() == changedState.Digest() {
		t.Fatalf("changed reasoning produced digest %016x", state.Digest())
	}

	wantPromptTokens := replyTokenOverhead
	for _, item := range items {
		units, unitsErr := contextItemUnits(item)
		if unitsErr != nil {
			t.Fatalf("contextItemUnits() error = %v", unitsErr)
		}
		wantPromptTokens += int(units)
	}
	gotPromptTokens, err := state.PromptTokens()
	if err != nil {
		t.Fatalf("PromptTokens() error = %v", err)
	}
	if gotPromptTokens != wantPromptTokens {
		t.Fatalf("PromptTokens() = %d, want %d", gotPromptTokens, wantPromptTokens)
	}
}
