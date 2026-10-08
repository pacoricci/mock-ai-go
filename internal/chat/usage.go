package chat

const (
	messageTokenOverhead = 4
	replyTokenOverhead   = 3
	bytesPerToken        = 4
)

// EstimatePromptTokens returns a fast deterministic estimate suitable for a
// mock server. It deliberately does not claim compatibility with a real model
// tokenizer.
func EstimatePromptTokens(messages []Message) int {
	tokens := replyTokenOverhead
	for _, message := range messages {
		tokens += messageTokenOverhead
		tokens += estimateTextTokens(message.Content)
	}
	return tokens
}

func estimateTextTokens(value string) int {
	if value == "" {
		return 0
	}
	return (len(value) + bytesPerToken - 1) / bytesPerToken
}

func newUsage(messages []Message, completionTokens int) Usage {
	promptTokens := EstimatePromptTokens(messages)
	return Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}
}

func requestUsage(request Request, completionTokens int) (Usage, error) {
	if request.ContextState == nil {
		return newUsage(request.Messages, completionTokens), nil
	}
	promptTokens, err := request.ContextState.PromptTokens()
	if err != nil {
		return Usage{}, err
	}
	if completionTokens > int(^uint(0)>>1)-promptTokens {
		return Usage{}, ErrInvalidContextState
	}
	return Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      promptTokens + completionTokens,
	}, nil
}
