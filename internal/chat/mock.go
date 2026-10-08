package chat

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/model"
)

const (
	absoluteMaxCompletionTokens = 1_000_000
	fnvOffsetBasis              = 14_695_981_039_346_656_037
	fnvPrime                    = 1_099_511_628_211
	estimatedBytesPerWord       = 8
	temporalSeedSalt            = 0xd1b54a32d192ed03
)

var (
	// ErrEmptyMockModelID indicates that a mock model has no identifier.
	ErrEmptyMockModelID = errors.New("mock chat model ID is empty")
	// ErrInvalidDefaultCompletionTokens indicates an unusable default output size.
	ErrInvalidDefaultCompletionTokens = errors.New("default completion tokens must be positive")
	// ErrInvalidMaxCompletionTokens indicates an unusable output limit.
	ErrInvalidMaxCompletionTokens = errors.New("max completion tokens is invalid")
	// ErrInvalidStreamChunkTokens indicates that a stream cannot make progress.
	ErrInvalidStreamChunkTokens = errors.New("stream tokens per chunk must be positive")
	// ErrInvalidStreamDelay indicates a negative stream delay.
	ErrInvalidStreamDelay = errors.New("stream delay must not be negative")
	// ErrInvalidStreamDelayJitter indicates unusable stream jitter.
	ErrInvalidStreamDelayJitter = errors.New("stream delay jitter is invalid")
	// ErrInvalidStreamInitialDelay indicates a negative initial stream delay.
	ErrInvalidStreamInitialDelay = errors.New("stream initial delay must not be negative")
	// ErrInvalidCompletionTokens indicates an unusable per-request output size.
	ErrInvalidCompletionTokens = errors.New("completion tokens must be positive")
	// ErrCompletionTokenLimit indicates that a request exceeds the configured limit.
	ErrCompletionTokenLimit = errors.New("completion token limit exceeded")
	// ErrNilEmit indicates that no streamed-chunk consumer was provided.
	ErrNilEmit = errors.New("chat emit function is nil")
)

// MockOptions controls deterministic completion generation.
type MockOptions struct {
	DefaultCompletionTokens int
	MaxCompletionTokens     int
	StreamTokensPerChunk    int
	StreamDelay             time.Duration
	StreamDelayJitter       time.Duration
	StreamInitialDelay      time.Duration
}

// MockModel generates deterministic pseudo-random text with xorshift64*.
// It is immutable and safe for concurrent use.
type MockModel struct {
	info                    model.Info
	defaultCompletionTokens int
	maxCompletionTokens     int
	streamTokensPerChunk    int
	streamDelay             time.Duration
	streamDelayJitter       time.Duration
	streamInitialDelay      time.Duration
}

// NewMockModel constructs a deterministic pseudo-random chat model.
func NewMockModel(id string, options MockOptions) (*MockModel, error) {
	if strings.TrimSpace(id) == "" {
		return nil, ErrEmptyMockModelID
	}
	if options.DefaultCompletionTokens <= 0 {
		return nil, ErrInvalidDefaultCompletionTokens
	}
	if options.MaxCompletionTokens < options.DefaultCompletionTokens || options.MaxCompletionTokens > absoluteMaxCompletionTokens {
		return nil, ErrInvalidMaxCompletionTokens
	}
	if options.StreamTokensPerChunk <= 0 {
		return nil, ErrInvalidStreamChunkTokens
	}
	if options.StreamDelay < 0 {
		return nil, ErrInvalidStreamDelay
	}
	if options.StreamDelayJitter < 0 || options.StreamDelayJitter > time.Duration(1<<63-1)-options.StreamDelay {
		return nil, ErrInvalidStreamDelayJitter
	}
	if options.StreamInitialDelay < 0 {
		return nil, ErrInvalidStreamInitialDelay
	}

	return &MockModel{
		info: model.Info{
			ID:   id,
			Kind: model.KindChat,
		},
		defaultCompletionTokens: options.DefaultCompletionTokens,
		maxCompletionTokens:     options.MaxCompletionTokens,
		streamTokensPerChunk:    options.StreamTokensPerChunk,
		streamDelay:             options.StreamDelay,
		streamDelayJitter:       options.StreamDelayJitter,
		streamInitialDelay:      options.StreamInitialDelay,
	}, nil
}

// Info returns the model metadata.
func (m *MockModel) Info() model.Info {
	return m.info
}

// Complete generates the entire deterministic response in one allocation-sized
// builder, as required by a non-streaming JSON response.
func (m *MockModel) Complete(ctx context.Context, request Request) (Response, error) {
	completionTokens, err := m.completionTokens(request)
	if err != nil {
		return Response{}, err
	}
	var content strings.Builder
	content.Grow(completionTokens * estimatedBytesPerWord)
	usage, err := m.generateText(ctx, request, completionTokens, false, func(chunk Chunk) error {
		_, err := content.Write(chunk.Delta)
		return err
	})
	if err != nil {
		return Response{}, err
	}
	return Response{
		Content: content.String(),
		Usage:   usage,
	}, nil
}

// Stream emits deterministic text incrementally. It keeps only the current
// chunk and PRNG state in memory and never starts an additional goroutine.
func (m *MockModel) Stream(ctx context.Context, request Request, emit Emit) (Usage, error) {
	if emit == nil {
		return Usage{}, ErrNilEmit
	}

	completionTokens, err := m.completionTokens(request)
	if err != nil {
		return Usage{}, err
	}
	return m.generateText(ctx, request, completionTokens, true, emit)
}

func (m *MockModel) generateText(
	ctx context.Context,
	request Request,
	completionTokens int,
	paced bool,
	emit Emit,
) (Usage, error) {
	if emit == nil {
		return Usage{}, ErrNilEmit
	}
	zeroUsage, err := requestUsage(request, 0)
	if err != nil {
		return Usage{}, err
	}
	if err := ctx.Err(); err != nil {
		return zeroUsage, fmt.Errorf("generate chat response: %w", err)
	}

	requestSeed := m.requestSeed(request, completionTokens)
	random := newXorshift64Star(requestSeed)
	var temporalRandom xorshift64Star
	if paced && m.streamDelayJitter > 0 {
		temporalRandom = newXorshift64Star(requestSeed ^ temporalSeedSalt ^ mixSeed(request.StreamNonce))
	}
	emittedTokens := 0
	chunk := make([]byte, 0, m.streamTokensPerChunk*estimatedBytesPerWord)
	var timer reusableTimer
	initialDelay := time.Duration(0)
	gapPacingEnabled := false
	if paced {
		initialDelay = m.streamInitialDelay
		gapPacingEnabled = m.streamDelay > 0 || m.streamDelayJitter > 0
	}
	if initialDelay > 0 || gapPacingEnabled {
		defer timer.stop()
	}

	if initialDelay > 0 {
		if err := timer.wait(ctx, initialDelay); err != nil {
			return zeroUsage, fmt.Errorf("generate chat response: %w", err)
		}
	}

	for emittedTokens < completionTokens {
		if err := ctx.Err(); err != nil {
			usage, usageErr := requestUsage(request, emittedTokens)
			if usageErr != nil {
				return Usage{}, usageErr
			}
			return usage, fmt.Errorf("generate chat response: %w", err)
		}

		chunkTokens := min(m.streamTokensPerChunk, completionTokens-emittedTokens)
		chunk = chunk[:0]
		for range chunkTokens {
			if emittedTokens > 0 {
				chunk = append(chunk, ' ')
			}
			chunk = append(chunk, mockVocabulary[random.next()&uint64(len(mockVocabulary)-1)]...)
			emittedTokens++
		}

		if err := emit(Chunk{Delta: chunk}); err != nil {
			usage, usageErr := requestUsage(request, emittedTokens)
			if usageErr != nil {
				return Usage{}, usageErr
			}
			return usage, fmt.Errorf("emit chat chunk: %w", err)
		}

		if gapPacingEnabled && emittedTokens < completionTokens {
			gap := jitteredStreamDelay(m.streamDelay, m.streamDelayJitter, &temporalRandom)
			if err := timer.wait(ctx, gap); err != nil {
				usage, usageErr := requestUsage(request, emittedTokens)
				if usageErr != nil {
					return Usage{}, usageErr
				}
				return usage, fmt.Errorf("generate chat response: %w", err)
			}
		}
	}

	return requestUsage(request, emittedTokens)
}

type reusableTimer struct {
	timer *time.Timer
}

func (t *reusableTimer) wait(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	if t.timer == nil {
		t.timer = time.NewTimer(delay)
	} else {
		t.timer.Reset(delay)
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.timer.C:
		return nil
	}
}

func (t *reusableTimer) stop() {
	if t.timer != nil {
		t.timer.Stop()
	}
}

func jitteredStreamDelay(delay, jitter time.Duration, random *xorshift64Star) time.Duration {
	if jitter == 0 {
		return delay
	}
	span := uint64(jitter)*2 + 1
	sample := uniformUint64n(random, span)
	return applyJitterSample(delay, jitter, sample)
}

func uniformUint64n(random *xorshift64Star, bound uint64) uint64 {
	if bound <= 1 {
		return 0
	}
	threshold := -bound % bound
	for {
		value := random.next()
		if value >= threshold {
			return value % bound
		}
	}
}

func applyJitterSample(delay, jitter time.Duration, sample uint64) time.Duration {
	offset := int64(sample - uint64(jitter))
	if offset < 0 {
		magnitude := time.Duration(-offset)
		if magnitude >= delay {
			return 0
		}
		return delay - magnitude
	}
	return delay + time.Duration(offset)
}

func (m *MockModel) completionTokens(request Request) (int, error) {
	completionTokens := request.MaxCompletionTokens
	if completionTokens == 0 {
		completionTokens = m.defaultCompletionTokens
	}
	if completionTokens < 0 {
		return 0, ErrInvalidCompletionTokens
	}
	if completionTokens > m.maxCompletionTokens {
		return 0, fmt.Errorf("%w: requested %d, maximum %d", ErrCompletionTokenLimit, completionTokens, m.maxCompletionTokens)
	}
	return completionTokens, nil
}

func (m *MockModel) requestSeed(request Request, completionTokens int) uint64 {
	if request.ContextState != nil {
		hash := uint64(fnvOffsetBasis)
		hashString(&hash, "mock-ai-go/responses-generation/v1")
		hashString(&hash, m.info.ID)
		hashUint64(&hash, uint64(completionTokens))
		hashUint64(&hash, request.ContextState.Digest())
		if request.Seed != nil {
			hashByte(&hash, 1)
			hashUint64(&hash, uint64(*request.Seed))
		}
		return hash
	}

	hash := uint64(fnvOffsetBasis)
	hashString(&hash, m.info.ID)
	hashUint64(&hash, uint64(completionTokens))
	if request.Seed != nil {
		hashByte(&hash, 1)
		hashUint64(&hash, uint64(*request.Seed))
	} else {
		hashByte(&hash, 0)
	}
	for _, message := range request.Messages {
		hashString(&hash, message.Role)
		hashString(&hash, message.Content)
	}
	return hash
}

func hashString(hash *uint64, value string) {
	for index := range len(value) {
		hashByte(hash, value[index])
	}
	hashByte(hash, 0xff)
}

func hashUint64(hash *uint64, value uint64) {
	for range 8 {
		hashByte(hash, byte(value))
		value >>= 8
	}
}

func hashByte(hash *uint64, value byte) {
	*hash ^= uint64(value)
	*hash *= fnvPrime
}

var mockVocabulary = [...]string{
	"amber", "apex", "atlas", "aurora", "birch", "bloom", "brisk", "brook",
	"cedar", "cinder", "cloud", "coral", "cosmos", "crisp", "delta", "drift",
	"ember", "falcon", "fern", "fjord", "flint", "flux", "forest", "frost",
	"glade", "gleam", "grove", "harbor", "haze", "indigo", "iris", "jade",
	"lumen", "lunar", "maple", "meadow", "mist", "nova", "oasis", "ocean",
	"orbit", "pearl", "pine", "prism", "quartz", "raven", "reef", "ridge",
	"river", "sage", "shore", "silver", "solar", "spark", "stone", "summit",
	"tide", "timber", "vale", "velvet", "wave", "willow", "zephyr", "zenith",
}
