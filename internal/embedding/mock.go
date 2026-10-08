// Package embedding generates deterministic synthetic text embeddings.
package embedding

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"strings"
)

const (
	// DefaultModelID identifies the default synthetic embedding model.
	DefaultModelID = "mock-embedding-model"
	// DefaultDimensions is the default vector length.
	DefaultDimensions = 1536
	// MaxDimensions bounds the configured vector length.
	MaxDimensions = 4096
	// MaxBatchSize bounds the number of inputs per request.
	MaxBatchSize = 2048
	// MaxInputBytes bounds the UTF-8 byte length of each input.
	MaxInputBytes = 32768
	// MaxValues bounds the total number of vector components per request.
	MaxValues = 1 << 20
)

// ValidationError describes an invalid embedding request parameter.
type ValidationError struct {
	Param   string
	Message string
}

// Error returns the validation failure description.
func (e *ValidationError) Error() string { return e.Message }

// MockModel is an immutable, stateless embedding generator safe for concurrent use.
type MockModel struct {
	id         string
	dimensions int
}

// NewMockModel constructs a model. Empty id and zero dimensions select defaults.
func NewMockModel(id string, dimensions int) (*MockModel, error) {
	if id == "" {
		id = DefaultModelID
	}
	if strings.TrimSpace(id) == "" {
		return nil, fmt.Errorf("embedding model ID must not be blank")
	}
	if dimensions == 0 {
		dimensions = DefaultDimensions
	}
	if dimensions < 1 || dimensions > MaxDimensions {
		return nil, fmt.Errorf("embedding dimensions must be between 1 and %d", MaxDimensions)
	}
	return &MockModel{id: id, dimensions: dimensions}, nil
}

// ID returns the configured model identifier.
func (m *MockModel) ID() string { return m.id }

// Dimensions returns the default and maximum requested vector length.
func (m *MockModel) Dimensions() int { return m.dimensions }

// Result contains caller-owned vectors and simulated input-token usage.
type Result struct {
	Vectors      [][]float32
	PromptTokens int
}

// Embed validates the entire batch before generating unit-length vectors.
// Dimensions must be between one and the configured length. Callers must not
// mutate inputs until this returns; returned vectors are independently owned.
// Each vector depends only on model ID, exact input bytes, and dimensions.
func (m *MockModel) Embed(ctx context.Context, inputs []string, dimensions int) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}
	if dimensions < 1 || dimensions > m.dimensions {
		return Result{}, &ValidationError{Param: "dimensions", Message: fmt.Sprintf("dimensions must be between 1 and %d", m.dimensions)}
	}
	if len(inputs) == 0 || len(inputs) > MaxBatchSize {
		return Result{}, &ValidationError{Param: "input", Message: fmt.Sprintf("input must contain between 1 and %d strings", MaxBatchSize)}
	}
	if len(inputs) > MaxValues/dimensions {
		return Result{}, &ValidationError{Param: "input", Message: fmt.Sprintf("input count times dimensions must not exceed %d", MaxValues)}
	}
	tokens := 0
	for i, input := range inputs {
		if input == "" || len(input) > MaxInputBytes {
			return Result{}, &ValidationError{Param: "input", Message: fmt.Sprintf("input[%d] must contain between 1 and %d UTF-8 bytes", i, MaxInputBytes)}
		}
		tokens += (len(input) + 3) / 4
	}
	result := Result{Vectors: make([][]float32, len(inputs)), PromptTokens: tokens}
	for i, input := range inputs {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		result.Vectors[i] = m.vector(input, dimensions)
	}
	return result, nil
}

func (m *MockModel) vector(input string, dimensions int) []float32 {
	hash := sha256.New()
	_, _ = hash.Write([]byte("mock-ai-go/embeddings/v1\x00"))
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(m.id)))
	_, _ = hash.Write(size[:])
	_, _ = hash.Write([]byte(m.id))
	_, _ = hash.Write([]byte(input))
	var block [sha256.Size + 8]byte
	hash.Sum(block[:0])
	vector := make([]float32, dimensions)
	squaredNorm := 0.0
	for start := 0; start < dimensions; start += 8 {
		binary.LittleEndian.PutUint64(block[sha256.Size:], uint64(start/8))
		digest := sha256.Sum256(block[:])
		for j := 0; j < 8 && start+j < dimensions; j++ {
			bits := binary.LittleEndian.Uint32(digest[j*4:]) >> 8
			value := float32((float64(bits)+0.5)/8388608 - 1)
			vector[start+j] = value
			squaredNorm += float64(value) * float64(value)
		}
	}
	norm := math.Sqrt(squaredNorm)
	for i, value := range vector {
		vector[i] = float32(float64(value) / norm)
	}
	return vector
}
