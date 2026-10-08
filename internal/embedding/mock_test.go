package embedding_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/pacoricci/mock-ai-go/internal/embedding"
)

func newModel(t *testing.T, id string, dimensions int) *embedding.MockModel {
	t.Helper()
	model, err := embedding.NewMockModel(id, dimensions)
	if err != nil {
		t.Fatal(err)
	}
	return model
}

func TestDeterminismBatchIndependenceAndOwnership(t *testing.T) {
	t.Parallel()
	model := newModel(t, "model", 32)
	inputs := []string{"Hello café 世界", "another document", "Hello café 世界"}
	batch, err := model.Embed(t.Context(), inputs, 32)
	if err != nil {
		t.Fatal(err)
	}
	restarted := newModel(t, "model", 32)
	for i, input := range inputs {
		single, err := restarted.Embed(t.Context(), []string{input}, 32)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(single.Vectors[0], batch.Vectors[i]) {
			t.Fatal("embedding changed with batch position or model reconstruction")
		}
	}
	if reflect.DeepEqual(batch.Vectors[0], batch.Vectors[1]) {
		t.Fatal("distinct inputs have identical vectors")
	}
	other, err := newModel(t, "other-model", 32).Embed(t.Context(), inputs, 32)
	if err != nil {
		t.Fatal(err)
	}
	if reflect.DeepEqual(other.Vectors[0], batch.Vectors[0]) {
		t.Fatal("model ID did not affect vector")
	}
	batch.Vectors[0][0] = 123
	if batch.Vectors[2][0] == 123 {
		t.Fatal("duplicate inputs share mutable vectors")
	}
	var group sync.WaitGroup
	for range 16 {
		group.Go(func() {
			result, err := model.Embed(t.Context(), []string{inputs[2], inputs[1]}, 32)
			if err != nil {
				t.Error(err)
				return
			}
			if !reflect.DeepEqual(result.Vectors[0], batch.Vectors[2]) {
				t.Error("concurrent embedding changed")
			}
		})
	}
	group.Wait()
}

func TestDimensionsNormalizationAndUsage(t *testing.T) {
	t.Parallel()
	model := newModel(t, "", embedding.MaxDimensions)
	if model.ID() != embedding.DefaultModelID {
		t.Fatal(model.ID())
	}
	for _, dimensions := range []int{1, 3, 256, 1536, embedding.MaxDimensions} {
		result, err := model.Embed(t.Context(), []string{"café", "世界", " "}, dimensions)
		if err != nil {
			t.Fatal(err)
		}
		if result.PromptTokens != 5 {
			t.Fatalf("usage = %d, want ceil(5/4)+ceil(6/4)+ceil(1/4)", result.PromptTokens)
		}
		for _, vector := range result.Vectors {
			if len(vector) != dimensions {
				t.Fatal("wrong dimensions")
			}
			norm := 0.0
			for _, value := range vector {
				norm += float64(value) * float64(value)
			}
			if math.Abs(norm-1) > 1e-6 {
				t.Fatalf("squared norm = %g", norm)
			}
		}
	}
	defaultModel := newModel(t, "", 0)
	if defaultModel.Dimensions() != embedding.DefaultDimensions {
		t.Fatal(defaultModel.Dimensions())
	}
}

func TestValidationAndCancellation(t *testing.T) {
	t.Parallel()
	model := newModel(t, "model", embedding.MaxDimensions)
	for _, test := range []struct {
		name       string
		inputs     []string
		dimensions int
		param      string
	}{
		{"missing", nil, 16, "input"},
		{"empty", []string{""}, 16, "input"},
		{"too long", []string{strings.Repeat("é", embedding.MaxInputBytes/2+1)}, 16, "input"},
		{"batch limit", make([]string, embedding.MaxBatchSize+1), 1, "input"},
		{"output limit", make([]string, embedding.MaxValues/embedding.MaxDimensions+1), embedding.MaxDimensions, "input"},
		{"zero dimensions", []string{"ok"}, 0, "dimensions"},
		{"negative dimensions", []string{"ok"}, -1, "dimensions"},
		{"large dimensions", []string{"ok"}, embedding.MaxDimensions + 1, "dimensions"},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := model.Embed(t.Context(), test.inputs, test.dimensions)
			var validation *embedding.ValidationError
			if !errors.As(err, &validation) || validation.Param != test.param {
				t.Fatalf("error = %v", err)
			}
			if result.Vectors != nil {
				t.Fatal("invalid batch returned partial output")
			}
		})
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := model.Embed(ctx, []string{"ok"}, 16); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for _, dimensions := range []int{-1, embedding.MaxDimensions + 1} {
		if _, err := embedding.NewMockModel("model", dimensions); err == nil {
			t.Fatal("accepted invalid dimensions")
		}
	}
	if _, err := embedding.NewMockModel(" ", 16); err == nil {
		t.Fatal("accepted blank model")
	}
}

func TestLimitBoundaries(t *testing.T) {
	t.Parallel()
	model := newModel(t, "", embedding.MaxDimensions)
	for _, test := range []struct {
		count, dimensions int
		text              string
	}{
		{1, 1, strings.Repeat("x", embedding.MaxInputBytes)},
		{embedding.MaxBatchSize, 1, "x"},
		{embedding.MaxValues / embedding.MaxDimensions, embedding.MaxDimensions, "x"},
	} {
		inputs := make([]string, test.count)
		for i := range inputs {
			inputs[i] = test.text
		}
		if _, err := model.Embed(t.Context(), inputs, test.dimensions); err != nil {
			t.Fatal(err)
		}
	}
}

func TestEmbeddingGoldenVector(t *testing.T) {
	t.Parallel()
	result, err := newModel(t, "model", 3).Embed(t.Context(), []string{"Hello café 世界"}, 3)
	if err != nil {
		t.Fatal(err)
	}
	// The version-one fixture pins hashing, expansion, and float32 normalization.
	want := []uint32{0x3f0767d0, 0x3f26096f, 0x3f0c1e43}
	for i, bits := range want {
		if got := math.Float32bits(result.Vectors[0][i]); got != bits {
			t.Fatalf("component %d = %#x, want %#x", i, got, bits)
		}
	}
}
