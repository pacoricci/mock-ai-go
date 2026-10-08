// Package model defines metadata shared by AI model capabilities.
package model

// Kind identifies a model capability.
type Kind string

const (
	// KindChat identifies a model that can generate chat completions.
	KindChat Kind = "chat"
)

// Info contains the minimal metadata needed to identify a model.
type Info struct {
	ID   string
	Kind Kind
}
