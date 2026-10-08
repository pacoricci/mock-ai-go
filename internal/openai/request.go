package openai

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

const maxRequestBytes = 1 << 20

func decodeJSONRequest[T any](w http.ResponseWriter, r *http.Request) (T, bool) {
	body := http.MaxBytesReader(w, r.Body, maxRequestBytes)
	decoder := json.NewDecoder(body)

	var request T
	if err := decoder.Decode(&request); err != nil {
		message := "Invalid JSON request body"
		var maxBytesError *http.MaxBytesError
		if errors.As(err, &maxBytesError) {
			message = "Request body is too large"
		}
		writeError(w, http.StatusBadRequest, message, "invalid_request_error", nil, nil)
		return request, false
	}

	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		writeError(w, http.StatusBadRequest, "Request body must contain a single JSON object", "invalid_request_error", nil, nil)
		return request, false
	}

	return request, true
}
