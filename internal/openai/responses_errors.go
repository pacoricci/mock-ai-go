package openai

import (
	"errors"
	"net/http"
)

type responseRequestError struct {
	message string
	param   string
	code    string
}

func newResponseRequestError(message, param, code string) *responseRequestError {
	return &responseRequestError{message: message, param: param, code: code}
}

func writeResponseRequestError(w http.ResponseWriter, requestError *responseRequestError) {
	writeError(
		w,
		http.StatusBadRequest,
		requestError.message,
		"invalid_request_error",
		optionalStringPointer(requestError.param),
		optionalStringPointer(requestError.code),
	)
}

func optionalStringPointer(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func writeResponseModelError(w http.ResponseWriter, err error) {
	writeModelError(w, err, "max_output_tokens", "The chat model failed to create a response")
}

func writeResponseStoreError(w http.ResponseWriter, err error) {
	code, message := responseStoreErrorDetails(err)
	writeError(w, http.StatusInternalServerError, message, "server_error", nil, &code)
}

func responseStoreErrorDetails(err error) (string, string) {
	if errors.Is(err, errResponseStateTooLarge) {
		return "response_store_capacity_exceeded", "The completed response exceeds the configured response store capacity"
	}
	return "server_error", "The response state could not be stored"
}
