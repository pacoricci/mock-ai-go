.PHONY: build build-bench bench fmt run test test-openai-sdk test-anthropic-sdk vet lint check openresponses-compliance docker-build docker-run docker-up clean

OPENRESPONSES_BASE_URL ?= http://127.0.0.1:8080/v1
OPENRESPONSES_API_KEY ?= mock-api-key
OPENRESPONSES_MODEL ?= mock-chat-model
OPENRESPONSES_FILTER ?= basic-response,assistant-phase,response-output-phase-schema,streaming-response,system-prompt,tool-calling,image-input,multi-turn

build:
	go build -o bin/mock-ai-go ./cmd/mock-ai-go

build-bench:
	go build -o bin/ssebench ./cmd/ssebench

bench:
	go test ./internal/chat ./internal/openai -run '^$$' -bench . -benchmem

fmt:
	gofmt -w $$(find . -name '*.go' -not -path './vendor/*')

run:
	go run ./cmd/mock-ai-go

test: test-openai-sdk test-anthropic-sdk
	go test ./...

test-openai-sdk:
	cd tests/openai-sdk && go test ./...

test-anthropic-sdk:
	cd tests/anthropic-sdk && go test ./...

vet:
	go vet ./...
	cd tests/openai-sdk && go vet ./...
	cd tests/anthropic-sdk && go vet ./...

lint:
	golangci-lint run ./...
	cd tests/openai-sdk && golangci-lint run ./...
	cd tests/anthropic-sdk && golangci-lint run ./...

check: test vet

openresponses-compliance:
	cd third_party/openresponses-compliance && bun run test:compliance --base-url "$(OPENRESPONSES_BASE_URL)" --api-key "$(OPENRESPONSES_API_KEY)" --model "$(OPENRESPONSES_MODEL)" --filter "$(OPENRESPONSES_FILTER)"

docker-build:
	docker build --tag mock-ai-go:local .

docker-run: docker-build
	docker run --rm --publish 127.0.0.1:8080:8080 mock-ai-go:local

docker-up:
	docker compose up --build

clean:
	go clean
	rm -rf bin coverage.out
