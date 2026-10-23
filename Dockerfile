# syntax=docker/dockerfile:1

FROM golang:1.26.10-alpine3.23@sha256:0525c952defebc693cdaf62cdf43afa32be5a7f377e99260642ea19c240c708e AS builder

WORKDIR /src

COPY go.mod ./
RUN go mod download

COPY . ./
# The production image builds only the service binary.
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/mock-ai-go ./cmd/mock-ai-go

FROM alpine:3.23.7@sha256:ff7f0c727dd796aa36787bfcd9a64bf35a6ce72a73451b3359fde849b6703d63

LABEL org.opencontainers.image.licenses="MIT"

RUN addgroup -S mockai && adduser -S -G mockai -H -h /nonexistent mockai

COPY --from=builder --chown=mockai:mockai /out/mock-ai-go /usr/local/bin/mock-ai-go
COPY LICENSE /usr/share/licenses/mock-ai-go/LICENSE

USER mockai

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q -O /dev/null http://127.0.0.1:8080/openapi.yaml || exit 1

ENTRYPOINT ["/usr/local/bin/mock-ai-go"]
