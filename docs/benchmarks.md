# Benchmarks

## HTTP/SSE load

`ssebench` tests Chat Completions and Responses, validates SSE, and records JSON
results. Build with `make build build-bench`, then start the server:

```sh
PPROF_ADDR=127.0.0.1:6060 SSE_DELAY=100ms SSE_TOKENS_PER_CHUNK=1 \
  MAX_COMPLETION_TOKENS=4096 ./bin/mock-ai-go
```

In another terminal:

```sh
mkdir -p bench/results
./bin/ssebench -scenario bench/scenarios/smoke.json -output bench/results/smoke.json
```

[Scenarios](../bench/scenarios) cover both protocols, concurrency, pacing, and
cancellation. `server_env` records settings; the CLI does not apply them or start
the server. Separate processes keep generator activity out of server metrics.
Use `ssebench -h` for flags; an empty `metrics_url` disables diagnostics.

## Output

The stderr summary shows stream counts, errors, throughput, and opening-latency/
content-gap percentiles (`n/a` without samples). JSON includes scenario,
environment, error samples, and optional metrics. Omit `-output` or use `-` to
write JSON to stdout.

Exit codes: 0 for no stream/protocol errors, 1 for execution/output failures or
invalid results, 2 for CLI usage errors. Planned cancellations are reported
separately and do not alone fail a run.

## Timing and metrics

Default pacing is 800 ms initially, then 100 ms ± 20 ms between chunks.
Unspecified settings retain defaults, even with a scenario's partial
`server_env`. Disable all three delays explicitly for unpaced tests; see
[configuration](configuration.md#streaming). JSON responses have no added latency.

At four words per chunk, expected duration is approximately
`800 ms + (ceil(output_words / 4) - 1) × 100 ms`, before jitter and backpressure:
about 1.5 s for 32 words or 5.7 s for 200. The default 128–512-word text range
takes approximately 3.9–13.5 s (mean 8.7 s). Request limits cap this range;
disable text-length sampling when a benchmark needs exact output sizes.
Words are not tokenizer tokens.
This profile does not model prompt processing, reasoning, queuing, or bursts.

Elapsed time starts after the initial diagnostics sample and ends when workers
return, before final metrics collection. Throughput includes ramp-up and streaming.
Diagnostics sample Go runtime metrics before, during, and after load; they exclude
OS RSS and total CPU. Memory per stream is a baseline-to-peak estimate.

## Reference capacity measurement

One local campaign on October 8, 2026 measured streaming capacity with the
default timing profile: 800 ms initial delay, four words per chunk, and 100 ms
± 20 ms gaps. Server and `ssebench` ran as separate processes on one Apple M4
(10 cores, 16 GiB, macOS 26.4, Go 1.27.1, low-power mode on) over loopback
HTTP/1.1, with a fresh server per run. The build was 6822cf0 plus uncommitted
changes, so treat these numbers as indicative rather than reproducible.

| Run | Result | Server RSS |
| --- | --- | --- |
| 8,000 long Chat streams, 5 s ramp then 60 s | No errors; open p95 800 ms, gap p99 120 ms | 301 MiB |
| 12,000 long Chat streams, 5 s ramp then 60 s | No errors; open p95 801 ms, gap p99 120 ms | 448 MiB |
| 16,000 complete 200-word Chat streams | All completed in 7.9 s | 584 MiB |
| 16,000 complete 200-word Responses streams, `store: false` | All completed in 7.9 s | 1,018 MiB |
| 16,000 long Chat streams, 55 s | Server exited after about 27 s: `failed to create new OS thread` | n/a |

12,000 streams was the highest sustained level that passed, not an absolute
limit. At 20,000 streams both the client (ephemeral ports) and the server (OS
threads) failed, so that run does not isolate the service. Runs lasted at most
one minute and did not measure a sustained request rate, TLS, or proxies.

## Microbenchmarks

`make bench` measures in-process generation/encoding and allocations. To repeat:

```sh
go test ./internal/chat ./internal/openai -run '^$' -bench . -benchmem -count 5
```

Record revision, toolchain, configuration, and machine when comparing results.
