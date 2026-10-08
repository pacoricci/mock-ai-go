// Command ssebench runs black-box SSE load scenarios.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"

	"github.com/pacoricci/mock-ai-go/internal/loadgen"
)

func main() {
	os.Exit(run())
}

func run() int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	return runCommand(ctx, os.Args[1:], os.Stdout, os.Stderr)
}

func runCommand(ctx context.Context, args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("ssebench", flag.ContinueOnError)
	flags.SetOutput(stderr)
	scenarioPath := flags.String("scenario", "", "path to a benchmark scenario JSON file")
	outputPath := flags.String("output", "-", "result JSON path, or - for stdout; summary goes to stderr")
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}

	if *scenarioPath == "" {
		_, _ = fmt.Fprintln(stderr, "-scenario is required")
		return 2
	}

	scenarioFile, err := os.Open(*scenarioPath)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "open scenario: %v\n", err)
		return 1
	}
	var scenario loadgen.Scenario
	err = json.NewDecoder(scenarioFile).Decode(&scenario)
	_ = scenarioFile.Close()
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "decode scenario: %v\n", err)
		return 1
	}

	result, err := loadgen.Run(ctx, scenario)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "run scenario: %v\n", err)
		return 1
	}

	output, closeOutput, err := resultWriter(*outputPath, stdout)
	if err != nil {
		_, _ = fmt.Fprintf(stderr, "open output: %v\n", err)
		return 1
	}
	encoder := json.NewEncoder(output)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		if closeErr := closeOutput(); closeErr != nil {
			_, _ = fmt.Fprintf(stderr, "close output after write failure: %v\n", closeErr)
		}
		_, _ = fmt.Fprintf(stderr, "write result: %v\n", err)
		return 1
	}
	if err := closeOutput(); err != nil {
		_, _ = fmt.Fprintf(stderr, "close output: %v\n", err)
		return 1
	}
	if err := writeSummary(stderr, result); err != nil {
		_, _ = fmt.Fprintf(stderr, "write summary: %v\n", err)
		return 1
	}

	if result.Errors > 0 || result.InvalidEvents > 0 || result.InvalidUsage > 0 {
		return 1
	}
	return 0
}

func resultWriter(path string, stdout io.Writer) (io.Writer, func() error, error) {
	if path == "-" {
		return stdout, func() error { return nil }, nil
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, nil, err
	}
	return file, file.Close, nil
}
