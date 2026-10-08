// Command mock-ai-go is the entry point for the application.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/pacoricci/mock-ai-go/internal/app"
	"github.com/pacoricci/mock-ai-go/internal/config"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run keeps deferred cleanup inside a function that returns before os.Exit.
func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	cfg, err := config.FromEnv()
	if err != nil {
		return err
	}
	return app.Run(ctx, os.Stdout, cfg)
}
