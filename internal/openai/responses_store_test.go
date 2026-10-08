package openai

import (
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

func TestMemoryResponseStoreRejectsInvalidOptions(t *testing.T) {
	t.Parallel()

	now := time.Now
	tests := []ResponseStoreOptions{
		{TTL: 0, MaxEntries: 1, MaxBytes: 1},
		{TTL: time.Hour, MaxEntries: 0, MaxBytes: 1},
		{TTL: time.Hour, MaxEntries: 1, MaxBytes: 0},
	}
	for _, options := range tests {
		if _, err := newMemoryResponseStore(options, now); err == nil {
			t.Fatalf("newMemoryResponseStore(%#v) succeeded", options)
		}
	}
	if _, err := newMemoryResponseStore(ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 1, MaxBytes: 1,
	}, nil); err == nil {
		t.Fatal("newMemoryResponseStore() accepted a nil clock")
	}
}

func TestMemoryResponseStoreSaveLoadDeleteAndSnapshotLifetime(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_700_000_000, 0)
	store := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 1, MaxBytes: 100,
	}, func() time.Time { return now })
	first := newTestResponseState("first", now, time.Hour, 40)
	if err := store.Put(first); err != nil {
		t.Fatalf("Put(first) error = %v", err)
	}
	loaded, found, err := store.Get(first.ID)
	if err != nil || !found || loaded != first {
		t.Fatalf("Get(first) = (%p, %t, %v), want (%p, true, nil)", loaded, found, err, first)
	}

	second := newTestResponseState("second", now.Add(time.Second), time.Hour, 40)
	if err := store.Put(second); err != nil {
		t.Fatalf("Put(second) error = %v", err)
	}
	if _, found, err := store.Get(first.ID); err != nil || found {
		t.Fatalf("Get(evicted first) = (_, %t, %v), want (_, false, nil)", found, err)
	}
	if loaded.ID != first.ID {
		t.Fatalf("previously loaded snapshot changed after eviction: %#v", loaded)
	}
	if err := store.Delete(second.ID); err != nil {
		t.Fatalf("Delete(second) error = %v", err)
	}
	if _, found, err := store.Get(second.ID); err != nil || found {
		t.Fatalf("Get(deleted second) = (_, %t, %v), want (_, false, nil)", found, err)
	}
}

func TestMemoryResponseStoreUsesAbsoluteTTLWithoutRefreshing(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_700_000_000, 0)
	store := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Minute, MaxEntries: 4, MaxBytes: 1_000,
	}, func() time.Time { return now })
	state := newTestResponseState("expires", now, time.Minute, 40)
	if err := store.Put(state); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	now = now.Add(59 * time.Second)
	if _, found, err := store.Get(state.ID); err != nil || !found {
		t.Fatalf("Get(before expiry) = (_, %t, %v), want (_, true, nil)", found, err)
	}
	now = now.Add(time.Second)
	if _, found, err := store.Get(state.ID); err != nil || found {
		t.Fatalf("Get(at expiry) = (_, %t, %v), want (_, false, nil)", found, err)
	}
}

func TestMemoryResponseStoreEvictsFIFOByEntryAndByteLimits(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_700_000_000, 0)
	entryStore := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 2, MaxBytes: 1_000,
	}, func() time.Time { return now })
	for _, id := range []string{"one", "two"} {
		if err := entryStore.Put(newTestResponseState(id, now, time.Hour, 40)); err != nil {
			t.Fatalf("Put(%q) error = %v", id, err)
		}
	}
	if _, _, err := entryStore.Get("one"); err != nil {
		t.Fatalf("Get(one) error = %v", err)
	}
	if err := entryStore.Put(newTestResponseState("three", now, time.Hour, 40)); err != nil {
		t.Fatalf("Put(three) error = %v", err)
	}
	assertStorePresence(t, entryStore, map[string]bool{"one": false, "two": true, "three": true})

	byteStore := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 10, MaxBytes: 100,
	}, func() time.Time { return now })
	if err := byteStore.Put(newTestResponseState("one", now, time.Hour, 60)); err != nil {
		t.Fatalf("Put(one) error = %v", err)
	}
	if err := byteStore.Put(newTestResponseState("two", now, time.Hour, 60)); err != nil {
		t.Fatalf("Put(two) error = %v", err)
	}
	assertStorePresence(t, byteStore, map[string]bool{"one": false, "two": true})
}

func TestMemoryResponseStoreRejectsOversizedRecord(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_700_000_000, 0)
	store := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 10, MaxBytes: 100,
	}, func() time.Time { return now })
	err := store.Put(newTestResponseState("oversized", now, time.Hour, 101))
	if !errors.Is(err, errResponseStateTooLarge) {
		t.Fatalf("Put(oversized) error = %v, want %v", err, errResponseStateTooLarge)
	}
	if len(store.records) != 0 || store.totalBytes != 0 {
		t.Fatalf("store retained oversized record: entries=%d bytes=%d", len(store.records), store.totalBytes)
	}
}

func TestMemoryResponseStorePurgesExpiredEntriesOnPut(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_700_000_000, 0)
	store := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Minute, MaxEntries: 2, MaxBytes: 100,
	}, func() time.Time { return now })
	if err := store.Put(newTestResponseState("expired", now, time.Minute, 60)); err != nil {
		t.Fatalf("Put(expired) error = %v", err)
	}
	now = now.Add(time.Minute)
	if err := store.Put(newTestResponseState("current", now, time.Minute, 60)); err != nil {
		t.Fatalf("Put(current) error = %v", err)
	}
	assertStorePresence(t, store, map[string]bool{"expired": false, "current": true})
}

func TestMemoryResponseStoreSupportsConcurrentAccess(t *testing.T) {
	t.Parallel()

	now := time.Unix(1_700_000_000, 0)
	store := newTestMemoryResponseStore(t, ResponseStoreOptions{
		TTL: time.Hour, MaxEntries: 128, MaxBytes: 1 << 20,
	}, func() time.Time { return now })

	var wait sync.WaitGroup
	errorsChannel := make(chan error, 128)
	for index := 0; index < 128; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			id := fmt.Sprintf("response-%d", index)
			if err := store.Put(newTestResponseState(id, now, time.Hour, 64)); err != nil {
				errorsChannel <- err
				return
			}
			state, found, err := store.Get(id)
			if err != nil {
				errorsChannel <- err
				return
			}
			if !found || state.ID != id {
				errorsChannel <- fmt.Errorf("Get(%q) = (%#v, %t)", id, state, found)
			}
		}()
	}
	wait.Wait()
	close(errorsChannel)
	for err := range errorsChannel {
		t.Errorf("concurrent access error: %v", err)
	}
}

func newTestMemoryResponseStore(
	t testing.TB,
	options ResponseStoreOptions,
	now func() time.Time,
) *memoryResponseStore {
	t.Helper()
	store, err := newMemoryResponseStore(options, now)
	if err != nil {
		t.Fatalf("newMemoryResponseStore() error = %v", err)
	}
	return store
}

func newTestResponseState(id string, completedAt time.Time, ttl time.Duration, logicalBytes int64) *responseState {
	return &responseState{
		ID:                   id,
		Model:                "mock-chat-model",
		EmittedFunctionCalls: map[string]struct{}{},
		FinalContext:         chat.NewContextState(),
		ExpiresAt:            completedAt.Add(ttl),
		LogicalBytes:         logicalBytes,
	}
}

func assertStorePresence(t testing.TB, store *memoryResponseStore, want map[string]bool) {
	t.Helper()
	for id, wantFound := range want {
		_, found, err := store.Get(id)
		if err != nil || found != wantFound {
			t.Errorf("Get(%q) = (_, %t, %v), want (_, %t, nil)", id, found, err, wantFound)
		}
	}
}
