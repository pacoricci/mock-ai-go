package openai

import (
	"container/list"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/pacoricci/mock-ai-go/internal/chat"
)

var errResponseStateTooLarge = errors.New("response state exceeds the store byte limit")

// ResponseStoreOptions configures process-local Response continuation state.
type ResponseStoreOptions struct {
	// TTL is the absolute lifetime of a terminal Response snapshot.
	TTL time.Duration
	// MaxEntries is the maximum number of terminal Response snapshots.
	MaxEntries int
	// MaxBytes is the maximum logical size of all Response snapshots.
	MaxBytes int64
}

// responseState holds only what a continuation reads: the model check, the
// aggregate context, and the call IDs that later function outputs may answer.
// Input and output content are not retained.
type responseState struct {
	ID                   string
	Model                string
	EmittedFunctionCalls map[string]struct{}
	FinalContext         chat.ContextState
	ExpiresAt            time.Time
	LogicalBytes         int64
}

// responseStore takes ownership of immutable responseState values passed to Put.
// Values returned by Get are read-only and remain valid after later eviction.
type responseStore interface {
	Get(id string) (*responseState, bool, error)
	Put(state *responseState) error
	Delete(id string) error
}

type memoryResponseStore struct {
	mu         sync.Mutex
	records    map[string]*memoryResponseStoreEntry
	order      *list.List
	totalBytes int64
	options    ResponseStoreOptions
	now        func() time.Time
}

type memoryResponseStoreEntry struct {
	state   *responseState
	element *list.Element
}

func newMemoryResponseStore(options ResponseStoreOptions, now func() time.Time) (*memoryResponseStore, error) {
	if options.TTL <= 0 {
		return nil, errors.New("response store TTL must be positive")
	}
	if options.MaxEntries <= 0 {
		return nil, errors.New("response store maximum entries must be positive")
	}
	if options.MaxBytes <= 0 {
		return nil, errors.New("response store maximum bytes must be positive")
	}
	if now == nil {
		return nil, errors.New("response store clock is nil")
	}
	return &memoryResponseStore{
		records: make(map[string]*memoryResponseStoreEntry),
		order:   list.New(),
		options: options,
		now:     now,
	}, nil
}

func (s *memoryResponseStore) Get(id string) (*responseState, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	entry, found := s.records[id]
	if !found {
		return nil, false, nil
	}
	if !s.now().Before(entry.state.ExpiresAt) {
		s.remove(entry)
		return nil, false, nil
	}
	return entry.state, true, nil
}

func (s *memoryResponseStore) Put(state *responseState) error {
	if state == nil {
		return errors.New("response state is nil")
	}
	if state.LogicalBytes <= 0 {
		return errors.New("response state logical size must be positive")
	}
	if state.LogicalBytes > s.options.MaxBytes {
		return fmt.Errorf("%w: record is %d bytes, limit is %d", errResponseStateTooLarge, state.LogicalBytes, s.options.MaxBytes)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.purgeExpired(s.now())
	if existing, found := s.records[state.ID]; found {
		s.remove(existing)
	}
	for len(s.records) >= s.options.MaxEntries || s.totalBytes+state.LogicalBytes > s.options.MaxBytes {
		oldest := s.order.Front()
		if oldest == nil {
			break
		}
		s.remove(s.records[oldest.Value.(string)])
	}
	element := s.order.PushBack(state.ID)
	s.records[state.ID] = &memoryResponseStoreEntry{state: state, element: element}
	s.totalBytes += state.LogicalBytes
	return nil
}

func (s *memoryResponseStore) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry, found := s.records[id]; found {
		s.remove(entry)
	}
	return nil
}

func (s *memoryResponseStore) purgeExpired(now time.Time) {
	for element := s.order.Front(); element != nil; element = s.order.Front() {
		entry := s.records[element.Value.(string)]
		if now.Before(entry.state.ExpiresAt) {
			return
		}
		s.remove(entry)
	}
}

func (s *memoryResponseStore) remove(entry *memoryResponseStoreEntry) {
	delete(s.records, entry.state.ID)
	s.order.Remove(entry.element)
	s.totalBytes -= entry.state.LogicalBytes
}
