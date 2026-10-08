package chat

import "testing"

func TestXorshift64StarIsDeterministicAndNonZero(t *testing.T) {
	t.Parallel()

	first := newXorshift64Star(0)
	second := newXorshift64Star(0)
	for range 32 {
		left, right := first.next(), second.next()
		if left == 0 || left != right {
			t.Fatalf("xorshift outputs = %d and %d", left, right)
		}
	}
}
