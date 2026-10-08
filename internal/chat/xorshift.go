package chat

const (
	xorshiftMultiplier uint64 = 2_685_821_657_736_338_717
	nonZeroSeed        uint64 = 0x9e3779b97f4a7c15
)

// xorshift64Star is a small, non-cryptographic pseudo-random generator.
// Its state must never be zero.
type xorshift64Star struct {
	state uint64
}

func newXorshift64Star(seed uint64) xorshift64Star {
	seed = mixSeed(seed)
	if seed == 0 {
		seed = nonZeroSeed
	}
	return xorshift64Star{state: seed}
}

func (r *xorshift64Star) next() uint64 {
	x := r.state
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	r.state = x
	return x * xorshiftMultiplier
}

// mixSeed avalanches similar request hashes before they enter xorshift64*.
func mixSeed(value uint64) uint64 {
	value ^= value >> 30
	value *= 0xbf58476d1ce4e5b9
	value ^= value >> 27
	value *= 0x94d049bb133111eb
	value ^= value >> 31
	return value
}
