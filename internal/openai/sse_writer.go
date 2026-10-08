package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"unicode/utf8"
)

const initialSSEBufferCapacity = 512

type sseWriter struct {
	w          io.Writer
	controller *http.ResponseController
	buffer     []byte
}

func newSSEWriter(w io.Writer, controller *http.ResponseController) *sseWriter {
	return &sseWriter{
		w:          w,
		controller: controller,
		buffer:     make([]byte, 0, initialSSEBufferCapacity),
	}
}

func (w *sseWriter) writeJSONEvent(eventType string, value any) error {
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}

	w.buffer = append(w.buffer[:0], "event: "...)
	w.buffer = append(w.buffer, eventType...)
	w.buffer = append(w.buffer, '\n')
	w.buffer = append(w.buffer, "data: "...)
	w.buffer = append(w.buffer, encoded...)
	w.buffer = append(w.buffer, '\n', '\n')
	return w.writeAndFlush()
}

func (w *sseWriter) writeDone() error {
	w.buffer = append(w.buffer[:0], "data: [DONE]\n\n"...)
	return w.writeAndFlush()
}

func (w *sseWriter) writeAndFlush() error {
	written, err := w.w.Write(w.buffer)
	if err != nil {
		return err
	}
	if written != len(w.buffer) {
		return io.ErrShortWrite
	}
	return w.controller.Flush()
}

func appendJSONStringBytes(destination, source []byte) []byte {
	destination = append(destination, '"')
	start := 0
	for index := 0; index < len(source); {
		if source[index] < utf8.RuneSelf {
			if source[index] >= 0x20 && source[index] != '\\' && source[index] != '"' {
				index++
				continue
			}

			destination = append(destination, source[start:index]...)
			switch source[index] {
			case '\\', '"':
				destination = append(destination, '\\', source[index])
			case '\n':
				destination = append(destination, '\\', 'n')
			case '\r':
				destination = append(destination, '\\', 'r')
			case '\t':
				destination = append(destination, '\\', 't')
			case '\b':
				destination = append(destination, '\\', 'b')
			case '\f':
				destination = append(destination, '\\', 'f')
			default:
				destination = append(destination, `\u00`...)
				destination = append(destination, hexDigit(source[index]>>4), hexDigit(source[index]&0x0f))
			}
			index++
			start = index
			continue
		}

		runeValue, size := utf8.DecodeRune(source[index:])
		if runeValue == utf8.RuneError && size == 1 {
			destination = append(destination, source[start:index]...)
			destination = append(destination, `\ufffd`...)
			index++
			start = index
			continue
		}
		if runeValue == '\u2028' || runeValue == '\u2029' {
			destination = append(destination, source[start:index]...)
			destination = append(destination, `\u202`...)
			destination = append(destination, hexDigit(byte(runeValue&0x0f)))
			index += size
			start = index
			continue
		}
		index += size
	}
	destination = append(destination, source[start:]...)
	destination = append(destination, '"')
	return destination
}

func hexDigit(value byte) byte {
	const digits = "0123456789abcdef"
	return digits[value]
}
