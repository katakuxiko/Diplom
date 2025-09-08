package util

import (
	"fmt"
	"time"
	"unicode/utf8"
)

// Timestamped генерирует имя файла с меткой времени
func Timestamped(name string) string {
	ts := time.Now().Format("20060102_150405")
	return fmt.Sprintf("%s__%s", ts, name)
}

// TruncateRunes — безопасное усечение по рунам
func TruncateRunes(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	rs := []rune(s)
	return string(rs[:n])
}
