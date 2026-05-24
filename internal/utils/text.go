package utils

import (
	"regexp"
	"strings"
)

var (
	reHyphenNewline   = regexp.MustCompile("-\n+")
	reMultipleNewline = regexp.MustCompile("\n{3,}")
	reWhitespace      = regexp.MustCompile(`\s+`)
)

// NormalizeText cleans typical PDF artifacts: hyphenation, extra newlines, spaces, NBSP, soft hyphens.
func NormalizeText(s string) string {
	if s == "" {
		return s
	}
	// Remove soft hyphen and normalize NBSP
	s = strings.ReplaceAll(s, "\u00ad", "")  // soft hyphen
	s = strings.ReplaceAll(s, "\u00A0", " ") // NBSP -> space

	// Join hyphenated line breaks: "учебно-\nвоспитательной" -> "учебновоспитательной"
	s = reHyphenNewline.ReplaceAllString(s, "")

	// Collapse >2 newlines to 2 to preserve paragraphs
	s = reMultipleNewline.ReplaceAllString(s, "\n\n")

	// Replace remaining newlines/tabs with single spaces and collapse spaces
	s = reWhitespace.ReplaceAllString(s, " ")

	return strings.TrimSpace(s)
}

// TruncateByChars ensures the string length does not exceed max characters.
func TruncateByChars(s string, max int) string {
	if max <= 0 {
		return ""
	}
	r := []rune(s)
	if len(r) <= max {
		return s
	}
	return string(r[:max])
}
