package pdf

import (
	"os/exec"
	"strings"
)

func ExtractText(path string) (string, error) {
	cmd := exec.Command("pdftotext", "-enc", "UTF-8", path, "-")
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func Sanitize(s string) string {
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\t", " ")
	s = strings.Join(strings.Fields(s), " ")
	return s
}

func ChunkByWords(text string, size, overlap int) []string {
	words := strings.Fields(text)
	if size <= 0 {
		size = 200
	}
	if overlap < 0 {
		overlap = 0
	}
	var out []string
	for i := 0; i < len(words); i += max(1, size-overlap) {
		end := i + size
		if end > len(words) {
			end = len(words)
		}
		out = append(out, strings.Join(words[i:end], " "))
		if end == len(words) {
			break
		}
	}
	return out
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
