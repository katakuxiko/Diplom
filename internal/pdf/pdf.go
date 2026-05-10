package pdf

import (
	"regexp"
	"strings"
	"unicode"

	"code.sajari.com/docconv"
)

func ExtractText(path string) (string, error) {
	res, err := docconv.ConvertPath(path)
	
	if err != nil {
		return "", err
	}
	return res.Body, nil
}

func Sanitize(s string) string {
	s = strings.ReplaceAll(s, "\r", "\n")
	s = strings.ReplaceAll(s, "\t", " ")

	// Убираем множественные пробелы, но сохраняем переносы строк
	lines := strings.Split(s, "\n")
	for i, line := range lines {
		lines[i] = strings.Join(strings.Fields(line), " ")
	}

	return strings.Join(lines, "\n")
}

// ChunkBySentences разбивает текст на чанки по предложениям с учетом overlap
func ChunkBySentences(text string, maxWords, overlap int) []string {
	if maxWords <= 0 {
		maxWords = 200
	}
	if overlap < 0 {
		overlap = 0
	}

	// Разбиваем на предложения
	sentences := splitIntoSentences(text)

	var chunks []string
	var currentChunk []string
	var wordCount int

	for i, sentence := range sentences {
		sentWords := len(strings.Fields(sentence))

		// Если добавление предложения превысит лимит и у нас уже есть контент
		if wordCount+sentWords > maxWords && len(currentChunk) > 0 {
			// Сохраняем текущий чанк
			chunks = append(chunks, strings.Join(currentChunk, " "))

			// Определяем overlap: берем последние предложения
			overlapSentences := calculateOverlapSentences(currentChunk, overlap)
			currentChunk = overlapSentences
			wordCount = countWords(currentChunk)
		}

		currentChunk = append(currentChunk, sentence)
		wordCount += sentWords

		// Если это последнее предложение
		if i == len(sentences)-1 && len(currentChunk) > 0 {
			chunks = append(chunks, strings.Join(currentChunk, " "))
		}
	}

	return chunks
}

// ChunkByWords оставляем для обратной совместимости
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

// splitIntoSentences разбивает текст на предложения
func splitIntoSentences(text string) []string {
	// Регулярное выражение для разбиения на предложения
	// Учитывает . ! ? с последующим пробелом и заглавной буквой
	re := regexp.MustCompile(`[.!?]+[\s]+`)

	parts := re.Split(text, -1)
	var sentences []string

	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if len(trimmed) > 0 {
			// Добавляем точку в конец, если её нет
			if !strings.HasSuffix(trimmed, ".") &&
				!strings.HasSuffix(trimmed, "!") &&
				!strings.HasSuffix(trimmed, "?") {
				trimmed += "."
			}
			sentences = append(sentences, trimmed)
		}
	}

	return sentences
}

// calculateOverlapSentences возвращает последние предложения для overlap
func calculateOverlapSentences(sentences []string, overlapWords int) []string {
	if overlapWords == 0 || len(sentences) == 0 {
		return []string{}
	}

	var result []string
	var words int

	// Идем с конца
	for i := len(sentences) - 1; i >= 0; i-- {
		sentWords := len(strings.Fields(sentences[i]))
		if words+sentWords > overlapWords {
			break
		}
		result = append([]string{sentences[i]}, result...)
		words += sentWords
	}

	return result
}

// countWords подсчитывает количество слов в списке предложений
func countWords(sentences []string) int {
	count := 0
	for _, s := range sentences {
		count += len(strings.Fields(s))
	}
	return count
}

// isCapitalized проверяет, начинается ли строка с заглавной буквы
func isCapitalized(s string) bool {
	s = strings.TrimSpace(s)
	if len(s) == 0 {
		return false
	}
	firstRune := []rune(s)[0]
	return unicode.IsUpper(firstRune)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
