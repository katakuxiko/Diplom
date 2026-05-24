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
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if maxWords <= 0 {
		maxWords = 200
	}
	if overlap < 0 {
		overlap = 0
	}
	if overlap >= maxWords {
		overlap = maxWords / 4
	}

	// Разбиваем на предложения
	sentences := splitIntoSentences(text)
	if len(sentences) == 0 {
		fallback := ChunkByWords(text, maxWords, overlap)
		return cleanupChunks(fallback, max(20, maxWords/5))
	}

	var chunks []string
	var currentChunk []string
	var wordCount int

	flushCurrent := func() {
		if len(currentChunk) == 0 {
			return
		}
		chunks = append(chunks, strings.Join(currentChunk, " "))
		overlapSentences := calculateOverlapSentences(currentChunk, overlap)
		currentChunk = overlapSentences
		wordCount = countWords(currentChunk)
	}

	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence == "" {
			continue
		}

		sentWords := len(strings.Fields(sentence))
		if sentWords == 0 {
			continue
		}

		// Очень длинные предложения делим дополнительно по словам,
		// чтобы не создавать сверхдлинные чанки.
		if sentWords > maxWords {
			flushCurrent()
			currentChunk = nil
			wordCount = 0

			longParts := ChunkByWords(sentence, maxWords, overlap)
			for _, part := range longParts {
				trimmed := strings.TrimSpace(part)
				if trimmed != "" {
					chunks = append(chunks, trimmed)
				}
			}
			continue
		}

		// Если добавление предложения превысит лимит и у нас уже есть контент
		if wordCount+sentWords > maxWords && len(currentChunk) > 0 {
			flushCurrent()
		}

		currentChunk = append(currentChunk, sentence)
		wordCount += sentWords
	}

	if len(currentChunk) > 0 {
		chunks = append(chunks, strings.Join(currentChunk, " "))
	}

	return cleanupChunks(chunks, max(20, maxWords/5))
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
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	// Нормализуем пробелы, но оставляем пунктуацию для корректного разбиения.
	text = strings.Join(strings.Fields(text), " ")

	// Берем последовательности до терминального знака или до конца строки.
	re := regexp.MustCompile(`[^.!?…]+(?:[.!?…]+|$)`)
	parts := re.FindAllString(text, -1)
	if len(parts) == 0 {
		return []string{text}
	}

	var sentences []string

	for _, part := range parts {
		trimmed := strings.TrimSpace(part)
		if len(trimmed) > 0 {
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

	// Если целиком ни одно предложение не поместилось в overlap,
	// берем хвост последнего предложения по словам.
	if len(result) == 0 && overlapWords > 0 {
		last := strings.Fields(sentences[len(sentences)-1])
		if len(last) > overlapWords {
			result = append(result, strings.Join(last[len(last)-overlapWords:], " "))
		}
	}

	return result
}

func cleanupChunks(chunks []string, minWords int) []string {
	if len(chunks) == 0 {
		return chunks
	}
	if minWords <= 0 {
		minWords = 1
	}

	clean := make([]string, 0, len(chunks))
	for _, chunk := range chunks {
		normalized := strings.TrimSpace(strings.Join(strings.Fields(chunk), " "))
		if normalized == "" {
			continue
		}

		if len(clean) > 0 && clean[len(clean)-1] == normalized {
			continue
		}

		if len(clean) > 0 && len(strings.Fields(normalized)) < minWords {
			clean[len(clean)-1] = strings.TrimSpace(clean[len(clean)-1] + " " + normalized)
			continue
		}

		clean = append(clean, normalized)
	}

	return clean
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
