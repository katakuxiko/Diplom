package service

import (
	"fmt"
	"log"
	"sort"
	"strings"
	"time"
	"unicode"

	"github.com/google/uuid"
	"github.com/katakuxiko/Diplom/internal/models"
	"github.com/katakuxiko/Diplom/internal/repository"
	"github.com/katakuxiko/Diplom/internal/utils"
	"github.com/pgvector/pgvector-go"
)

type RAGService struct {
	ChunkRepository *repository.ChunkRepository
	llm             *LLMClient
}

type RetrievalDiagnostics struct {
	RetrievalMode     string  `json:"retrieval_mode"`
	RetrievalQuery    string  `json:"retrieval_query,omitempty"`
	FallbackUsed      bool    `json:"fallback_used"`
	FallbackQuery     string  `json:"fallback_query,omitempty"`
	TopK              int     `json:"top_k"`
	ExpandedTopK      int     `json:"expanded_top_k"`
	VectorCandidates  int     `json:"vector_candidates"`
	KeywordCandidates int     `json:"keyword_candidates"`
	CandidatesTotal   int     `json:"candidates_total"`
	SelectedChunks    int     `json:"selected_chunks"`
	MaxCosineDistance float32 `json:"max_cosine_distance"`
	MaxDistanceGap    float32 `json:"max_distance_gap"`
	MinChunkChars     int     `json:"min_chunk_chars"`
	ContextBudget     int     `json:"context_budget"`
	ContextCharsUsed  int     `json:"context_chars_used"`
}

const (
	defaultTopK           = 5
	maxExpandedCandidates = 30
	defaultContextBudget  = 8000
	maxContextBudget      = 20000
	defaultMinChunkChars  = 50
	defaultMaxCosineDist  = float32(0.60)
	defaultMaxDistanceGap = float32(0.18)
	defaultVectorWeight   = float32(0.62)
	defaultKeywordWeight  = float32(0.28)
	defaultRRFWeight      = float32(0.10)
	defaultRRFDenominator = float32(60.0)
)

func NewRAGService(ChunkRepository *repository.ChunkRepository, llm *LLMClient) *RAGService {
	return &RAGService{ChunkRepository: ChunkRepository, llm: llm}
}

func (s *RAGService) Ask(query string, topK int, chatID uuid.UUID, settings *models.AskSettings, accessLevel int, history []models.ChatContextMessage) (string, []models.Chunk, error) {
	answer, chunks, _, err := s.AskWithDiagnostics(query, topK, chatID, settings, accessLevel, history)
	return answer, chunks, err
}

func (s *RAGService) AskWithDiagnostics(query string, topK int, chatID uuid.UUID, settings *models.AskSettings, accessLevel int, history []models.ChatContextMessage) (string, []models.Chunk, RetrievalDiagnostics, error) {
	diagnostics := RetrievalDiagnostics{}

	if topK <= 0 {
		topK = defaultTopK
	}
	diagnostics.TopK = topK
	diagnostics.RetrievalQuery = strings.TrimSpace(query)

	filteredChunks, retrieveErr := s.retrieveChunksForQuery(query, topK, chatID, settings, accessLevel, &diagnostics)
	if retrieveErr != nil {
		return "", nil, diagnostics, retrieveErr
	}

	if len(filteredChunks) == 0 && shouldTryRussianFallback(query) && s.llm != nil {
		translatedQuery, translateErr := s.llm.TranslateQueryToRussianForRetrieval(query, settings)
		if translateErr != nil {
			log.Printf("retrieval translation fallback failed: %v", translateErr)
		} else {
			translatedQuery = strings.TrimSpace(translatedQuery)
			if translatedQuery != "" && !sameNormalizedQuery(query, translatedQuery) {
				fallbackDiagnostics := RetrievalDiagnostics{TopK: topK, RetrievalQuery: translatedQuery}
				fallbackChunks, fallbackErr := s.retrieveChunksForQuery(translatedQuery, topK, chatID, settings, accessLevel, &fallbackDiagnostics)
				if fallbackErr != nil {
					log.Printf("retrieval fallback search failed: %v", fallbackErr)
				} else if len(fallbackChunks) > 0 {
					fallbackDiagnostics.FallbackUsed = true
					fallbackDiagnostics.FallbackQuery = translatedQuery
					diagnostics = fallbackDiagnostics
					filteredChunks = fallbackChunks
				} else {
					diagnostics.FallbackQuery = translatedQuery
				}
			}
		}
	}

	// Если после фильтрации не осталось чанков
	if len(filteredChunks) == 0 {
		return "К сожалению, в загруженных документах не найдено релевантной информации по вашему запросу. Попробуйте переформулировать вопрос или загрузите дополнительные материалы.", nil, diagnostics, nil
	}

	// Build normalized, compact context with a character budget
	// Heuristic: cap context to ~8000 chars (~2k tokens), enough for fast responses
	contextBudget := defaultContextBudget
	if settings != nil && settings.MaxTokens > 0 {
		// Leave room for answer; use ~1.5x of answer tokens as context chars (approx 4 chars per token)
		budget := int(float32(settings.MaxTokens) * 1.5 * 4)
		if budget > 0 {
			if budget > maxContextBudget {
				budget = maxContextBudget
			}
			contextBudget = budget
		}
	}
	diagnostics.ContextBudget = contextBudget

	var b strings.Builder
	used := 0
	for i, ch := range filteredChunks {
		normalized := utils.NormalizeText(ch.Text)
		sourceLabel := strings.TrimSpace(ch.DocName)
		if sourceLabel == "" {
			sourceLabel = strings.TrimSpace(ch.ChunkName)
		}
		header := fmt.Sprintf("Фрагмент %d: ", i+1)
		if sourceLabel != "" {
			header = fmt.Sprintf("Фрагмент %d [%s]: ", i+1, sourceLabel)
		}
		piece := header + normalized + "\n"
		pieceLen := len([]rune(piece))
		if used+pieceLen > contextBudget {
			remaining := contextBudget - used
			if remaining <= 0 {
				break
			}
			piece = utils.TruncateByChars(piece, remaining)
			pieceLen = len([]rune(piece))
		}
		b.WriteString(piece)
		used += pieceLen
		if used >= contextBudget {
			break
		}
	}
	diagnostics.ContextCharsUsed = used
	ctx := b.String()

	startTime := time.Now()
	answer, err := s.llm.AskWithSettings(query, ctx, settings, history)
	fmt.Printf("⏱️  LLM response time: %v\n", time.Since(startTime))
	if err != nil {
		return "", nil, diagnostics, fmt.Errorf("llm error: %w", err)
	}

	return answer, filteredChunks, diagnostics, nil
}

func (s *RAGService) retrieveChunksForQuery(query string, topK int, chatID uuid.UUID, settings *models.AskSettings, accessLevel int, diagnostics *RetrievalDiagnostics) ([]models.Chunk, error) {
	if diagnostics == nil {
		return nil, fmt.Errorf("diagnostics is nil")
	}

	if topK <= 0 {
		topK = defaultTopK
	}
	diagnostics.TopK = topK

	retrievalMode := resolveRetrievalMode(settings)
	diagnostics.RetrievalMode = retrievalMode

	expandedTopK := topK * 2
	if retrievalMode == "hybrid" {
		expandedTopK = topK * 3
	}
	if expandedTopK > maxExpandedCandidates {
		expandedTopK = maxExpandedCandidates
	}
	if expandedTopK < topK {
		expandedTopK = topK
	}
	diagnostics.ExpandedTopK = expandedTopK

	minChunkChars, maxCosineDistance, maxDistanceGap := resolveRetrievalThresholds(settings)
	diagnostics.MinChunkChars = minChunkChars
	diagnostics.MaxCosineDistance = maxCosineDistance
	diagnostics.MaxDistanceGap = maxDistanceGap

	var vectorChunks []models.Chunk
	if retrievalMode != "keyword" {
		v, embErr := s.llm.EmbeddingWithSettings(query, settings)
		if embErr != nil {
			return nil, fmt.Errorf("embedding error: %w", embErr)
		}
		vec := pgvector.NewVector(v)

		vectorResult, searchErr := s.ChunkRepository.SearchByVector(vec, expandedTopK, chatID, accessLevel)
		if searchErr != nil {
			return nil, fmt.Errorf("search error: %w", searchErr)
		}
		for i := range vectorResult {
			vectorResult[i].RetrievalSource = "vector"
			vectorResult[i].HybridScore = cosineDistanceToSimilarity(vectorResult[i].Score)
		}
		vectorChunks = vectorResult
		diagnostics.VectorCandidates = len(vectorChunks)
	} else {
		diagnostics.VectorCandidates = 0
	}

	var candidates []models.Chunk
	switch retrievalMode {
	case "vector":
		candidates = vectorChunks
		diagnostics.KeywordCandidates = 0
	case "keyword":
		keywordChunks, kErr := s.ChunkRepository.SearchByKeyword(query, expandedTopK, chatID, accessLevel)
		if kErr != nil {
			return nil, fmt.Errorf("keyword search error: %w", kErr)
		}
		diagnostics.KeywordCandidates = len(keywordChunks)
		candidates = s.rankKeywordCandidates(query, keywordChunks)
	default:
		keywordChunks, kErr := s.ChunkRepository.SearchByKeyword(query, expandedTopK, chatID, accessLevel)
		if kErr != nil {
			return nil, fmt.Errorf("keyword search error: %w", kErr)
		}
		diagnostics.KeywordCandidates = len(keywordChunks)
		keywordChunks = s.rankKeywordCandidates(query, keywordChunks)
		candidates = s.mergeHybridCandidates(query, vectorChunks, keywordChunks, expandedTopK*2)
	}

	diagnostics.CandidatesTotal = len(candidates)
	filteredChunks := s.filterRelevantChunks(candidates, topK, settings)
	diagnostics.SelectedChunks = len(filteredChunks)

	return filteredChunks, nil
}

func shouldTryRussianFallback(query string) bool {
	q := strings.TrimSpace(strings.ToLower(query))
	if q == "" {
		return false
	}

	latinCount := 0
	cyrillicCount := 0
	for _, r := range q {
		if unicode.In(r, unicode.Latin) {
			latinCount++
		}
		if unicode.In(r, unicode.Cyrillic) {
			cyrillicCount++
		}
	}

	return latinCount >= 3 && cyrillicCount == 0
}

func sameNormalizedQuery(a, b string) bool {
	normalize := func(v string) string {
		return strings.ToLower(strings.Join(strings.Fields(strings.TrimSpace(v)), " "))
	}
	return normalize(a) == normalize(b)
}

func resolveRetrievalMode(settings *models.AskSettings) string {
	if settings == nil {
		return "hybrid"
	}
	mode := strings.ToLower(strings.TrimSpace(settings.RetrievalMode))
	switch mode {
	case "vector", "keyword", "hybrid":
		return mode
	default:
		return "hybrid"
	}
}

func resolveRetrievalThresholds(settings *models.AskSettings) (int, float32, float32) {
	minChunkLen := defaultMinChunkChars
	maxCosineDist := defaultMaxCosineDist
	maxGapFromTopHit := defaultMaxDistanceGap
	if settings != nil {
		if settings.MinChunkChars > 0 {
			minChunkLen = settings.MinChunkChars
		}
		if settings.MaxCosineDistance > 0 {
			maxCosineDist = settings.MaxCosineDistance
		}
		if settings.MaxDistanceGap > 0 {
			maxGapFromTopHit = settings.MaxDistanceGap
		}
	}

	return minChunkLen, maxCosineDist, maxGapFromTopHit
}

// ApplyRetrievalThresholds используется для калибровки порогов на evaluation run.
func (s *RAGService) ApplyRetrievalThresholds(chunks []models.Chunk, topK int, settings *models.AskSettings) []models.Chunk {
	return s.filterRelevantChunks(chunks, topK, settings)
}

// filterRelevantChunks фильтрует чанки по релевантности
func (s *RAGService) filterRelevantChunks(chunks []models.Chunk, maxChunks int, settings *models.AskSettings) []models.Chunk {
	if len(chunks) == 0 {
		return chunks
	}
	if maxChunks <= 0 {
		maxChunks = defaultTopK
	}

	minChunkLen, maxCosineDist, maxGapFromTopHit := resolveRetrievalThresholds(settings)

	useScore := false
	bestScore := float32(1e9)
	for _, ch := range chunks {
		if ch.Score > 0 {
			useScore = true
			if ch.Score < bestScore {
				bestScore = ch.Score
			}
		}
	}

	// Отбираем содержательные фрагменты и, если доступен score, режем хвост по distance.
	filtered := make([]models.Chunk, 0, maxChunks)
	for _, ch := range chunks {
		if len([]rune(strings.TrimSpace(ch.Text))) < minChunkLen {
			continue
		}

		if useScore && ch.Score > 0 {
			if ch.Score > maxCosineDist {
				continue
			}
			if ch.Score-bestScore > maxGapFromTopHit {
				continue
			}
		}

		filtered = append(filtered, ch)
		if len(filtered) >= maxChunks {
			break
		}
	}

	if len(filtered) == 0 {
		for _, ch := range chunks {
			if len([]rune(strings.TrimSpace(ch.Text))) >= minChunkLen {
				filtered = append(filtered, ch)
				if len(filtered) >= maxChunks {
					break
				}
			}
		}
	}

	return filtered
}

func (s *RAGService) rankKeywordCandidates(query string, chunks []models.Chunk) []models.Chunk {
	if len(chunks) == 0 {
		return chunks
	}

	ranked := make([]models.Chunk, 0, len(chunks))
	for _, ch := range chunks {
		ch.KeywordScore = lexicalOverlapScore(query, ch.Text)
		ch.HybridScore = ch.KeywordScore
		ch.RetrievalSource = "keyword"
		ranked = append(ranked, ch)
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].KeywordScore == ranked[j].KeywordScore {
			return len(ranked[i].Text) > len(ranked[j].Text)
		}
		return ranked[i].KeywordScore > ranked[j].KeywordScore
	})

	return ranked
}

func (s *RAGService) mergeHybridCandidates(query string, vectorChunks, keywordChunks []models.Chunk, maxCandidates int) []models.Chunk {
	type scoredCandidate struct {
		chunk       models.Chunk
		vectorRank  int
		keywordRank int
		vectorSim   float32
		keywordSim  float32
		rrf         float32
		total       float32
	}

	if maxCandidates <= 0 {
		maxCandidates = len(vectorChunks) + len(keywordChunks)
	}

	candidates := make(map[string]*scoredCandidate, len(vectorChunks)+len(keywordChunks))

	for i, ch := range vectorChunks {
		key := makeChunkKey(ch)
		cand := &scoredCandidate{
			chunk:       ch,
			vectorRank:  i,
			keywordRank: -1,
			vectorSim:   cosineDistanceToSimilarity(ch.Score),
			keywordSim:  lexicalOverlapScore(query, ch.Text),
		}
		cand.chunk.KeywordScore = cand.keywordSim
		cand.chunk.RetrievalSource = "vector"
		candidates[key] = cand
	}

	for i, ch := range keywordChunks {
		key := makeChunkKey(ch)
		kwScore := ch.KeywordScore
		if kwScore <= 0 {
			kwScore = lexicalOverlapScore(query, ch.Text)
		}

		if cand, ok := candidates[key]; ok {
			cand.keywordRank = i
			if kwScore > cand.keywordSim {
				cand.keywordSim = kwScore
			}
			cand.chunk.KeywordScore = cand.keywordSim
			cand.chunk.RetrievalSource = "hybrid"
			continue
		}

		cand := &scoredCandidate{
			chunk:       ch,
			vectorRank:  -1,
			keywordRank: i,
			vectorSim:   0,
			keywordSim:  kwScore,
		}
		cand.chunk.KeywordScore = kwScore
		cand.chunk.HybridScore = kwScore
		cand.chunk.RetrievalSource = "keyword"
		candidates[key] = cand
	}

	merged := make([]scoredCandidate, 0, len(candidates))
	for _, cand := range candidates {
		cand.rrf = rrf(cand.vectorRank) + rrf(cand.keywordRank)
		cand.total = defaultVectorWeight*cand.vectorSim + defaultKeywordWeight*cand.keywordSim + defaultRRFWeight*cand.rrf
		if cand.vectorRank < 0 {
			cand.total = defaultKeywordWeight*cand.keywordSim + defaultRRFWeight*cand.rrf
		}
		cand.chunk.HybridScore = cand.total
		merged = append(merged, *cand)
	}

	sort.SliceStable(merged, func(i, j int) bool {
		if merged[i].total == merged[j].total {
			if merged[i].vectorSim == merged[j].vectorSim {
				return merged[i].keywordSim > merged[j].keywordSim
			}
			return merged[i].vectorSim > merged[j].vectorSim
		}
		return merged[i].total > merged[j].total
	})

	result := make([]models.Chunk, 0, len(merged))
	for _, cand := range merged {
		result = append(result, cand.chunk)
		if len(result) >= maxCandidates {
			break
		}
	}

	return result
}

func makeChunkKey(ch models.Chunk) string {
	if ch.ID != uuid.Nil {
		return ch.ID.String()
	}
	if ch.DocID != uuid.Nil {
		return ch.DocID.String() + "::" + strings.TrimSpace(ch.ChunkName)
	}
	return strings.TrimSpace(ch.DocName) + "::" + strings.TrimSpace(ch.ChunkName) + "::" + utils.TruncateByChars(strings.TrimSpace(ch.Text), 80)
}

func rrf(rank int) float32 {
	if rank < 0 {
		return 0
	}
	return 1.0 / (defaultRRFDenominator + float32(rank+1))
}

func cosineDistanceToSimilarity(distance float32) float32 {
	similarity := 1 - distance
	if similarity < 0 {
		return 0
	}
	if similarity > 1 {
		return 1
	}
	return similarity
}

func lexicalOverlapScore(query, text string) float32 {
	terms := queryTerms(query)
	if len(terms) == 0 {
		return 0
	}

	lowerText := strings.ToLower(text)
	hits := 0
	for _, term := range terms {
		if strings.Contains(lowerText, term) {
			hits++
		}
	}

	return float32(hits) / float32(len(terms))
}

func queryTerms(query string) []string {
	raw := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	if len(raw) == 0 {
		return nil
	}

	seen := make(map[string]struct{}, len(raw))
	terms := make([]string, 0, len(raw))
	for _, t := range raw {
		t = strings.TrimSpace(t)
		if len([]rune(t)) < 3 {
			continue
		}
		if _, ok := seen[t]; ok {
			continue
		}
		seen[t] = struct{}{}
		terms = append(terms, t)
	}

	return terms
}
