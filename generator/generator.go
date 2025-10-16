package generator

import (
	"bufio"
	"fmt"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	"markmach/tokenizer"
	"markmach/trainer"
)

type AnswerGenerator struct {
	chain        *trainer.MarkovChain
	tokenizer    *tokenizer.Tokenizer
	maxLength    int
	tokenEntropy map[string]float64
	minEntropy   float64
	maxEntropy   float64
}

// Настройки генератора
type Config struct {
	MaxLength          int
	UsePunctuation     bool
	MaxThematicEntropy float64
}

func NewAnswerGenerator(chain *trainer.MarkovChain, config Config) *AnswerGenerator {
	tokenizerConfig := tokenizer.Config{
		KeepPunctuation: config.UsePunctuation,
		ToLowerCase:     true,
	}

	generator := &AnswerGenerator{
		chain:        chain,
		tokenizer:    tokenizer.NewTokenizer(tokenizerConfig),
		maxLength:    config.MaxLength,
		tokenEntropy: make(map[string]float64),
		minEntropy:   math.MaxFloat64,
		maxEntropy:   -math.MaxFloat64,
	}
	generator.analyzeEntropy()

	return generator
}

// Анализируем энтропию токенов в цепи Маркова
func (g *AnswerGenerator) analyzeEntropy() {
	fmt.Printf("Analyzing entropy for %d tokens...\n", len(g.chain.Vocab))

	tokenContexts := make(map[string]map[string]int)
	for prefix, suffixes := range g.chain.Chain {
		prefixTokens := g.parsePrefix(prefix)

		for _, token := range prefixTokens {
			g.addTokenContext(token, prefixTokens, tokenContexts)
		}

		for suffix := range suffixes {
			g.addTokenContext(suffix, prefixTokens, tokenContexts)
		}
	}

	for token, contexts := range tokenContexts {
		entropy := g.calculateEntropy(contexts)
		g.tokenEntropy[token] = entropy

		if entropy < g.minEntropy {
			g.minEntropy = entropy
		}
		if entropy > g.maxEntropy {
			g.maxEntropy = entropy
		}
	}

	fmt.Printf("Entropy analysis complete. Range: [%.3f, %.3f]\n", g.minEntropy, g.maxEntropy)
	g.printTopThematicTokens(15)
}

// Добавляем контекст для токена
func (g *AnswerGenerator) addTokenContext(token string, context []string, tokenContexts map[string]map[string]int) {
	if tokenContexts[token] == nil {
		tokenContexts[token] = make(map[string]int)
	}

	contextKey := g.createContextKey(context)
	tokenContexts[token][contextKey]++
}

// Создаем ключ для контекста
func (g *AnswerGenerator) createContextKey(tokens []string) string {
	sorted := make([]string, len(tokens))
	copy(sorted, tokens)
	sort.Strings(sorted)
	return strings.Join(sorted, "|")
}

// Вычисляем энтропию Шеннона для токена
func (g *AnswerGenerator) calculateEntropy(contexts map[string]int) float64 {
	total := 0
	for _, count := range contexts {
		total += count
	}

	if total == 0 {
		return math.MaxFloat64
	}

	entropy := 0.0
	for _, count := range contexts {
		probability := float64(count) / float64(total)
		entropy -= probability * math.Log2(probability)
	}

	return entropy
}

// Проверяем, является ли токен тематическим
func (g *AnswerGenerator) isThematicToken(token string) bool {
	entropy, exists := g.tokenEntropy[token]
	if !exists {
		return false
	}

	return entropy < 2.0
}

// Фильтруем ключевые слова по тематике
func (g *AnswerGenerator) getThematicKeywords(keywords []string) []string {
	var thematic []string
	for _, keyword := range keywords {
		if g.isThematicToken(keyword) {
			thematic = append(thematic, keyword)
		}
	}
	return thematic
}

// Печатаем топ тематических токенов
func (g *AnswerGenerator) printTopThematicTokens(limit int) {
	type tokenEntropy struct {
		token   string
		entropy float64
	}

	var tokens []tokenEntropy
	for token, entropy := range g.tokenEntropy {
		tokens = append(tokens, tokenEntropy{token, entropy})
	}

	sort.Slice(tokens, func(i, j int) bool {
		return tokens[i].entropy < tokens[j].entropy
	})

	fmt.Printf("\n=== Top %d Thematic Tokens ===\n", limit)
	for i := 0; i < min(limit, len(tokens)); i++ {
		fmt.Printf("%d. %s (entropy: %.3f)\n", i+1, tokens[i].token, tokens[i].entropy)
	}
	fmt.Println()
}

// Разбираем строковый префикс обратно в токены
func (g *AnswerGenerator) parsePrefix(prefix string) []string {
	cleaned := strings.TrimPrefix(prefix, "[")
	cleaned = strings.TrimSuffix(cleaned, "]")

	if cleaned == "" {
		return []string{}
	}

	tokens := strings.Split(cleaned, " ")
	var result []string
	for _, token := range tokens {
		cleanToken := strings.TrimSpace(token)
		if cleanToken != "" {
			result = append(result, cleanToken)
		}
	}

	return result
}

// Генерируем ответ на вопрос пользователя
func (g *AnswerGenerator) GenerateAnswer(question string) string {
	keywords := g.extractKeywords(question)

	if len(keywords) == 0 {
		return "Пожалуйста, задайте вопрос."
	}

	thematicKeywords := g.getThematicKeywords(keywords)
	searchKeywords := keywords
	if len(thematicKeywords) > 0 {
		searchKeywords = thematicKeywords
		fmt.Printf("Using thematic keywords: %v\n", thematicKeywords)
	}

	relevantSentences := g.chain.Search(searchKeywords, 5)
	if len(relevantSentences) == 0 {
		return "К сожалению, я не нашел информации по вашему вопросу в изученном материале."
	}

	bestSentence := g.findBestSentence(relevantSentences, searchKeywords)
	answer := g.generateFromSentence(bestSentence, searchKeywords)

	return g.formatAnswer(answer)
}

// Учет тематики
func (g *AnswerGenerator) generateFromSentence(sentence string, keywords []string) string {
	tokens := g.tokenizer.Tokenize(sentence)

	var cleanTokens []string
	for _, token := range tokens {
		if token != "<start>" && token != "<end>" {
			cleanTokens = append(cleanTokens, token)
		}
	}

	if len(cleanTokens) == 0 {
		return sentence
	}

	startPos := g.findThematicStartPosition(cleanTokens, keywords)
	result := g.generateThematicContinuation(cleanTokens, startPos, keywords)

	return result
}

// Находим позицию с максимальной тематической релевантностью
func (g *AnswerGenerator) findThematicStartPosition(tokens []string, keywords []string) int {
	bestPos := 0
	bestScore := -1.0

	for i := 0; i <= len(tokens)-g.chain.Order+1; i++ {
		end := min(i+g.chain.Order-1, len(tokens))
		segment := tokens[i:end]

		prefixKey := fmt.Sprintf("%v", segment)
		if _, exists := g.chain.Chain[prefixKey]; !exists {
			continue
		}

		score := 0.0
		for _, token := range segment {
			for _, keyword := range keywords {
				if token == keyword {
					entropy, exists := g.tokenEntropy[token]
					if exists {
						score += 1.0 / (entropy + 0.1)
					} else {
						score += 0.5
					}
					break
				}
			}
		}

		if score > bestScore {
			bestScore = score
			bestPos = i
		}
	}

	return bestPos
}

// Генерируем продолжение с приоритетом тематических токенов
func (g *AnswerGenerator) generateThematicContinuation(tokens []string, startPos int, keywords []string) string {
	var result []string

	if startPos > 0 {
		result = append(result, tokens[:startPos]...)
	}

	currentPos := startPos
	attempts := 0
	maxAttempts := 10

	for len(result) < g.maxLength && currentPos < len(tokens) && attempts < maxAttempts {
		var currentPrefix []string
		if len(result) >= g.chain.Order-1 {
			currentPrefix = result[len(result)-g.chain.Order+1:]
		} else {
			needed := g.chain.Order - 1 - len(result)
			startIdx := max(0, currentPos-needed)
			currentPrefix = append(tokens[startIdx:currentPos], result...)
		}

		nextTokens := g.chain.GetNextTokens(currentPrefix)
		if nextTokens != nil && len(nextTokens) > 0 {
			nextToken := g.selectThematicToken(nextTokens, keywords)

			if nextToken == "<end>" {
				break
			}

			result = append(result, nextToken)
			attempts = 0
		} else {
			if currentPos < len(tokens) {
				result = append(result, tokens[currentPos])
				currentPos++
			}
			attempts++
		}
	}

	if len(result) < g.maxLength/2 && currentPos < len(tokens) {
		remaining := min(g.maxLength-len(result), len(tokens)-currentPos)
		result = append(result, tokens[currentPos:currentPos+remaining]...)
	}

	return g.tokenizer.JoinTokens(result)
}

// Выбираем следующий токен с учетом тематики
func (g *AnswerGenerator) selectThematicToken(probabilities map[string]float64, keywords []string) string {
	weightedProbabilities := make(map[string]float64)
	totalWeight := 0.0

	for token, prob := range probabilities {
		weight := prob

		for _, keyword := range keywords {
			if token == keyword {
				entropy, exists := g.tokenEntropy[token]
				if exists && entropy < 2.0 {
					weight *= (3.0 - entropy)
				} else {
					weight *= 1.5
				}
				break
			}
		}

		weightedProbabilities[token] = weight
		totalWeight += weight
	}

	if totalWeight > 0 {
		for token := range weightedProbabilities {
			weightedProbabilities[token] /= totalWeight
		}
	}

	return g.selectNextToken(weightedProbabilities)
}

// Извлекаем ключевые слова из вопроса
func (g *AnswerGenerator) extractKeywords(question string) []string {
	tokens := g.tokenizer.Tokenize(question)

	var keywords []string
	stopWords := map[string]bool{
		"что": true, "как": true, "зачем": true, "почему": true,
		"где": true, "когда": true, "какой": true, "какая": true,
		"какое": true, "какие": true, "объясни": true, "расскажи": true,
		"пожалуйста": true, "мог": true, "бы": true, "ли": true,
		"<start>": true, "<end>": true,
	}

	for _, token := range tokens {
		if !stopWords[token] && len(token) > 1 && !isPunctuation(token) {
			keywords = append(keywords, token)
		}
	}

	return keywords
}

// Находим наиболее релевантное предложение
func (g *AnswerGenerator) findBestSentence(sentences []string, keywords []string) string {
	if len(sentences) == 0 {
		return ""
	}

	bestScore := -1
	bestSentence := sentences[0]

	for _, sentence := range sentences {
		score := 0
		tokens := g.tokenizer.Tokenize(sentence)

		for _, token := range tokens {
			for _, keyword := range keywords {
				if token == keyword {
					score++
					break
				}
			}
		}

		if score > bestScore {
			bestScore = score
			bestSentence = sentence
		}
	}

	return bestSentence
}

// Выбираем следующий токен на основе вероятностей
func (g *AnswerGenerator) selectNextToken(probabilities map[string]float64) string {
	rand.Seed(time.Now().UnixNano())
	r := rand.Float64()

	cumulative := 0.0
	for token, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			return token
		}
	}

	var maxToken string
	maxProb := 0.0
	for token, prob := range probabilities {
		if prob > maxProb {
			maxProb = prob
			maxToken = token
		}
	}
	return maxToken
}

// Форматируем конечный ответ
func (g *AnswerGenerator) formatAnswer(answer string) string {
	answer = strings.ReplaceAll(answer, "<start>", "")
	answer = strings.ReplaceAll(answer, "<end>", "")
	answer = strings.TrimSpace(answer)
	answer = regexp.MustCompile(`\s+`).ReplaceAllString(answer, " ")
	answer = regexp.MustCompile(`\s*,\s*`).ReplaceAllString(answer, ", ")
	answer = regexp.MustCompile(`\s*\.\s*`).ReplaceAllString(answer, ". ")
	answer = strings.TrimSpace(answer)

	if len(answer) > 0 {
		runes := []rune(answer)
		if len(runes) > 0 {
			runes[0] = unicode.ToUpper(runes[0])
			answer = string(runes)
		}
	}

	if len(answer) > 0 && !strings.HasSuffix(answer, ".") &&
		!strings.HasSuffix(answer, "!") && !strings.HasSuffix(answer, "?") {
		answer += "."
	}

	return answer
}

func (g *AnswerGenerator) InteractiveMode() {
	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("Вопрос: ")
		if !scanner.Scan() {
			break
		}

		question := strings.TrimSpace(scanner.Text())

		if question == "выход" || question == "exit" || question == "quit" {
			fmt.Println("До свидания!")
			break
		}

		if question == "" {
			continue
		}

		fmt.Println("Обрабатываю вопрос...")
		answer := g.GenerateAnswer(question)
		fmt.Printf("Ответ: %s\n\n", answer)
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Ошибка чтения ввода: %v\n", err)
	}
}

func isPunctuation(token string) bool {
	if len(token) == 0 {
		return false
	}

	punctuation := ",.!?;:"
	for _, p := range punctuation {
		if string(p) == token {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
