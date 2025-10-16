package trainer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Представление цепи Маркова
type MarkovChain struct {
	Order  int                       // Порядок цепи (N)
	Chain  map[string]map[string]int // Цепь: prefix -> {suffix -> count}
	Sums   map[string]int            // Суммы для быстрого расчета вероятностей
	Index  map[string][]string       // Инвертированный индекс: слово -> предложения
	Vocab  map[string]int            // Словарь токенов с частотами
	Topics map[string][]string
}

// Настройки обучения
type TrainConfig struct {
	Order        int    // Порядок цепи (N-граммы)
	MinFrequency int    // Минимальная частота токена
	SaveModel    bool   // Сохранять модель на диск
	ModelPath    string // Путь для сохранения модели
}

// Создание нового "тренера" цепи Маркова
func NewMarkovTrainer(config TrainConfig) *MarkovChain {
	return &MarkovChain{
		Order: config.Order,
		Chain: make(map[string]map[string]int),
		Sums:  make(map[string]int),
		Index: make(map[string][]string),
		Vocab: make(map[string]int),
	}
}

// Обучение цепи Маркова на токенизированных предложениях
func (mc *MarkovChain) Train(tokenizedSentences [][]string) error {
	if len(tokenizedSentences) == 0 {
		return fmt.Errorf("no data to train on")
	}
	fmt.Printf("Training Markov chain with order %d on %d sentences...\n", mc.Order, len(tokenizedSentences))

	mc.buildIndexAndVocab(tokenizedSentences)
	for _, sentence := range tokenizedSentences {
		mc.processSentence(sentence)
	}
	mc.calculateSums()

	fmt.Printf("Training completed. Chain size: %d prefixes\n", len(mc.Chain))
	fmt.Printf("Vocabulary size: %d tokens\n", len(mc.Vocab))
	fmt.Printf("Index size: %d words\n", len(mc.Index))

	return nil
}

// Строим инвертированный индекс и словарь
func (mc *MarkovChain) buildIndexAndVocab(sentences [][]string) {
	for _, sentence := range sentences {
		originalSentence := joinSentence(sentence)

		for _, token := range sentence {
			if token == "<start>" || token == "<end>" {
				continue
			}
			mc.Vocab[token]++
			mc.Index[token] = append(mc.Index[token], originalSentence)
		}
	}

	for word, sentences := range mc.Index {
		unique := make(map[string]bool)
		var uniqueSentences []string
		for _, s := range sentences {
			if !unique[s] {
				unique[s] = true
				uniqueSentences = append(uniqueSentences, s)
			}
		}
		mc.Index[word] = uniqueSentences
	}
}

// Обработка предложения и добавление его в цепь
func (mc *MarkovChain) processSentence(sentence []string) {
	if len(sentence) < mc.Order {
		return
	}

	for i := 0; i <= len(sentence)-mc.Order; i++ {
		prefix := joinTokens(sentence[i : i+mc.Order-1])
		suffix := sentence[i+mc.Order-1]

		if mc.Chain[prefix] == nil {
			mc.Chain[prefix] = make(map[string]int)
		}
		mc.Chain[prefix][suffix]++
	}
}

// Вычисление суммы для быстрого расчета вероятностей
func (mc *MarkovChain) calculateSums() {
	for prefix, suffixes := range mc.Chain {
		sum := 0
		for _, count := range suffixes {
			sum += count
		}
		mc.Sums[prefix] = sum
	}
}

// Возврат возможных последующих токенов для префикса
func (mc *MarkovChain) GetNextTokens(prefix []string) map[string]float64 {
	prefixKey := joinTokens(prefix)
	suffixes, exists := mc.Chain[prefixKey]
	if !exists {
		return nil
	}

	probabilities := make(map[string]float64)
	total := float64(mc.Sums[prefixKey])

	for suffix, count := range suffixes {
		probabilities[suffix] = float64(count) / total
	}

	return probabilities
}

// Поиск предложений по ключевым словам
func (mc *MarkovChain) Search(keywords []string, limit int) []string {
	sentenceScores := make(map[string]int)

	for _, keyword := range keywords {
		if sentences, exists := mc.Index[keyword]; exists {
			for _, sentence := range sentences {
				sentenceScores[sentence]++
			}
		}
	}

	type scoredSentence struct {
		sentence string
		score    int
	}

	var scored []scoredSentence
	for sentence, score := range sentenceScores {
		scored = append(scored, scoredSentence{sentence, score})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	var results []string
	for i, ss := range scored {
		if i >= limit {
			break
		}
		results = append(results, ss.sentence)
	}

	return results
}

// Сохраняем модель на диск
func (mc *MarkovChain) Save(filepath string) error {
	model := struct {
		Order int                       `json:"order"`
		Chain map[string]map[string]int `json:"chain"`
		Sums  map[string]int            `json:"sums"`
		Index map[string][]string       `json:"index"`
		Vocab map[string]int            `json:"vocab"`
	}{
		Order: mc.Order,
		Chain: mc.Chain,
		Sums:  mc.Sums,
		Index: mc.Index,
		Vocab: mc.Vocab,
	}

	data, err := json.MarshalIndent(model, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model: %w", err)
	}

	err = os.WriteFile(filepath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write model file: %w", err)
	}

	fmt.Printf("Model saved to %s\n", filepath)
	return nil
}

// Загрузка модели с диска
func Load(filepath string) (*MarkovChain, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read model file: %w", err)
	}

	var model struct {
		Order int                       `json:"order"`
		Chain map[string]map[string]int `json:"chain"`
		Sums  map[string]int            `json:"sums"`
		Index map[string][]string       `json:"index"`
		Vocab map[string]int            `json:"vocab"`
	}

	err = json.Unmarshal(data, &model)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal model: %w", err)
	}

	mc := &MarkovChain{
		Order: model.Order,
		Chain: model.Chain,
		Sums:  model.Sums,
		Index: model.Index,
		Vocab: model.Vocab,
	}

	fmt.Printf("Model loaded from %s (order: %d, chain size: %d)\n",
		filepath, mc.Order, len(mc.Chain))
	return mc, nil
}

// Статистика модели
func (mc *MarkovChain) GetStats() map[string]interface{} {
	totalTransitions := 0
	for _, sum := range mc.Sums {
		totalTransitions += sum
	}

	return map[string]interface{}{
		"order":                      mc.Order,
		"prefixes":                   len(mc.Chain),
		"vocabulary_size":            len(mc.Vocab),
		"index_size":                 len(mc.Index),
		"total_transitions":          totalTransitions,
		"avg_transitions_per_prefix": float64(totalTransitions) / float64(len(mc.Chain)),
	}
}

// Объединение токенов в ключ для цепи
func joinTokens(tokens []string) string {
	return fmt.Sprintf("%v", tokens)
}

// Объединение токенов в читаемое предложение
func joinSentence(tokens []string) string {
	var result string
	for i, token := range tokens {
		if i > 0 && !isPunctuation(token) {
			result += " "
		}
		result += token
	}
	return result
}

// Проверяем, является ли токен знаком препинания
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
