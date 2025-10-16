package tokenizer

import (
	"regexp"
	"strings"
	"unicode"
)

// Разбивку текста на токены
type Tokenizer struct {
	punctuationRegex *regexp.Regexp
	wordRegex        *regexp.Regexp
	keepPunctuation  bool
}

// Настройки токенизатора
type Config struct {
	KeepPunctuation bool // Сохранять знаки препинания как отдельные токены
	ToLowerCase     bool // Приводить к нижнему регистру
}

// Создание нового экземпляря токенизатора
func NewTokenizer(config Config) *Tokenizer {
	t := &Tokenizer{
		keepPunctuation: config.KeepPunctuation,
	}

	t.punctuationRegex = regexp.MustCompile(`[.!?,;:'"()\[\]{}…–—]`)
	t.wordRegex = regexp.MustCompile(`[\p{L}\p{N}-]+`)

	return t
}

// Разбитие текста на токены
func (t *Tokenizer) Tokenize(text string) []string {
	text = strings.ToLower(text)

	var tokens []string
	if t.keepPunctuation {
		tokens = t.tokenizeWithPunctuation(text)
	} else {
		tokens = t.tokenizeWordsOnly(text)
	}
	tokens = t.addSpecialTokens(tokens)

	return tokens
}

// Разбитие массива предложений на токены
func (t *Tokenizer) TokenizeSentences(sentences []string) [][]string {
	var tokenizedSentences [][]string

	for _, sentence := range sentences {
		if strings.TrimSpace(sentence) == "" {
			continue
		}
		tokens := t.Tokenize(sentence)
		if len(tokens) > 0 {
			tokenizedSentences = append(tokenizedSentences, tokens)
		}
	}

	return tokenizedSentences
}

// Разбитие текста на токены с сохранением знаков препинания
func (t *Tokenizer) tokenizeWithPunctuation(text string) []string {
	var tokens []string
	runes := []rune(text)
	i := 0

	for i < len(runes) {
		r := runes[i]

		if unicode.IsSpace(r) {
			i++
			continue
		}

		if t.isPunctuation(r) {
			tokens = append(tokens, string(r))
			i++
		} else if unicode.IsLetter(r) || unicode.IsNumber(r) || r == '-' {
			start := i
			for i < len(runes) && (unicode.IsLetter(runes[i]) || unicode.IsNumber(runes[i]) || runes[i] == '-') {
				i++
			}
			word := string(runes[start:i])
			if word != "start" && word != "end" {
				tokens = append(tokens, word)
			}
		} else {
			i++
		}
	}

	return tokens
}

// Разбитие текста только на слова (без знаков препинания)
func (t *Tokenizer) tokenizeWordsOnly(text string) []string {
	matches := t.wordRegex.FindAllString(text, -1)

	var tokens []string
	for _, match := range matches {
		if match != "" && match != "-" {
			tokens = append(tokens, match)
		}
	}

	return tokens
}

// Проверка, является ли руна знаком препинания
func (t *Tokenizer) isPunctuation(r rune) bool {
	return t.punctuationRegex.MatchString(string(r))
}

// Добавление специальных токенов в начало и конец
func (t *Tokenizer) addSpecialTokens(tokens []string) []string {
	if len(tokens) == 0 {
		return []string{"<start>", "<end>"}
	}

	result := make([]string, 0, len(tokens)+2)
	result = append(result, "<start>")
	result = append(result, tokens...)
	result = append(result, "<end>")

	return result
}

// Объединение токенов обратно в текст (для отладки)
func (t *Tokenizer) JoinTokens(tokens []string) string {
	var result strings.Builder

	for i, token := range tokens {
		if i > 0 && !t.isPunctuationToken(token) {
			result.WriteString(" ")
		}
		result.WriteString(token)
	}

	return result.String()
}

// Проверка, является ли токен знаком препинания
func (t *Tokenizer) isPunctuationToken(token string) bool {
	if len(token) == 0 {
		return false
	}

	if token == "<start>" || token == "<end>" {
		return false
	}

	return t.punctuationRegex.MatchString(string(token[0]))
}

// Создание словаря уникальных токенов
func (t *Tokenizer) Vocabulary(tokenizedSentences [][]string) map[string]int {
	vocab := make(map[string]int)

	for _, sentence := range tokenizedSentences {
		for _, token := range sentence {
			vocab[token]++
		}
	}

	return vocab
}

// Фильтрация токенов по минимальной частоте
func (t *Tokenizer) FilterByFrequency(vocab map[string]int, minFrequency int) map[string]int {
	filtered := make(map[string]int)

	for token, freq := range vocab {
		if freq >= minFrequency {
			filtered[token] = freq
		}
	}

	return filtered
}
