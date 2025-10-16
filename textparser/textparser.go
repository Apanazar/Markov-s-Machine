package textparser

import (
	"bufio"
	"fmt"
	"os"
	"regexp"
	"strings"
	"unicode"
)

// Результаты парсинга текста
type ParseResult struct {
	RawText    string   // Очищенный сырой текст
	Sentences  []string // Разбивка на предложения
	Paragraphs []string // Разбивка на абзацы
}

// Парсинг текстовых файлов
type TextParser struct {
	htmlTagRegex     *regexp.Regexp
	multiSpaceRegex  *regexp.Regexp
	sentenceEndRegex *regexp.Regexp
}

// Создание нового экземпляр парсера
func NewTextParser() *TextParser {
	return &TextParser{
		htmlTagRegex:     regexp.MustCompile(`<[^>]*>`),
		multiSpaceRegex:  regexp.MustCompile(`\s+`),
		sentenceEndRegex: regexp.MustCompile(`([.!?…]+)`),
	}
}

// Обработка текстового файла и возврат структурированных данных
func (p *TextParser) Parse(filename string) (*ParseResult, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	cleanText := p.cleanText(string(content))

	sentences := p.splitSentences(cleanText)
	paragraphs := p.splitParagraphs(string(content))

	return &ParseResult{
		RawText:    cleanText,
		Sentences:  sentences,
		Paragraphs: paragraphs,
	}, nil
}

// Очистка текста от лишнего форматирования
func (p *TextParser) cleanText(text string) string {
	text = p.htmlTagRegex.ReplaceAllString(text, "")
	text = p.multiSpaceRegex.ReplaceAllString(text, " ")
	text = regexp.MustCompile(`[^\p{L}\p{N}\p{P}\s]`).ReplaceAllString(text, "")
	text = strings.TrimSpace(text)
	return text
}

// Разбивка текста на предложения
func (p *TextParser) splitSentences(text string) []string {
	text = regexp.MustCompile(`\s*([.!?…])\s*`).ReplaceAllString(text, "$1 ")
	text = strings.TrimSpace(text)

	var sentences []string
	start := 0

	for i := 0; i < len(text); i++ {
		if p.isSentenceEnd(text, i) {
			end := i + 1
			sentence := strings.TrimSpace(text[start:end])
			if len(sentence) > 10 {
				sentences = append(sentences, sentence)
			}
			start = end
			for start < len(text) && unicode.IsSpace(rune(text[start])) {
				start++
			}
			i = start - 1
		}
	}

	if start < len(text) {
		lastSentence := strings.TrimSpace(text[start:])
		if len(lastSentence) > 10 {
			sentences = append(sentences, lastSentence)
		}
	}

	return sentences
}

// Проверка является ли позиция концом предложения
func (p *TextParser) isSentenceEnd(text string, pos int) bool {
	if pos >= len(text) {
		return false
	}

	char := text[pos]

	if char == '.' || char == '!' || char == '?' {
		if p.isAbbreviation(text, pos) {
			return false
		}

		if pos+1 >= len(text) {
			return true
		}
		nextChar := text[pos+1]
		return unicode.IsSpace(rune(nextChar)) || unicode.IsUpper(rune(nextChar))
	}

	return false
}

// Проверка является ли точка частью сокращения
func (p *TextParser) isAbbreviation(text string, pos int) bool {
	if pos < 1 || pos >= len(text) || text[pos] != '.' {
		return false
	}

	abbreviations := []string{"т.д.", "т.п.", "и т.д.", "и т.п.", "др.", "пр.", "см.", "рис.", "стр."}

	for _, abbr := range abbreviations {
		if pos+1 >= len(abbr) && strings.HasSuffix(text[:pos+1], abbr) {
			return true
		}
	}

	return false
}

// Разбивка текста на абзацы
func (p *TextParser) splitParagraphs(text string) []string {
	file, err := os.Open(text)
	if err != nil {
		return p.splitParagraphsFromText(text)
	}
	defer file.Close()

	var paragraphs []string
	var currentParagraph strings.Builder
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" {
			if currentParagraph.Len() > 0 {
				paragraphs = append(paragraphs, currentParagraph.String())
				currentParagraph.Reset()
			}
		} else {
			if currentParagraph.Len() > 0 {
				currentParagraph.WriteString(" ")
			}
			currentParagraph.WriteString(line)
		}
	}

	if currentParagraph.Len() > 0 {
		paragraphs = append(paragraphs, currentParagraph.String())
	}

	return paragraphs
}

// Разбивка текста на абзацы когда файл недоступен
func (p *TextParser) splitParagraphsFromText(text string) []string {
	rawParagraphs := strings.Split(text, "\n\n")

	var paragraphs []string
	for _, paragraph := range rawParagraphs {
		paragraph = strings.TrimSpace(paragraph)
		paragraph = regexp.MustCompile(`\n+`).ReplaceAllString(paragraph, " ")
		paragraph = strings.TrimSpace(paragraph)

		if paragraph != "" && len(paragraph) > 20 {
			paragraphs = append(paragraphs, paragraph)
		}
	}

	if len(paragraphs) <= 1 {
		paragraphs = nil
		lines := strings.Split(text, "\n")
		var currentParagraph strings.Builder

		for _, line := range lines {
			line = strings.TrimSpace(line)
			if line == "" {
				if currentParagraph.Len() > 0 {
					paragraphs = append(paragraphs, currentParagraph.String())
					currentParagraph.Reset()
				}
			} else {
				if currentParagraph.Len() > 0 {
					currentParagraph.WriteString(" ")
				}
				currentParagraph.WriteString(line)
			}
		}

		if currentParagraph.Len() > 0 {
			paragraphs = append(paragraphs, currentParagraph.String())
		}
	}

	return paragraphs
}

// Сохранение всех результатов парсинга в файлы
func (p *TextParser) SaveResults(result *ParseResult, baseFilename string) error {
	if err := p.SaveSentences(result.Sentences, baseFilename+"_sentences.txt"); err != nil {
		return err
	}

	if err := p.SaveParagraphs(result.Paragraphs, baseFilename+"_paragraphs.txt"); err != nil {
		return err
	}

	if err := os.WriteFile(baseFilename+"_cleaned.txt", []byte(result.RawText), 0644); err != nil {
		return err
	}

	return nil
}

// Сохранение предложений в файл
func (p *TextParser) SaveSentences(sentences []string, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for i, sentence := range sentences {
		_, err := writer.WriteString(fmt.Sprintf("%d: %s\n", i+1, sentence))
		if err != nil {
			return err
		}
	}
	return writer.Flush()
}

// Сохранение абзацев в файл
func (p *TextParser) SaveParagraphs(paragraphs []string, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	for i, paragraph := range paragraphs {
		_, err := writer.WriteString(fmt.Sprintf("=== Paragraph %d ===\n%s\n\n", i+1, paragraph))
		if err != nil {
			return err
		}
	}
	return writer.Flush()
}

// Загрузка ранее спарсенных данных из файлов
func (p *TextParser) LoadParsedData(baseFilename string) (*ParseResult, error) {
	cleanedBytes, err := os.ReadFile(baseFilename + "_cleaned.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read cleaned text: %w", err)
	}

	sentences, err := p.loadSentences(baseFilename + "_sentences.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read sentences: %w", err)
	}

	paragraphs, err := p.loadParagraphs(baseFilename + "_paragraphs.txt")
	if err != nil {
		return nil, fmt.Errorf("failed to read paragraphs: %w", err)
	}

	return &ParseResult{
		RawText:    string(cleanedBytes),
		Sentences:  sentences,
		Paragraphs: paragraphs,
	}, nil
}

// Загрузка предложений из файла
func (p *TextParser) loadSentences(filename string) ([]string, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(content), "\n")
	var sentences []string

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ": ", 2)
		if len(parts) == 2 {
			sentences = append(sentences, parts[1])
		}
	}

	return sentences, nil
}

// Загрузка абзацев из файла
func (p *TextParser) loadParagraphs(filename string) ([]string, error) {
	content, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	sections := strings.Split(string(content), "\n\n")
	var paragraphs []string

	for _, section := range sections {
		section = strings.TrimSpace(section)
		if section == "" {
			continue
		}
		lines := strings.Split(section, "\n")
		if len(lines) > 1 {
			paragraphs = append(paragraphs, strings.Join(lines[1:], " "))
		}
	}

	return paragraphs, nil
}
