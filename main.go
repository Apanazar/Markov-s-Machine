package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"markmach/generator"
	"markmach/textparser"
	"markmach/tokenizer"
	"markmach/trainer"
)

func main() {
	parseCmd := flag.NewFlagSet("parse", flag.ExitOnError)
	parseFile := parseCmd.String("file", "", "Path to the text file to parse")

	tokenizeCmd := flag.NewFlagSet("tokenize", flag.ExitOnError)
	tokenizeFile := tokenizeCmd.String("file", "", "Path to the parsed data file (from parse command)")
	keepPunctuation := tokenizeCmd.Bool("punctuation", false, "Keep punctuation as separate tokens")
	tokenizeUseSentences := tokenizeCmd.Bool("sentences", true, "Use sentences for tokenization")
	tokenizeUseParagraphs := tokenizeCmd.Bool("paragraphs", false, "Use paragraphs for tokenization")

	trainCmd := flag.NewFlagSet("train", flag.ExitOnError)
	trainFile := trainCmd.String("file", "", "Path to the parsed data file")
	modelPath := trainCmd.String("model", "output/markov_model.json", "Path to save the trained model")
	order := trainCmd.Int("order", 3, "Order of Markov chain (2 for bigrams, 3 for trigrams, etc.)")
	trainUseSentences := trainCmd.Bool("sentences", true, "Use sentences for training")
	trainUseParagraphs := trainCmd.Bool("paragraphs", false, "Use paragraphs for training")

	chatCmd := flag.NewFlagSet("chat", flag.ExitOnError)
	chatModelPath := chatCmd.String("model", "output/markov_model.json", "Path to the trained model")
	maxLength := chatCmd.Int("length", 50, "Maximum answer length in tokens")
	maxEntropy := chatCmd.Float64("entropy", 2.0, "Max entropy for thematic tokens")

	if len(os.Args) < 2 {
		fmt.Println("Expected 'parse', 'tokenize' or 'train' subcommand")
		fmt.Println("Usage: go run main.go parse --file path/to/file.txt")
		fmt.Println("Usage: go run main.go tokenize --file path/to/parsed_data.txt [--punctuation] [--sentences|--paragraphs]")
		fmt.Println("Usage: go run main.go train --file path/to/parsed_data.txt [--order 3] [--sentences|--paragraphs] [--model output/model.json]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "parse":
		parseCmd.Parse(os.Args[2:])
		if *parseFile == "" {
			fmt.Println("Please provide a file path using --file flag")
			os.Exit(1)
		}

		parser := textparser.NewTextParser()
		result, err := parser.Parse(*parseFile)
		if err != nil {
			log.Fatalf("Error parsing file: %v", err)
		}

		fmt.Printf("Successfully parsed file: %s\n", *parseFile)
		fmt.Printf("Number of sentences: %d\n", len(result.Sentences))
		fmt.Printf("Number of paragraphs: %d\n", len(result.Paragraphs))
		fmt.Printf("Raw text length: %d characters\n", len(result.RawText))

		fmt.Println("\n=== First 3 sentences ===")
		for i, sentence := range result.Sentences {
			if i >= 3 {
				break
			}
			fmt.Printf("%d: %s\n", i+1, sentence)
		}

		fmt.Println("\n=== First 2 paragraphs ===")
		for i, paragraph := range result.Paragraphs {
			if i >= 2 {
				break
			}
			fmt.Printf("%d: %s\n", i+1, paragraph)
		}

		outputBase := "output/result"
		err = parser.SaveResults(result, outputBase)
		if err != nil {
			log.Printf("Warning: could not save results to files: %v", err)
		} else {
			fmt.Printf("\nResults saved to %s_*.txt files\n", outputBase)
		}

	case "tokenize":
		tokenizeCmd.Parse(os.Args[2:])
		if *tokenizeFile == "" {
			fmt.Println("Please provide a file path using --file flag")
			os.Exit(1)
		}

		parser := textparser.NewTextParser()
		result, err := parser.LoadParsedData(*tokenizeFile)
		if err != nil {
			log.Fatalf("Error loading parsed data: %v", err)
		}

		tokenizerConfig := tokenizer.Config{
			KeepPunctuation: *keepPunctuation,
			ToLowerCase:     true,
		}
		tkz := tokenizer.NewTokenizer(tokenizerConfig)

		fmt.Printf("Tokenizing parsed data from: %s\n", *tokenizeFile)
		fmt.Printf("Keep punctuation: %v\n", *keepPunctuation)
		fmt.Printf("Using sentences: %v\n", *tokenizeUseSentences)
		fmt.Printf("Using paragraphs: %v\n", *tokenizeUseParagraphs)

		var tokenizedData [][]string
		var dataType string

		if *tokenizeUseSentences {
			tokenizedData = tkz.TokenizeSentences(result.Sentences)
			dataType = "sentences"
			fmt.Printf("Number of sentences: %d\n", len(tokenizedData))
		} else if *tokenizeUseParagraphs {
			tokenizedData = tkz.TokenizeSentences(result.Paragraphs)
			dataType = "paragraphs"
			fmt.Printf("Number of paragraphs: %d\n", len(tokenizedData))
		} else {
			tokenizedData = [][]string{tkz.Tokenize(result.RawText)}
			dataType = "full text"
			fmt.Printf("Full text tokenized\n")
		}

		vocab := tkz.Vocabulary(tokenizedData)
		fmt.Printf("Vocabulary size: %d unique tokens\n", len(vocab))

		fmt.Printf("\n=== First 3 tokenized %s ===\n", dataType)
		for i, tokens := range tokenizedData {
			if i >= 3 {
				break
			}
			fmt.Printf("%d: %v\n", i+1, tokens)
			fmt.Printf("   Reconstructed: %s\n", tkz.JoinTokens(tokens))
		}

		fmt.Println("\n=== Token statistics ===")
		totalTokens := 0
		for _, tokens := range tokenizedData {
			totalTokens += len(tokens)
		}
		fmt.Printf("Total tokens: %d\n", totalTokens)
		fmt.Printf("Average tokens per %s: %.1f\n", dataType, float64(totalTokens)/float64(len(tokenizedData)))

		fmt.Println("\n=== Top 20 most frequent tokens ===")
		count := 0
		for token, freq := range vocab {
			if count >= 20 {
				break
			}

			if token != "<start>" && token != "<end>" {
				fmt.Printf("  %s: %d\n", token, freq)
				count++
			}
		}

		err = saveTokenizedData(tokenizedData, vocab, fmt.Sprintf("output/tokens_%s", dataType))
		if err != nil {
			log.Printf("Warning: could not save tokenized data: %v", err)
		} else {
			fmt.Printf("\nTokenized data saved to output/tokens_%s_*.txt\n", dataType)
		}

	case "train":
		trainCmd.Parse(os.Args[2:])
		if *trainFile == "" {
			fmt.Println("Please provide a file path using --file flag")
			os.Exit(1)
		}

		parser := textparser.NewTextParser()
		result, err := parser.LoadParsedData(*trainFile)
		if err != nil {
			log.Fatalf("Error loading parsed data: %v", err)
		}

		tokenizerConfig := tokenizer.Config{
			KeepPunctuation: true,
			ToLowerCase:     true,
		}
		tkz := tokenizer.NewTokenizer(tokenizerConfig)

		var tokenizedData [][]string

		if *trainUseSentences {
			tokenizedData = tkz.TokenizeSentences(result.Sentences)
			fmt.Printf("Training on %d sentences...\n", len(tokenizedData))
		} else if *trainUseParagraphs {
			tokenizedData = tkz.TokenizeSentences(result.Paragraphs)
			fmt.Printf("Training on %d paragraphs...\n", len(tokenizedData))
		} else {
			tokenizedData = [][]string{tkz.Tokenize(result.RawText)}
			fmt.Printf("Training on full text...\n")
		}

		trainConfig := trainer.TrainConfig{
			Order:     *order,
			SaveModel: true,
			ModelPath: *modelPath,
		}

		markovTrainer := trainer.NewMarkovTrainer(trainConfig)
		err = markovTrainer.Train(tokenizedData)
		if err != nil {
			log.Fatalf("Error training model: %v", err)
		}

		err = markovTrainer.Save(*modelPath)
		if err != nil {
			log.Fatalf("Error saving model: %v", err)
		}

		stats := markovTrainer.GetStats()
		fmt.Println("\n=== Model Statistics ===")
		for key, value := range stats {
			fmt.Printf("%s: %v\n", key, value)
		}

		fmt.Println("\n=== Chain Examples ===")
		exampleCount := 0
		for prefix, suffixes := range markovTrainer.Chain {
			if exampleCount >= 5 {
				break
			}
			fmt.Printf("Prefix: %s -> ", prefix)
			for suffix, count := range suffixes {
				fmt.Printf("%s(%d) ", suffix, count)
			}
			fmt.Println()
			exampleCount++
		}

	case "chat":
		chatCmd.Parse(os.Args[2:])

		fmt.Printf("Loading model from %s...\n", *chatModelPath)
		markovChain, err := trainer.Load(*chatModelPath)
		if err != nil {
			log.Fatalf("Error loading model: %v", err)
		}

		generatorConfig := generator.Config{
			MaxLength:          *maxLength,
			UsePunctuation:     true,
			MaxThematicEntropy: *maxEntropy,
		}

		answerGenerator := generator.NewAnswerGenerator(markovChain, generatorConfig)
		answerGenerator.InteractiveMode()

	default:
		fmt.Println("Expected 'parse', 'tokenize', 'train' or 'chat' subcommand")
		os.Exit(1)
	}
}

func saveTokenizedData(sentences [][]string, vocab map[string]int, baseFilename string) error {
	os.MkdirAll("output", 0755)

	file, err := os.Create(baseFilename + "_sentences.txt")
	if err != nil {
		return err
	}
	defer file.Close()

	for i, sentence := range sentences {
		fmt.Fprintf(file, "Sentence %d: %v\n", i+1, sentence)
	}

	file, err = os.Create(baseFilename + "_vocabulary.txt")
	if err != nil {
		return err
	}
	defer file.Close()

	fmt.Fprintln(file, "Token -> Frequency")
	for token, freq := range vocab {
		fmt.Fprintf(file, "%s -> %d\n", token, freq)
	}

	return nil
}
