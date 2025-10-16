`markmach parse --file .\data\data.txt`

`markmach tokenize --file output/result --sentences --punctuation --paragraphs`

`markmach train --file output/result --order 3 --sentences  --model output/markov_model.json`

`markmach chat --length 200 --entropy 1.0`
