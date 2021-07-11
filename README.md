# E2A10

## Objectives
The objective of this assignment is to replace the (random) embedding of the code presented in the class with GloVe embeddings. 

## GloVe Embeddings
GloVe stands for 'Global Vectors for Word Representation'. It is an unsupervised learning algorithm used to obtain an efficient vector represntations of words in order to better capture their meanings and their relations. A frequently given example is that the 'distance' between the word `man` and `woman` must be close to the distance between `king` and `queen`.


![](https://nlp.stanford.edu/projects/glove/images/man_woman_small.jpg)


These embeddings are available at [nlp.stanford.edu](https://nlp.stanford.edu/projects/glove/). There are a number of models available that differ in the source of text, the embedding dimensions, and the number of words in the vocabulary.

## My Work

### Code Segments

#### Building the list of vectors
```python
glove_words = []
idx = 0
glove_word2id = {}
glove_vectors = []

with open(f'{glove_path}/glove.twitter.27B.200d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        glove_words.append(word)
        glove_word2id[word] = idx
        idx += 1
        word_vector = np.array(line[1:]).astype(np.float)
        glove_vectors.append(word_vector)
```

### Previous Attempts

First, I started with the Wikipedia (400K vocab, 300d vectors). I wanted to know how many of the vectors in my training data was missing 




