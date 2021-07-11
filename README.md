# E2A10

## Objectives
The objective of this assignment is to replace the (random) embedding of the code presented in the class with GloVe embeddings. 

## GloVe Embeddings
GloVe stands for 'Global Vectors for Word Representation'. It is an unsupervised learning algorithm used to obtain an efficient vector represntations of words in order to better capture their meanings and their relations. A frequently given example is that the 'distance' between the word `man` and `woman` must be close to the distance between `king` and `queen`.


![](https://nlp.stanford.edu/projects/glove/images/man_woman_small.jpg)


These embeddings are available at [nlp.stanford.edu](https://nlp.stanford.edu/projects/glove/). There are a number of models available that differ in the source of text, the embedding dimensions, and the number of words in the vocabulary.


## Code Segments

### Building the list of vectors

This code builds a list of GloVe vectors. It also mainains a dictionary for mapping from word to index and a list (which helps to map from index to word).

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
### Unknown Words

To ensure that there are no words in the training data that lack a GloVe representation, the `InputLang` class was populated with the words in our GloVe vectors' list. If the number of words increased from the ones present in GloVe, that indicated the presence of missing words.

```python
self.word2index = { k : v for k , v in sorted(glove_word2id.items(), key=operator.itemgetter(1))}
self.word2count = { word : 1 for word in glove_words }
self.index2word = { i : word for word, i in glove_word2id.items() }
self.n_words = len(glove_words) # 1917494
```

Result: `n_words` changes from **1917494** to **1917495**. The missing word? It is *ooita* 😂

(Theses lines were removed in subseqeunt training steps, defaulting to the `eos` and `sos` tokens - which are (obviously) missing in the GloVe representations since they are not words per se)

### Matrix of Embeddings

After building the dictionary, a matrix of embeddings was populated. This matrix contains only the embeddings of the words in the vocabulary and not the entire 1.9M+ words!

```python
matrix_len = input_lang.n_words
weights_matrix = np.zeros((matrix_len, 300))
words_found = 0
for i, word in enumerate(input_lang.word2index):
    try: 
        weights_matrix[i] = glove_word2vec[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(300, ))

```

### The Embedding Layer

These embeddings are used as the first layer in our EncoderRNN. Also, since these are 'trained', we need to make sure that backpropagation doesn't affect these embeddings. So we make them non-trainable by setting the `requires_grad` argument of their parameters to `False`

```python
self.embedding = nn.Embedding(input_size, hidden_size)
self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
for param in self.embedding.parameters():
        param.requires_grad = False
```

## Results

### English-French

#### No Embeddings

#### English Embeddings

#### English + French Embeddings 



### French-English

#### No Embeddings

#### English Embeddings

#### English + French Embeddings 
