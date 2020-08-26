# Neural Tagger

## GRU Models

- Unidirectional GRU
- Bi-directional GRU

## Features
- Word identity => Word embedding (learned params)
- Word shape => Shape embedding of 10 dimensions (learned params)
- Word mentions => Mention embedding (binary-valued, learned parameters)

## Computation Graph

[w1, w2,..., w_n] => Embedding(100) \ 
[s1, s2,..., s_n] => Embedding(10) -- Merge(concat) => GRU/BiGRU => [Dense(relu)]? => Dense(numLabels, softmax)
[m1, m2,..., m_n] => Embedding(2)   / 

m_j is a binary value (0 or 1) which encodes whether the j-th word in the sentence appears in an in-name token set which is extracted from the training data.