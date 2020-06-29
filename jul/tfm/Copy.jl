using Flux
using Transformers
using Transformers.Basic
using Random

Random.seed!(123)

labels = collect(1:10)
(bos, eos) = (11, 12)
unk = 0

labels = [unk, bos, eos, labels...]
vocab = Vocabulary(labels, unk)

# a function to create a sample data of a pair (input, output) which are the same random sequence
sample_data() = (d = rand(1:10, 10); (d, d))
# a function to pad a given sequence
pad(x) = [bos, x..., eos]

sample = pad.(sample_data())
encoded_sample = vocab(sample[1])

# define word embedding and position embedding
wordEmbedding = Embed(512, length(vocab))
positionEmbedding = PositionEmbedding(512)

# wrapper to get embedding
function embedding(x)
    we = wordEmbedding(x, inv(sqrt(512)))
    ps = positionEmbedding(we)
    we .+ ps
end

#define 2 layers of encoder
encode_t1 = Transformer(512, 8, 64, 2048)
encode_t2 = Transformer(512, 8, 64, 2048)

#define 2 layers of decoder
decode_t1 = TransformerDecoder(512, 8, 64, 2048)
decode_t2 = TransformerDecoder(512, 8, 64, 2048)

#define a linear layer to get the final output probabilities
linear = Positionwise(Dense(512, length(vocab)), logsoftmax)

function encoder_forward(x)
    e = embedding(x)
    t1 = encode_t1(e)
    t2 = encode_t2(t1)
    t2
end
  
function decoder_forward(x, m)
    e = embedding(x)
    t1 = decode_t1(e, m)
    t2 = decode_t2(t1, m)
    linear(t2)
end

# run the model on the sample
enc = encoder_forward(encoded_sample)
probs = decoder_forward(encoded_sample, enc)

