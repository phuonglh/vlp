import sentencepiece as sp


def train():
    path = "dat/txt/100K.txt"

    lines = open(path, encoding="utf8").readlines()
    for i in range(10):
        print(lines[i])


    # train a BPE model 
    sp.SentencePieceTrainer.train('--model_type=bpe --input={} --model_prefix=./pyt/tok/bpe --vocab_size=10000'.format(path))

    # train a UNI model
    sp.SentencePieceTrainer.train('--model_type=unigram --input={} --model_prefix=./pyt/tok/uni --vocab_size=10000'.format(path))

def testBPE():
    bpe = sp.SentencePieceProcessor()
    bpe.load('bpe.model')
    print('BPE: {}'.format(bpe.encode_as_pieces('Tôi là Lê Hồng Phương')))


def testUNI():
    uni = sp.SentencePieceProcessor()
    uni.load('uni.model')
    print('UNI: {}'.format(uni.encode_as_pieces('Tôi là Lê Hồng Phương')))

train()

testBPE()
testUNI()