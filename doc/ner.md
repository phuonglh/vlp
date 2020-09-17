# Named Entity Recognition for LAD

## Data Preparation
- The resulting file is in `dat/ner/lad.tsv`. This file is in the 2-col format.
- First, run the `rename` method of `Utils.scala` to rename folders/files to tsv files, which are stored in `dat/ner/stm`
- Then, run the `convert` method of `Utils.scala` to convert 4-col format files in `dat/ner/stm` into 2-col format files, which are stored in `dat/ner/lad`. This conversion use BIO format for entities.
- Finally, run the `CorpusReader.scala` file to combine and filter all labeled sentences in `dat/ner/lad` into a 2-col format file `dat/ner/lad.tsv`.

## Training a Bidirectional CMM

- Train the forward model:
    `bloop run -m vlp.ner.Tagger ner -- -m train -d dat/ner/lad.tsv -p /opt/models/ner/lad/ -j`
- Train the backward model:
    `bloop run -m vlp.ner.Tagger ner -- -m train -d dat/ner/lad.tsv -p /opt/models/ner/lad/ -j -r`

## Evaluating the Model

  `bloop run -m vlp.ner.Tagger ner -- -m eval -d dat/ner/lad.tsv -p /opt/models/ner/lad/ -j`

  This will create the file `dat/ner/lad.tsv.out`. Then, one can use the `conlleval` script to evaluate the tagging result:

  `conlleval < lad.tsv.out` 

```
processed 202819 tokens with 8286 phrases; found: 8552 phrases; correct: 7527.
accuracy:  96.21%; precision:  88.01%; recall:  90.84%; FB1:  89.40
             DATE: precision:  91.64%; recall:  93.18%; FB1:  92.40  2190
              DOC: precision:  88.59%; recall:  86.81%; FB1:  87.69  1323
              LOC: precision:  72.29%; recall:  82.38%; FB1:  77.01  841
              ORG: precision:  88.69%; recall:  92.38%; FB1:  90.50  3882
              PER: precision:  93.99%; recall:  93.69%; FB1:  93.84  316
```