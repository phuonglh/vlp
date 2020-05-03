# vlp

A Vietnamese text processing library developed in the Scala programming language.

## Introduction

This is a repository of a Scala project which implements some basic tasks of Vietnamese text processing.
Each basic task is implemented in a module. 
* `tok`: tokenizer, which implements a rule-based word segmentation approach;
* `tag`: tagger, which implements a conditional Markov model for sequence tagging;
* `tdp`: dependency parser, which implements a transition-based dependency parsing approach;

## Tokenizer

The main class of the tokenizer module is `vlp.tok.Tokenizer`. It segments a given text into tokens. Each 
token is represented by a triple (`position`, `shape`, `content`). This class can take two arguments of an input text file and an output file. The input file must exist and contain plain text, arranged in lines. The 
output file will be created by the program. For example:

  `$java vlp.tok.Tokenizer path/to/inp.txt path/to/out.txt`

If the output file is not provided, the result will be shown to the console. If both the input and output files are not provided, a sample sentence will be processed and its result is shown to the console.  

The tokenizer makes use of parallel processing which exploits **all CPU cores** of a single machine. For this reason, on large file, it is still fast. On my laptop, the tokenizer can process an input file of more than 532,000 sentences (about 1,000,000 syllables) in about 100 seconds.

For really big input files in a big data setting, it is more convenient to use it with the Apache Spark library. We provide a transformer-based implementation of the Vietnamese tokenizer, in the class `vlp.tok.TokenizerTransformer`. This can be integrated into the machine learning pipeline of the Apache Spark Machine Learning library, in the same way as the standard `org.apache.spark.ml.feature.Tokenizer`.

## Contact

Any bug reports, suggestions and collaborations are welcome. I am
reachable at: 
* LE-HONG Phuong, http://mim.hus.edu.vn/lhp/
* College of Science, Vietnam National University, Hanoi