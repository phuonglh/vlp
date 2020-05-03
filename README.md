# vlp

A Vietnamese text processing library developed in the Scala programming language.

## Introduction

This is a repository of a Scala project which implements some basic tasks of Vietnamese text processing.
Each basic task is implemented in a module. 
- `tok`: tokenizer, which implements a rule-based word segmentation approach;
- `tag`: tagger, which implements a conditional Markov model for sequence tagging;
- `tdp`: dependency parser, which implements a transition-based dependency parsing approach;
- `ner`: named entity recognizer, which implements a bidirectional conditional Markov model for sequence tagging;

## Tokenizer

The tokenizer module is bundled in the file `tok.jar`. See section `Compile and Assembly` below to know 
how to create this jar file from source.

The main class of the tokenizer module is `vlp.tok.Tokenizer`. It segments a given text into tokens. Each 
token is represented by a triple (`position`, `shape`, `content`). This class can take two arguments of an input text file and an output file. The input file must exist and contain plain text, arranged in lines. The 
output file will be created by the program. For example:

  `$java -jar tok.jar path/to/inp.txt path/to/out.txt`

If the output file is not provided, the result will be shown to the console. If both the input and output files are not provided, a sample sentence will be processed and its result is shown to the console.  

The tokenizer makes use of parallel processing which exploits **all CPU cores** of a single machine. For this reason, on large file it is still fast. On my laptop, the tokenizer can process an input file of more than 532,000 sentences (about 1,000,000 syllables) in about 100 seconds.

For really large input files in a big data setting, it is more convenient to use the tokenizer together with the [Apache Spark](http://spark.apache.org) library. We provide a transformer-based implementation of the Vietnamese tokenizer, in the class `vlp.tok.TokenizerTransformer`. This can be integrated into the machine learning pipeline of the Apache Spark Machine Learning library, in the same way as the standard `org.apache.spark.ml.feature.Tokenizer`. Note that the wrapper transformer depends on Apache Spark but not the tokenizer. If you do not want to use Apache Spark, you can simply copy the self-contained tokenizer and import it to your project, ignoring all Apache Spark dependencies.

## Compile and Package

### Some Notes
- Most of the modules depends on the Machine Learning library of Apache Spark.
- This big data technology permits to process millions of texts with a very high speed.
- The services can be used in two modes: batch processing (offline) or on-the-fly (online).
- The program is developed in the Scala programming language. It needs a Java Runtime Environment (JRE) 
  to run, or a Java Development Kit (JDK) environment to compile and package. We use Java version 8.0.
- Since the code is developed in Scala, you need to have Scala too (version 2.12).
- If you want to compile and build the software from source, you need a Scala build tool 
  to manage all dependencies and produce a binary JAR file. We use [SBT](https://www.scala-sbt.org/download.html)

### Compile
- Go to the main directory `cd vlp` on your command line console.
- Invoke `sbt` console with the command `sbt`.
- In the sbt, compile the entire project with the command `compile`. All requried libraries are automatically downloaded, only at the first time.
- In the sbt, package the project with the command `assembly`.
- The resulting JAR files are in sub-projects, for example 
  * tokenizer is in `tok/target/scala-2.12/tok.jar`
  * part-of-speech tagger is in `tag/target/scala-2.12/tag.jar`

## Contact

Any bug reports, suggestions and collaborations are welcome. I am
reachable at: 
* LE-HONG Phuong, http://mim.hus.edu.vn/lhp/
* College of Science, Vietnam National University, Hanoi