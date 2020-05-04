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

The tokenizer module is bundled in the file `tok.jar`. See section `Compile and Package` below to know 
how to create this jar file from source.

The main class of the tokenizer module is `vlp.tok.Tokenizer`. It segments a given text into tokens. Each 
token is represented by a triple (`position`, `shape`, `content`). This class can take two arguments of an input text file and an output file. The input file must exist and contain plain text, arranged in lines. The 
output file will be created by the program. For example:

  `$java -jar tok.jar path/to/inp.txt path/to/out.txt`

If the output file is not provided, the result will be shown to the console. If both the input and output files are not provided, a sample sentence will be processed and its result is shown to the console.  

The tokenizer makes use of parallel processing which exploits **all CPU cores** of a single machine. For this reason, on large file it is still fast. On my laptop, the tokenizer can process an input file of more than 532,000 sentences (about 1,000,000 syllables) in about 100 seconds.

For really large input files in a big data setting, it is more convenient to use the tokenizer together with the [Apache Spark](http://spark.apache.org) library so that it is easily to port to a cluster of multiple nodes. We provide a transformer-based implementation of the Vietnamese tokenizer, in the class `vlp.tok.TokenizerTransformer`. This can be integrated into the machine learning pipeline of the Apache Spark Machine Learning library, in the same way as the standard `org.apache.spark.ml.feature.Tokenizer`. Note that the wrapper transformer depends on Apache Spark but not the tokenizer. If you do not want to use Apache Spark, you can simply copy the self-contained tokenizer and import it to your project, ignoring all Apache Spark dependencies.

## Part-of-Speech Tagger

The tagger module implements a simple first-order conditional Markov model for sequence tagging. The basic features include current word, previous word, next word, current word shape, next word shape, previous previous word, and next next word. Each local transition probability is specified by a multinomial logistic regression model.

On the standard VLSP 2010 part-of-speech tagged treebank, this simple model gives a training accuracy is 0.9638 when all the corpus is used for training. A pre-trained model is provided in the directory `dat/tag`.

Since the machine learning pipeline in use is that of Apache Spark, this module depends on Apache Spark. Suppose that you have alreadly a version of Apache Spark installed (say, version 2.4.5). 

### Tagging Mode

To tag an input text file containing sentences, each on a line, invoke the command

  `$spark-submit tag/target/scala-2.11/tag.jar -m tag -i dat/sample.txt`

Option `-m` specifies the running mode. If the pre-trained model is not on the default path, you must specify it explicitly with option `-p`, as follows:

  `$spark-submit tag/target/scala-2.11/tag.jar -m tag -i dat/sample.txt -p path/to/model`

Note that the input text file does not need to be tokenized in advanced, the tagger will call the tokenizer module to segment the text into words before tagging.

### Training Mode

To train a model, you will need the VLSP 2010 part-of-speech tagged corpus (which has about 10,000 manually tagged sentences). Suppose that the corpus is provided at the default path `dat/vtb-tagged.txt`:

  `$spark-submit tag/target/scala-2.11/tag.jar -m train`

If the data is at another location, specify it with option `-d`:

  `$spark-submit tag/target/scala-2.11/tag.jar -m train -d path/to/tagged/corpus`

The resulting model will be saved to its default directory `dat/tag`. This can be changed with option `-p` as above. After training, the evaluation mode is called automatically to print out training performance scores.

There are some option for fine-tuning the training, such as `-f` (for min feature frequency cutoff, default value is 3) or `-u` (for domain dimension, default value is 16,384). See the code for detail.

By default, the master URL is set to `local[*]`, which means that all CPU cores of the current machine are used by Apache Spark. You can specify a custom master URL with option `-M`.

## Compile and Package

### Some Notes
- Most of the modules depends on the Machine Learning library of Apache Spark.
- This big data technology permits to process millions of texts with a very high speed.
- The services can be used in two modes: batch processing (offline) or on-the-fly (online).
- The program is developed in the Scala programming language. It needs a Java Runtime Environment (JRE) 
  to run, or a Java Development Kit (JDK) environment to compile and package. We use Java version 8.0.
- Since the code is developed in Scala, you need to have Scala too.
- If you want to compile and build the software from source, you need a Scala build tool 
  to manage all dependencies and produce a binary JAR file. We use [SBT](https://www.scala-sbt.org/download.html).

### Compile and Package
- Go to the main directory `cd vlp` on your command line console.
- Invoke `sbt` console with the command `sbt`.
- In the sbt, compile the entire project with the command `compile`. All requried libraries are automatically downloaded, only at the first time.
- In the sbt, package the project with the command `assembly`.
- The resulting JAR files are in sub-projects, for example 
  * tokenizer is in `tok/target/scala-2.11/tok.jar`
  * part-of-speech tagger is in `tag/target/scala-2.11/tag.jar`
  * etc

## Contact

Any bug reports, suggestions and collaborations are welcome. I am
reachable at: 
* LE-HONG Phuong, http://mim.hus.edu.vn/lhp/
* College of Science, Vietnam National University, Hanoi