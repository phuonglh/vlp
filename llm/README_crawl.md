## 1. Introduction 

This project contains code for crawling news and extracting their main text contents. The raw texts 
are saved to a local JSON file and optionally sent to a Kafka server. 

## 2. Usage

### 2.1. Run from Binary

You need a Java Runtime Environemtn (JRE) version 1.8+ to run the code. The project is packaged into a executable JAR file `bin/llm.jar`. You can simple change to the root directory of the project and run in a CLI:

`java -jar bin/llm.jar`

This command read news sources from `dat/sources.json`, crawl and extract contents. The resulting file is `dat/yyyymmdd.json`. 

### 2.2. Run from Source

The code is written in the Scala programming language, which requires a Java runtime library and Scala core language to run. Some library dependencies are specified in the `build.sbt` build file. 

The main entry class is `vlp.llm.NewsIndexer`. To compile/run the project from source, use the [SBT](https://www.scala-sbt.org) build tool. This tool reads the `build.sbt` configuration file, download and install required libraries automatically. 

If your system has [Bloop](https://scalacenter.github.io/bloop/), you can quickly install the project from a CLI as follows:

- Change to the root directory of the project (which contains `build.sbt`)
- Run `sbt bloopInstall` to install required libs. This command only need to execute once.

Once the project is compiled, we can run the main file to extract news using

`bloop run -p llm -m vlp.llm.NewsIndexer` 

As above, this command read news sources from `dat/sources.json`, crawl and extract contents. The resulting file is `dat/yyyymmdd.json`. 

## 3. Contact
- Developer: Phuong Le-Hong
- Email: phuonglh@gmail.com
- Website: [http://vlp.group/lhp/](http://vlp.group/lhp/)
