// phuonglh, May 3, 2020
// 
val sparkVersion = "2.4.8"
val jobServerVersion = "0.11.1"

javacOptions ++= Seq("-encoding", "UTF-8", "-XDignore.symbol.file", "true")

scalacOptions ++= Seq(
    "-Xfatal-warnings",
    "-deprecation",
    "-feature",
    "-unchecked",
    "-language:implicitConversions",
    "-language:higherKinds",
    "-language:existentials",
    "-language:postfixOps"
)

lazy val commonSettings = Seq(
  scalaVersion := "2.11.12", // upgrade from 2.11.12 to 2.12.12
  name := "vlp",
  organization := "phuonglh.com",
  version := "1.0",
  libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "com.github.scopt" %% "scopt" % "3.7.1",
  ),
  run in Compile := Defaults.runTask(fullClasspath in Compile, mainClass in (Compile, run), runner in (Compile, run)).evaluated,
  runMain in Compile := Defaults.runMainTask(fullClasspath in Compile, runner in(Compile, run)).evaluated  
)
// root project
lazy val root = (project in file("."))
  .aggregate(tok, tag, tdp, ner, tpm, tcl, idx, vdr, vdg, vec, zoo, biz, nli, sjs, qas)

// tokenization module
lazy val tok = (project in file("tok"))
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tok.Tokenizer"),
    assemblyJarName in assembly := "tok.jar"
  )

// part-of-speech tagging module
lazy val tag = (project in file("tag"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tag.Tagger"),
    assemblyJarName in assembly := "tag.jar"
  )

// transition-based dependency parsing module
lazy val tdp = (project in file("tdp"))
  .dependsOn(tag)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tdp.Parser"),
    assemblyJarName in assembly := "tdp.jar",
    libraryDependencies ++= Seq(
      "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
    )
  )

// named entity recognization module
lazy val ner = (project in file("ner"))
  .dependsOn(tag)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.ner.Tagger"),
    assemblyJarName in assembly := "ner.jar",
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "3.7.1",
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.2-spark_2.4.3" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0" % "provided"
    )
  )

  // topic modeling module
lazy val tpm = (project in file("tpm"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tpm.LDA"),
    assemblyJarName in assembly := "tpm.jar"
  )

  // text classification module
lazy val tcl = (project in file("tcl"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tcl.Classifier"),
    assemblyJarName in assembly := "tcl.jar",
    libraryDependencies ++= Seq(
    )
  )

// text indexing module
lazy val idx = (project in file("idx"))
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.idx.NewsIndexer"),
    assemblyJarName in assembly := "idx.jar",
    libraryDependencies ++= Seq(
      "mysql" % "mysql-connector-java" % "8.0.16",
      "org.elasticsearch.client" % "elasticsearch-rest-high-level-client" % "7.1.1",
      "de.l3s.boilerpipe" % "boilerpipe" % "1.1.0",
      "xerces" % "xercesImpl" % "2.11.0",
      "net.sourceforge.nekohtml" % "nekohtml" % "1.9.22" % "provided",
      "org.glassfish" % "javax.json" % "1.1.4",
      "org.apache.kafka" % "kafka-clients" % "2.6.0"
    ),
    run / fork := true,
    run / javaOptions ++= Seq("-Xmx8g", "-Djdk.tls.trustNameService=true", "-Dcom.sun.jndi.ldap.object.disableEndpointIdentification=true")
  )

// Vietnamese diacritics restoration module (CMM-based approach)
lazy val vdr = (project in file("vdr")) 
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.vdr.Restorer"),
    assemblyJarName in assembly := "vdr.jar",
    libraryDependencies ++= Seq()
  )

// Vietnamese diacritics generation module (RNN-based approaches)
lazy val vdg = (project in file("vdg"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.vdg.Generator"),
    assemblyJarName in assembly := "vdg.jar",
    resolvers += Resolver.mavenLocal,
    libraryDependencies ++= Seq(
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.2-spark_2.4.3" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0" % "provided",
      "com.intel.analytics.zoo" % "zoo-core-mkl-linux" % "0.8.1" % "provided",
      "com.intel.analytics.zoo" % "zoo-core-torchnet-java-mac" % "0.8.1" % "provided"
    )
  )

// Natural language inference module
lazy val nli = (project in file("nli"))
  .dependsOn(tag)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.nli.Teller"),
    assemblyJarName in assembly := "nli.jar",
    resolvers += Resolver.mavenLocal,
    libraryDependencies ++= Seq(
      "org.scalaj" %% "scalaj-http" % "2.4.2",
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.2-spark_2.4.3" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0" % "provided",
      "com.intel.analytics.zoo" % "zoo-core-mkl-linux" % "0.8.1" % "provided",
      "com.intel.analytics.zoo" % "zoo-core-torchnet-java-mac" % "0.8.1" % "provided"
    )
  )

// Word to vector module
lazy val vec = (project in file("vec"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.vec.W2V"),
    assemblyJarName in assembly := "vec.jar",
    resolvers += Resolver.mavenLocal
  )

// Models that depend on the Analytic Zoo framework
lazy val zoo = (project in file("zoo"))
  .dependsOn(tok, biz)
  .settings(
    commonSettings,
    assemblyJarName in assembly := "zoo.jar",
    resolvers += Resolver.mavenLocal,
    libraryDependencies ++= Seq(
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.2-spark_2.4.3" % "0.10.0" % "provided",
      "com.intel.analytics.zoo" % "zoo-core-dist-all" % "0.10.0" % "provided"
    ),
    assemblyMergeStrategy in assembly := {
      case x if x.contains("log4j.properties") => MergeStrategy.first
      case x =>
        val oldStrategy = (assemblyMergeStrategy in assembly).value
        oldStrategy(x)
    }
  )

// Analytic Zoo (for assembly as a uber jar to be used as a dependency)
lazy val biz = (project in file("biz"))
  .settings(
    commonSettings,
    assemblyJarName in assembly := "biz.jar",
    resolvers += Resolver.mavenLocal,
    libraryDependencies ++= Seq(
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.12.2-spark_2.4.3" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0" % "provided"
    ),
    assemblyMergeStrategy in assembly := {
      case x if x.contains("com/intel/analytics/bigdl/bigquant/") => MergeStrategy.first
      case x if x.contains("com/intel/analytics/bigdl/mkl/") => MergeStrategy.first
      case x if x.contains("org/tensorflow/") => MergeStrategy.first
      case x if x.contains("google/protobuf") => MergeStrategy.first
      case x if x.contains("org/apache/spark/unused") => MergeStrategy.first
      case x =>
        val oldStrategy = (assemblyMergeStrategy in assembly).value
        oldStrategy(x)
    }
  )

// Spark-JobServer module
lazy val sjs = (project in file("sjs"))
  .dependsOn(tdp, ner)
  .settings(
    commonSettings,
    assemblyJarName in assembly := "sjs.jar",
    resolvers ++= Seq("Artifactory" at "https://sparkjobserver.jfrog.io/artifactory/jobserver/"),
    libraryDependencies ++= Seq(
      "spark.jobserver" % "job-server-api_2.11" % jobServerVersion % "provided",
      "spark.jobserver" % "job-server-extras_2.11" % jobServerVersion % "provided"
    )
  )

// Text Mining Insights for Vietnam's Post-Pandemic Green Recovery module
lazy val tmi = (project in file("tmi"))
 .dependsOn(tpm)
 .settings(
   commonSettings,
   mainClass in assembly := Some("vlp.tmi.NewsIndexer"),
   assemblyJarName in assembly := "tmi.jar",
   libraryDependencies ++= Seq(
     "de.l3s.boilerpipe" % "boilerpipe" % "1.1.0",
     "xerces" % "xercesImpl" % "2.11.0",
     "net.sourceforge.nekohtml" % "nekohtml" % "1.9.22" % "provided",
     "org.glassfish" % "javax.json" % "1.1.4",
     "org.apache.kafka" % "kafka-clients" % "2.6.0",
     "org.scalaj" %% "scalaj-http" % "2.4.2",
     "org.twitter4j" % "twitter4j-core" % "4.0.6",
     "org.twitter4j" % "twitter4j-stream" % "4.0.6",
     "org.apache.bahir" %% "spark-streaming-twitter" % "2.4.0" // depends on twitter4j-* version 4.0.6
   ),
   run / fork := true,
   run / javaOptions ++= Seq("-Xmx8g", "-Djdk.tls.trustNameService=true", "-Dcom.sun.jndi.ldap.object.disableEndpointIdentification=true")
 )

// QAS module
lazy val qas = (project in file("qas"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.qas.Indexer"),
    assemblyJarName in assembly := "qas.jar",
    libraryDependencies ++= Seq(
      "org.elasticsearch.client" % "elasticsearch-rest-high-level-client" % "7.1.1",
      "org.json4s" %% "json4s-native" % "3.5.3", // should use version 3.5.3 to fix a bug
      "org.json4s" %% "json4s-jackson" % "3.5.3",      
      "org.scalaj" %% "scalaj-http" % "2.4.2",
      "com.typesafe.akka" %% "akka-actor" % "2.5.27",
      "com.typesafe.akka" %% "akka-http" % "10.1.11",
      "com.typesafe.akka" %% "akka-stream" % "2.5.27",
      "com.typesafe.akka" %% "akka-slf4j" % "2.5.27",
      "org.slf4j" % "slf4j-simple" % "1.7.16",
      "de.heikoseeberger" %% "akka-http-jackson" % "1.28.0"
    ),
    run / fork := true,
    run / javaOptions ++= Seq("-Xmx8g", "-Djdk.tls.trustNameService=true", "-Dcom.sun.jndi.ldap.object.disableEndpointIdentification=true")
  )
