// phuonglh, May 3, 2020
// 
val sparkVersion = "2.4.5"

javacOptions ++= Seq("-encoding", "UTF-8")

lazy val commonSettings = Seq(
  scalaVersion := "2.11.12",
  name := "vlp",
  organization := "phuonglh.com",
  version := "1.0",
  libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "com.github.scopt" %% "scopt" % "3.7.1",
  )
)

// root project
lazy val root = (project in file("."))
  .aggregate(tok, tag, tdp)

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
      "com.github.scopt" %% "scopt" % "3.7.1"
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
      "org.glassfish" % "javax.json" % "1.1.4"
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
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.10.0-spark_2.4.3" % "0.8.1" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "0.10.0" % "provided",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0" % "provided"
    )
  )

// Word to vector module
lazy val vec = (project in file("vec"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.vec.W2V"),
    assemblyJarName in assembly := "vec.jar",
    resolvers += Resolver.mavenLocal,
    libraryDependencies ++= Seq(
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.10.0-spark_2.4.3" % "0.8.1",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "0.10.0",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0"
    ),
  )



// Analytic Zoo (for assembly only as a uber jar to be used as a dependency)
lazy val zoo = (project in file("zoo"))
  .settings(
    commonSettings,
    assemblyJarName in assembly := "zoo.jar",
    resolvers += Resolver.mavenLocal,
    libraryDependencies ++= Seq(
      "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.10.0-spark_2.4.3" % "0.8.1",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-mac" % "0.10.0",
      "com.intel.analytics.bigdl.core.native.mkl" % "mkl-java-x86_64-linux" % "0.10.0"
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

