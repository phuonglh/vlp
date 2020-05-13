// phuonglh, May 3, 2020
// 
val sparkVersion = "2.4.5"

lazy val commonSettings = Seq(
  scalaVersion := "2.11.12",
  name := "vlp",
  organization := "phuonglh.com",
  version := "1.0",
  libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
  )
)

// root project

lazy val root = (project in file("."))
  .aggregate(tok, tag, tdp)

// tokenizer module
lazy val tok = (project in file("tok"))
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tok.Tokenizer"),
    assemblyJarName in assembly := "tok.jar"
  )

// part-of-speech tagger module
lazy val tag = (project in file("tag"))
  .dependsOn(tok)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tag.Tagger"),
    assemblyJarName in assembly := "tag.jar",
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "3.7.1"
    )
  )

// transition-based dependency parser module
lazy val tdp = (project in file("tdp"))
  .dependsOn(tag)
  .settings(
    commonSettings,
    mainClass in assembly := Some("vlp.tdp.Parser"),
    assemblyJarName in assembly := "tdp.jar",
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "3.7.1",
      "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
    )
  )

// named entity recognizer module
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
    assemblyJarName in assembly := "tpm.jar",
    libraryDependencies ++= Seq(
      "com.github.scopt" %% "scopt" % "3.7.1"
    )
  )