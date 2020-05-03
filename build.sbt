// phuonglh, May 3, 2020
// 
ThisBuild / scalaVersion := "2.12.11"
ThisBuild / name := "vlp"
ThisBuild / organization := "phuonglh.com"
ThisBuild / version := "1.0"
val sparkVersion = "2.4.5"

ThisBuild / libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
)

// root project

lazy val root = (project in file("."))
  .aggregate(tok, tag, tdp)

// tokenizer module
lazy val tok = (project in file("tok"))
  .settings(
    mainClass in assembly := Some("vlp.tok.Tokenizer"),
    assemblyJarName in assembly := "tok.jar"
  )

// part-of-speech tagger module
lazy val tag = (project in file("tag"))
  .dependsOn(tok)
  .settings(
    assemblyJarName in assembly := "tag.jar"
  )

// transition-based dependency parser module
lazy val tdp = (project in file("tdp"))
  .dependsOn(tag)
  .settings(
    assemblyJarName in assembly := "tdp.jar"
  )

// named entity recognizer module
lazy val ner = (project in file("ner"))
  .dependsOn(tag)
  .settings(
    assemblyJarName in assembly := "ner.jar"
  )
