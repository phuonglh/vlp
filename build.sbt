// phuonglh, May 3, 2020
// 
ThisBuild / scalaVersion := "2.12.11"
ThisBuild / name := "vlp"
ThisBuild / organization := "phuonglh.com"
ThisBuild / version := "1.0"

ThisBuild / libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.5",
  "org.apache.spark" %% "spark-sql" % "2.4.5",
  "org.apache.spark" %% "spark-mllib" % "2.4.5"
)

// root project

lazy val root = (project in file("."))
  .aggregate(tok, tag, tdp)

// tokenizer module
lazy val tok = (project in file("tok"))

// tagger module
lazy val tag = (project in file("tag"))
  .dependsOn(tok)

// transition-based dependency parser module
lazy val tdp = (project in file("tdp"))
  .dependsOn(tag)

