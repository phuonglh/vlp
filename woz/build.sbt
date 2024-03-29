// phuonglh, February 14, 2023
val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"
val sparkNLPVersion = "4.3.2"

scalaVersion := "2.12.15"

javacOptions ++= Seq("-source", "1.8", "-target", "1.8")
javacOptions ++= Seq("-encoding", "UTF-8")
scalacOptions ++= Seq("-Xfatal-warnings", "-deprecation", "-feature", "-unchecked",
    "-language:implicitConversions", "-language:higherKinds", "-language:existentials", "-language:postfixOps"
)

lazy val commonSettings = Seq(
  name := "woz",
  organization := "phuonglh.com",
  version := "1.0.0",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
    "com.google.protobuf" % "protobuf-java" % "3.21.9",
    "com.github.scopt" %% "scopt" % "3.7.1",
    "org.scala-lang" % "scala-reflect" % "2.12.15",
    "org.json4s" %% "json4s-jackson" % "3.5.3", // use 3.5.3 to fix a bug of JSON
    "com.databricks" %% "spark-xml" % "0.16.0", // to read XML files into Spark
    "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion,
    "org.scalatest" %% "scalatest" % "3.1.1" % "test"
  )
)

lazy val woz = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.woz.Classifier"),
      assembly / assemblyJarName := "woz.jar",
      libraryDependencies ++= Seq(),
   run / fork := true
  )

  /** Disables tests in assembly */
assembly / test := {}

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x if x.startsWith("NativeLibrary") => MergeStrategy.last
  case x if x.startsWith("aws") => MergeStrategy.last
  case _ => MergeStrategy.last
}
