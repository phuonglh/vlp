// phuonglh, February 14, 2023
val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"

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
    "org.slf4j" % "slf4j-simple" % "1.7.36",
     "org.json4s" %% "json4s-native" % "3.4.0",
    "org.scalatest" %% "scalatest" % "3.1.1" % "test"
  )
)

lazy val root = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.woz.DomainFilter"),
      assembly / assemblyJarName := "woz.jar",
      libraryDependencies ++= Seq(),
   run / fork := true
  )
