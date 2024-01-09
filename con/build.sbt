val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"
val sparkNLPVersion = "4.3.2"


javacOptions ++= Seq("-encoding", "UTF-8")
scalacOptions ++= Seq("-Xfatal-warnings", "-deprecation", "-feature", "-unchecked",
    "-language:implicitConversions", "-language:higherKinds", "-language:existentials", "-language:postfixOps"
)

lazy val commonSettings = Seq(
  name := "con",
  organization := "phuonglh.com",
  version := "1.0.0",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
    "com.google.protobuf" % "protobuf-java" % "3.22.2",
    "com.github.scopt" %% "scopt" % "4.1.0",
    "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLPVersion
  )
)

lazy val con = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.con.MED"),
      assembly / assemblyJarName := "con.jar",
      libraryDependencies ++= Seq(),
   run / fork := true
  )
