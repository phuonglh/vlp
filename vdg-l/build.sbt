// phuonglh, November 11, 2022
val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"

javacOptions ++= Seq("-encoding", "UTF-8")
scalacOptions ++= Seq("-Xfatal-warnings", "-deprecation", "-feature", "-unchecked",
    "-language:implicitConversions", "-language:higherKinds", "-language:existentials", "-language:postfixOps"
)

lazy val commonSettings = Seq(
  name := "vdg",
  organization := "phuonglh.com",
  version := "1.0.0",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
     "com.google.protobuf" % "protobuf-java" % "3.21.9",
    "com.github.scopt" %% "scopt" % "3.7.1"
  )
)

lazy val vdg = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.vdg.VDG"),
      assembly / assemblyJarName := "vdgl.jar",
      libraryDependencies ++= Seq(),
   run / fork := true
  )
