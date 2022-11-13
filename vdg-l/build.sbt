// phuonglh, November 11, 2022
val sparkVersion = "3.2.0"
val bigdlVersion = "0.13.0"

javacOptions ++= Seq("-encoding", "UTF-8")
scalacOptions ++= Seq("-Xfatal-warnings", "-deprecation", "-feature", "-unchecked",
    "-language:implicitConversions", "-language:higherKinds", "-language:existentials", "-language:postfixOps"
)

lazy val commonSettings = Seq(
  name := "vdg-l",
  organization := "phuonglh.com",
  version := "1.0.0",
  libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
    // BigDL version 0.13.0 uses Spark 3.0.0 
    "com.intel.analytics.bigdl" % "bigdl-SPARK_3.0" % "0.13.0" % "provided", 
    // Analytics Zoo version 0.11.0 uses BigDL version 0.13.0
    "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.13.0-spark_3.0.0" % "0.11.0" % "provided",
    "com.intel.analytics.zoo" % "zoo-core-mkl-mac" % "0.11.0" % "provided",
    "com.intel.analytics.zoo" % "zoo-core-mkl-linux" % "0.11.0" % "provided",
    "com.github.scopt" %% "scopt" % "3.7.1"
  )
)

lazy val root = (project in file("."))
    .settings(commonSettings, 
      assembly / mainClass := Some("vlp.vdg.VDG"),
      assembly / assemblyJarName := "vdgl.jar",
      libraryDependencies ++= Seq(),
   run / fork := true
  )
