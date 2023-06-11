ThisBuild / version := "1.0"

ThisBuild / scalaVersion := "2.12.12"

val sparkVersion = "3.1.2"
val bigdlVersion = "2.1.0"

lazy val root = (project in file("."))
  .settings(
    name := "vcb",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "com.intel.analytics.bigdl" % "bigdl-dllib-spark_3.1.2" % bigdlVersion,
      "com.google.protobuf" % "protobuf-java" % "3.21.9",
    )
  )
