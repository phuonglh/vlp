import com.typesafe.sbt.packager.archetypes.JavaAppPackaging

enablePlugins(JavaServerAppPackaging)
enablePlugins(JavaAppPackaging)

name := "llm"
scalaVersion := "2.12.15"
val scalaTestVersion = "3.2.14"

javacOptions ++= Seq("-source", "11", "-target", "11")

licenses := Seq("Apache-2.0" -> url("https://opensource.org/licenses/Apache-2.0"))

ThisBuild / developers := List(
  Developer(
    id = "phuonglh",
    name = "phuonglh",
    email = "phuonglh@gmail.com",
    url = url("https://github.com/phuonglh")))

libraryDependencies ++= {
  Seq(
    "org.json4s" %% "json4s-native" % "3.5.3", 
    "org.json4s" %% "json4s-jackson" % "3.5.3",
    "commons-io" % "commons-io" % "2.5",
    "org.apache.kafka" % "kafka-clients" % "2.6.0",
    "de.l3s.boilerpipe" % "boilerpipe" % "1.1.0",
    "xerces" % "xercesImpl" % "2.11.0",
    "net.sourceforge.nekohtml" % "nekohtml" % "1.9.22",
    "org.apache.spark" % "spark-core_2.12" % "3.4.0",
    "org.apache.spark" % "spark-sql_2.12" % "3.4.0",
    "org.apache.spark" % "spark-mllib_2.12" % "3.4.0",
    "com.github.scopt" %% "scopt" % "3.7.1"
  )
}

/** Disables tests in assembly */
assembly / test := {}

assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x if x.startsWith("NativeLibrary") => MergeStrategy.last
  case x if x.startsWith("aws") => MergeStrategy.last
  case _ => MergeStrategy.last
}

assembly / assemblyJarName := "llm.jar"
assembly / mainClass := Some("vlp.llm.OSCAR")