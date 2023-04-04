import com.typesafe.sbt.packager.archetypes.JavaAppPackaging

enablePlugins(JavaServerAppPackaging)
enablePlugins(JavaAppPackaging)

val scalaTestVersion = "3.2.14"

name := "jsl"

version := "4.3.2"

scalaVersion := "2.12.15"

javacOptions ++= Seq("-source", "1.8", "-target", "1.8")

licenses := Seq("Apache-2.0" -> url("https://opensource.org/licenses/Apache-2.0"))

ThisBuild / developers := List(
  Developer(
    id = "phuonglh",
    name = "phuonglh",
    email = "phuonglh@gmail.com",
    url = url("https://github.com/phuonglh")))

val sparkVer = "3.3.1"
val sparkNLP = "4.3.2"

libraryDependencies ++= {
  Seq(
    "org.apache.spark" %% "spark-core" % sparkVer % Provided,
    "org.apache.spark" %% "spark-mllib" % sparkVer % Provided,
    "org.scalatest" %% "scalatest" % scalaTestVersion % "test",
    "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNLP,
    "org.apache.kafka" % "kafka-clients" % "2.6.0",
    "de.l3s.boilerpipe" % "boilerpipe" % "1.1.0",
    "xerces" % "xercesImpl" % "2.11.0",
    "net.sourceforge.nekohtml" % "nekohtml" % "1.9.22" % "provided",
    "org.glassfish" % "javax.json" % "1.1.4",
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

/*
 * If you wish to make a Uber JAR (Fat JAR) without Spark NLP
 * because your environment already has Spark NLP included same as Apache Spark
**/
//assemblyExcludedJars in assembly := {
//  val cp = (fullClasspath in assembly).value
//  cp filter {
//    j => {
//        j.data.getName.startsWith("spark-nlp")
//    }
//  }
//}
