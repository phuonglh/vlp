ThisBuild / version := "1.0.0"

ThisBuild / scalaVersion := "2.13.12"

resolvers += "Unidata" at "https://artifacts.unidata.ucar.edu/repository/unidata-all/"

lazy val root = (project in file("."))
  .settings(
    name := "s2s",
    libraryDependencies ++= Seq(
      "edu.ucar" % "cdm-core" % "5.5.2",
    )
  )
