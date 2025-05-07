name := """fahmi"""
organization := "com.fahmi"

version := "0.1"

lazy val root = (project in file("."))

scalaVersion := "2.12.4"

libraryDependencies += "com.github.tototoshi" %% "scala-csv" % "1.3.5"

unmanagedJars in Compile += file("libs/InaNLP.jar")
unmanagedJars in Compile += file("libs/ipostagger.jar")
