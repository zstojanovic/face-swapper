name := "face-swapper"
version := "0.1"
scalaVersion := "2.12.11"

Compile / run / fork := true

libraryDependencies += "org.bytedeco" % "javacv-platform" % "1.5.2"
