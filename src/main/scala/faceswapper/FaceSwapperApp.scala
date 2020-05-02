package faceswapper

import org.bytedeco.opencv.opencv_core.Mat
import org.bytedeco.opencv.global.opencv_core._
import org.bytedeco.opencv.global.opencv_highgui._

object FaceSwapperApp extends App {
  if (args.isEmpty) {
    println(
      """Usage:
        |  face-swapper [cameraIndex]
        |or
        |  face-swapper [videoFileName]
        |
        |No arguments specified, using cameraIndex = 0""".stripMargin)
  }
  val source = VideoSource.create(args.toSeq.headOption.getOrElse("0"))
  val detector = new FaceDetectorAndTracker(source)
  val faceSwapper = new FaceSwapper()

  val frame = new Mat()
  var fps = 0d
  var time_start = getTickCount
  while (waitKey(1) != 27 && detector.process(frame)) {
    val faces = detector.getFaces
    if (faces.size() == FaceCount) {
      faceSwapper.swapFaces(frame, faces.get(0), faces.get(1))
    }
    val time_end = getTickCount
    val time_per_frame = (time_end - time_start) / getTickFrequency

    fps = (15 * fps + (1 / time_per_frame)) / 16
    printf("Total time: %3.5f | FPS: %3.2f\n", time_per_frame, fps)
    imshow("face-swap", frame)
    time_start = getTickCount
  }
}
