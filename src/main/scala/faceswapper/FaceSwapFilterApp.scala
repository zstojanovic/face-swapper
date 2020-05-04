package faceswapper

import org.bytedeco.opencv.global.opencv_core._
import org.bytedeco.opencv.global.opencv_highgui._
import org.bytedeco.opencv.opencv_core.Mat

object FaceSwapFilterApp extends App {
  val sourceA = new VideoFileSource(args(0))
  val sourceB = new VideoFileSource(args(1))
  val detectorA = new FaceDetectorAndTracker(sourceA)
  val detectorB = new FaceDetectorAndTracker(sourceB)
  val faceSwapper = new FaceSwapper()

  val frameA = new Mat()
  val frameB = new Mat()
  var fps = 0d
  var time_start = getTickCount
  while (waitKey(1) != 27 && detectorA.process(frameA) && detectorB.process(frameB)) {
    val facesA = detectorA.getFaces
    val facesB = detectorB.getFaces
    if (facesA.size() >= 1 && facesB.size() >= 1) {
      faceSwapper.swapFaces(frameA, facesA.get(0), frameB, facesB.get(0))
    }

    val time_end = getTickCount
    val time_per_frame = (time_end - time_start) / getTickFrequency

    fps = (15 * fps + (1 / time_per_frame)) / 16
    printf("Total time: %3.5f | FPS: %3.2f\n", time_per_frame, fps)
    imshow("face-swap", frameB)
    time_start = getTickCount
  }
}
