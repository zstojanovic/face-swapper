package faceswapper

import org.bytedeco.opencv.opencv_core._
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier
import org.bytedeco.opencv.global.opencv_imgproc._

import scala.collection.mutable.ArrayBuffer

class FaceDetectorAndTracker(source: VideoSource) {
  private val CascadeFilename = "haarcascade_frontalface_default.xml"
  private val classifier = new CascadeClassifier(CascadeFilename)
  if (classifier.empty()) {
    println(s"Error loading cascade file $CascadeFilename")
    println("If it's missing, you can find it at https://github.com/opencv/opencv/tree/master/data/haarcascades")
    sys.exit(-1)
  }

  private val originalFrameSize = source.size()
  private val downscaledFrameWidth = 256
  private val downscaledFrameSize = new Size(downscaledFrameWidth,
    (downscaledFrameWidth * originalFrameSize.height) / originalFrameSize.width)
  private val ratio = originalFrameSize.width.toFloat / downscaledFrameSize.width

  private val downscaledFrame: Mat = new Mat()
  private val faces = new RectVector()

  def process(frame: Mat): Boolean = {
    if (source.grab(frame)) {
      resize(frame, downscaledFrame, downscaledFrameSize)

      // face size is 1/5th to 2/3rd of screen height
      val newFaces = new RectVector()
      classifier.detectMultiScale(downscaledFrame, newFaces, 1.1, 3, 0,
        new Size(downscaledFrame.rows / 5, downscaledFrame.rows / 5),
        new Size(downscaledFrame.rows * 2 / 3, downscaledFrame.rows * 2 / 3))

      if (newFaces.empty()) {
        faces.clear()
      } else {
        if (faces.empty()) {
          val sortedBySize = newFaces.get().sortBy(-_.width())
          faces.clear()
          sortedBySize.take(2).foreach(faces.push_back)
        } else if (faces.size == 1) {
          val sortedBySimilarity = newFaces.get().sortBy(r => rectDiff(r, faces.get(0)))
          faces.clear()
          faces.push_back(sortedBySimilarity.head)
          val sortedBySize = sortedBySimilarity.drop(1).sortBy(-_.width())
          sortedBySize.take(1).foreach(faces.push_back)
        } else { // faces.size == 2
          val search = newFaces.get().flatMap { r =>
            Array(
              SearchResult(r, rectDiff(r, faces.get(0)), 0),
              SearchResult(r, rectDiff(r, faces.get(1)), 1))
          }
          val sorted = search.sortBy(_.diff)
          val closest = sorted.head
          val secondClosest = sorted.drop(1).filter(s => s.index != closest.index && s.r != closest.r).take(1)
          faces.clear()
          faces.push_back(closest.r)
          secondClosest.map(_.r).foreach(faces.push_back)
        }
      }
      true
    } else {
      false
    }
  }

  case class SearchResult(r: Rect, diff: Long, index: Int)

  /**
   * Calculates metric of a difference between two Rects
   */
  private def rectDiff(a: Rect, b: Rect): Long = {
    val i = rectIntersect(a, b).area()
    if (i > 0) {
      -i
    } else {
      pointNorm(pointMinus(new Point(a.x+a.width/2, a.y+a.height/2), new Point(b.x+b.width/2, b.y+b.height/2))).toLong
    }
  }

  def getFaces: RectVector = {
    val rects = new RectVector()
    for (face <- faces.get()) {
      rects.push_back(new Rect(
        (face.x * ratio).toInt, (face.y * ratio).toInt,
        (face.width * ratio).toInt, (face.height * ratio).toInt))
    }
    rects
  }
}
