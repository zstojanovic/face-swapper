package faceswapper

import org.bytedeco.javacv.{FFmpegFrameGrabber, Frame, OpenCVFrameConverter, OpenCVFrameGrabber}
import org.bytedeco.opencv.opencv_core.{Mat, Size}

abstract class VideoSource {
  val converter = new OpenCVFrameConverter.ToMat
  def size(): Size
  def grabFrame(): Frame

  def grab(image: Mat): Boolean = {
    val frame = grabFrame()
    if (frame != null) {
      converter.convert(frame).copyTo(image)
      true
    } else {
      false
    }
  }
}

object VideoSource {
  def create(source: String): VideoSource =
    if (source.forall(Character.isDigit)) {
      new CameraSource(source.toInt)
    } else {
      new VideoFileSource(source)
    }
}

class VideoFileSource(filename: String) extends VideoSource {
  private val grabber = new FFmpegFrameGrabber(filename)
  try {
    grabber.start()
  } catch {
    case e: Exception =>
      println(s"Couldn't open video source file $filename (${e.getMessage})")
      sys.exit(-1)
  }
  val s = new Size(grabber.getImageWidth, grabber.getImageHeight)

  def size(): Size = s

  def grabFrame(): Frame =
    grabber.grabFrame(false, true, true, false)
}

class CameraSource(deviceIndex: Int) extends VideoSource {
  private val grabber = new OpenCVFrameGrabber(deviceIndex)
  try {
    grabber.start()
  } catch {
    case e: Exception =>
      println(s"Couldn't open camera source with index $deviceIndex (${e.getMessage})")
      sys.exit(-1)
  }
  val s = new Size(grabber.getImageWidth, grabber.getImageHeight)

  def size(): Size = s

  def grabFrame(): Frame =
    grabber.grabFrame()
}