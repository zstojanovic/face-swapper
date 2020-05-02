package faceswapper

import org.bytedeco.javacpp.indexer.FloatIndexer
import org.bytedeco.opencv.opencv_core.{AbstractScalar, Mat, Point, Point2f, Point2fVectorVector, Rect, RectVector, Size}
import org.bytedeco.opencv.opencv_face.FacemarkLBF
import org.bytedeco.opencv.global.opencv_imgproc._
import org.bytedeco.opencv.global.opencv_core._

class FaceSwapper() {
  private val LandmarksFilename = "lbfmodel.yaml"
  private val faceLandmarkDetector: FacemarkLBF = FacemarkLBF.create
  try {
    faceLandmarkDetector.loadModel(LandmarksFilename)
  } catch {
    case e: Exception =>
      println(s"Error loading face landmark model file $LandmarksFilename")
      println("If it's missing, you can find it at https://github.com/kurnianggoro/GSOC2017/blob/master/data")
      sys.exit(-1)
  }

  private var smallFrame: Mat = new Mat
  private var rectAnn: Rect = _
  private var rectBob: Rect = _
  private var bigRectAnn = new Rect
  private var bigRectBob = new Rect
  private val pointsAnn = new Point(9)
  private val pointsBob = new Point(9)
  private val affineTransformPointsAnn = new Point2f(3)
  private val affineTransformPointsBob = new Point2f(3)
  private val featherAmountAnn = new Size
  private val featherAmountBob = new Size
  private var transAnnToBob = new Mat
  private val transBobToAnn = new Mat
  private val maskAnn = new Mat
  private val maskBob = new Mat
  private val warpedMaskAnn = new Mat
  private val warpedMaskBob = new Mat
  private val refinedMasks = new Mat
  private val refinedAnnAndBobWarped = new Mat
  private val refinedBobAndAnnWarped = new Mat
  private val faceAnn = new Mat
  private val faceBob = new Mat
  private val warpedFaces = new Mat
  private val warpedFaceAnn = new Mat
  private val warpedFaceBob = new Mat
  private val sourceHistInt = Array.ofDim[Int](3,256)
  private val targetHistInt = Array.ofDim[Int](3,256)
  private val sourceHistogram = Array.ofDim[Float](3,256)
  private val targetHistogram = Array.ofDim[Float](3,256)
  private val lut = new Mat(1, 256, CV_8UC3)
  private val MinusOnePoint = new Point(-1, -1)
  private val warpedFacesF = new Mat()
  private val smallFrameF = new Mat()
  private val refinedMasksF = new Mat()

  def swapFaces(frame: Mat, rectAnn: Rect, rectBob: Rect) {
    calcSmallFrame(frame, rectAnn, rectBob)
    findFacePoints()
    calcTransformationMatrices()

    calcMasks()
    calcWarpedMasks()
    calcRefinedMasks()

    extractFaces()
    calcWarpedFaces()
    colorCorrectFaces()

    pasteFacesOnFrame()
  }

  private def calcSmallFrame(frame: Mat, rAnn: Rect, rBob: Rect): Unit = {
    var bounding_rect = rectUnion(rAnn, rBob)

    bounding_rect = rectMinusPoint(bounding_rect, new Point(50, 50))
    bounding_rect = rectPlusSize(bounding_rect, new Size(100, 100))

    bounding_rect = rectIntersect(bounding_rect, new Rect(0, 0, frame.cols, frame.rows))

    this.rectAnn = rectMinusPoint(rAnn, bounding_rect.tl())
    this.rectBob = rectMinusPoint(rBob, bounding_rect.tl())

    bigRectAnn =
      rectIntersect(
        rectPlusSize(
          rectMinusPoint(this.rectAnn, new Point(rAnn.width / 4, rAnn.height / 4)),
          new Size(rAnn.width / 2, rAnn.height / 2)),
        new Rect(0, 0, bounding_rect.width, bounding_rect.height))
    bigRectBob =
      rectIntersect(
        rectPlusSize(
          rectMinusPoint(this.rectBob, new Point(rBob.width / 4, rBob.height / 4)),
          new Size(rBob.width / 2, rBob.height / 2)),
        new Rect(0, 0, bounding_rect.width, bounding_rect.height))

    smallFrame = frame(bounding_rect)
  }

  private def findFacePoints(): Unit = {
    val shapes = new Point2fVectorVector
    faceLandmarkDetector.fit(smallFrame, new RectVector(rectAnn, rectBob), shapes)

    def set1(point: Point, i: Int, a: Int, b: Int) = {
      set2(point, i, shapes.get(a).get(b))
    }

    def set2(point: Point, i: Int, p2: Point2f) = {
      point.position(i).x(p2.x.toInt)
      point.position(i).y(p2.y.toInt)
    }

    def set3(point: Point2f, i: Int, p2: Point) = {
      point.position(i).x(p2.x)
      point.position(i).y(p2.y)
    }

    def set4(point: Point2f, i: Int, p2: Point2f) = {
      point.position(i).x(p2.x.toInt)
      point.position(i).y(p2.y.toInt)
    }

    set1(pointsAnn, 0, 0, 0)
    set1(pointsAnn, 1, 0, 3)
    set1(pointsAnn, 2, 0, 5)
    set1(pointsAnn, 3, 0, 8)
    set1(pointsAnn, 4, 0, 11)
    set1(pointsAnn, 5, 0, 13)
    set1(pointsAnn, 6, 0, 16)
    val nose_length_ann = pointMinus(shapes.get(0).get(27), shapes.get(0).get(30))
    set2(pointsAnn, 7, pointPlus(shapes.get(0).get(26), nose_length_ann))
    set2(pointsAnn, 8, pointPlus(shapes.get(0).get(17), nose_length_ann))

    set1(pointsBob, 0, 1, 0)
    set1(pointsBob, 1, 1, 3)
    set1(pointsBob, 2, 1, 5)
    set1(pointsBob, 3, 1, 8)
    set1(pointsBob, 4, 1, 11)
    set1(pointsBob, 5, 1, 13)
    set1(pointsBob, 6, 1, 16)
    val nose_length_bob = pointMinus(shapes.get(1).get(27), shapes.get(1).get(30))
    set2(pointsBob, 7, pointPlus(shapes.get(1).get(26), nose_length_bob))
    set2(pointsBob, 8, pointPlus(shapes.get(1).get(17), nose_length_bob))

    set3(affineTransformPointsAnn, 0, pointsAnn.position(3))
    set4(affineTransformPointsAnn, 1, shapes.get(0).get(36))
    set4(affineTransformPointsAnn, 2, shapes.get(0).get(45))

    set3(affineTransformPointsBob, 0, pointsBob.position(3))
    set4(affineTransformPointsBob, 1, shapes.get(1).get(36))
    set4(affineTransformPointsBob, 2, shapes.get(1).get(45))

    featherAmountAnn.width((
      pointNorm(
        pointMinus(
          new Point(pointsAnn.position(0).x, pointsAnn.position(0).y),
          new Point(pointsAnn.position(6).x, pointsAnn.position(6).y))) / 8).toInt)
    featherAmountAnn.height(featherAmountAnn.width())
    featherAmountBob.width((
      pointNorm(
        pointMinus(
          new Point(pointsBob.position(0).x, pointsBob.position(0).y),
          new Point(pointsBob.position(6).x, pointsBob.position(6).y))) / 8).toInt)
    featherAmountBob.height(featherAmountBob.width())
  }

  private def calcTransformationMatrices(): Unit = {
    transAnnToBob = getAffineTransform(
      affineTransformPointsAnn.position(0), affineTransformPointsBob.position(0))
    invertAffineTransform(transAnnToBob, transBobToAnn)
  }

  private def calcMasks(): Unit = {
    maskAnn.create(smallFrame.size, CV_8UC1)
    maskBob.create(smallFrame.size, CV_8UC1)
    maskAnn.put(AbstractScalar.BLACK)
    maskBob.put(AbstractScalar.BLACK)

    fillConvexPoly(maskAnn, pointsAnn.position(0), 9, AbstractScalar.WHITE)
    fillConvexPoly(maskBob, pointsBob.position(0), 9, AbstractScalar.WHITE)
  }

  private def calcWarpedMasks(): Unit = {
    warpAffine(maskAnn, warpedMaskAnn, transAnnToBob,
      smallFrame.size, INTER_NEAREST, BORDER_CONSTANT, AbstractScalar.BLACK)
    warpAffine(maskBob, warpedMaskBob, transBobToAnn,
      smallFrame.size, INTER_NEAREST, BORDER_CONSTANT, AbstractScalar.BLACK)
  }

  private def calcRefinedMasks(): Unit = {
    bitwise_and(maskAnn, warpedMaskBob, refinedAnnAndBobWarped)
    bitwise_and(maskBob, warpedMaskAnn, refinedBobAndAnnWarped)

    refinedAnnAndBobWarped.copyTo(refinedMasks, refinedAnnAndBobWarped)
    refinedBobAndAnnWarped.copyTo(refinedMasks, refinedBobAndAnnWarped)
    featherMask(refinedMasks(bigRectAnn), featherAmountAnn)
    featherMask(refinedMasks(bigRectBob), featherAmountBob)
  }

  private def extractFaces(): Unit = {
    smallFrame.copyTo(faceAnn, maskAnn)
    smallFrame.copyTo(faceBob, maskBob)
  }

  private def calcWarpedFaces(): Unit = {
    warpAffine(faceAnn, warpedFaceAnn, transAnnToBob,
      smallFrame.size, INTER_NEAREST, BORDER_CONSTANT, AbstractScalar.BLACK)
    warpAffine(faceBob, warpedFaceBob, transBobToAnn,
      smallFrame.size, INTER_NEAREST, BORDER_CONSTANT, AbstractScalar.BLACK)

    warpedFaceAnn.copyTo(warpedFaces, warpedMaskAnn)
    warpedFaceBob.copyTo(warpedFaces, warpedMaskBob)
  }

  private def colorCorrectFaces() {
    specifyHistogram(smallFrame(bigRectAnn), warpedFaces(bigRectAnn), warpedMaskBob(bigRectAnn))
    specifyHistogram(smallFrame(bigRectBob), warpedFaces(bigRectBob), warpedMaskAnn(bigRectBob))
  }

  private def featherMask(refinedMasks: Mat, featherAmount: Size) {
    erode(refinedMasks, refinedMasks, getStructuringElement(MORPH_RECT, featherAmount),
      MinusOnePoint, 1, BORDER_CONSTANT, AbstractScalar.BLACK)
    blur(refinedMasks, refinedMasks, featherAmount, MinusOnePoint, BORDER_CONSTANT)
  }

  private def pasteFacesOnFrame(): Unit = {
    warpedFaces.convertTo(warpedFacesF, CV_32FC3)
    smallFrame.copyTo(smallFrameF)
    smallFrameF.convertTo(smallFrameF, CV_32FC3)

    cvtColor(refinedMasks, refinedMasksF, COLOR_GRAY2BGR)
    // Normalize the alpha mask to keep intensity between 0 and 1
    refinedMasksF.convertTo(refinedMasksF, CV_32FC3, 1.0 / 255, 0)

    // Storage for output image
    val output = Mat.zeros(warpedFacesF.size(), warpedFacesF.`type`()).asMat()

    // Multiply the foreground with the alpha matte
    multiply(refinedMasksF, warpedFacesF, warpedFacesF)


    // Multiply the background with ( 1 - alpha )
    applyFunction(refinedMasksF, f => 1.0f - f)
    multiply(refinedMasksF, smallFrameF, smallFrameF)

    // Add the masked foreground and background.
    add(warpedFacesF, smallFrameF, output)

    //applyFunction(output, f => f / 255)
    output.convertTo(output, smallFrame.`type`())
    output.copyTo(smallFrame)
  }

  def applyFunction(a: Mat, f: Float => Float): Unit = {
    val indexer = a.createIndexer().asInstanceOf[FloatIndexer]
    for (i <- 0 until (a.rows * a.cols * a.channels)) {
      indexer.put(i, f(indexer.get(i)))
    }
  }

  private def specifyHistogram(sourceImage: Mat, targetImage: Mat, mask: Mat) {
    for (i <- 0 until 256; j <- 0 until 3) {
      sourceHistInt(j)(i) = 0
      targetHistInt(j)(i) = 0
    }

    for (i <- 0 until mask.rows) {
      val current_mask_pixel = mask.row(i).data
      val current_source_pixel = sourceImage.row(i).data
      val current_target_pixel = targetImage.row(i).data
      for (j <- 0 until mask.cols) {
        if (current_mask_pixel.position(j).get != 0) {
          sourceHistInt(0)(current_source_pixel.position(j*3).get & 0xFF) =
            sourceHistInt(0)(current_source_pixel.position(j*3).get & 0xFF) + 1
          sourceHistInt(1)(current_source_pixel.position(j*3+1).get & 0xFF) =
            sourceHistInt(1)(current_source_pixel.position(j*3+1).get & 0xFF) + 1
          sourceHistInt(2)(current_source_pixel.position(j*3+2).get & 0xFF) =
            sourceHistInt(2)(current_source_pixel.position(j*3+2).get & 0xFF) + 1

          targetHistInt(0)(current_target_pixel.position(j*3).get & 0xFF) =
            targetHistInt(0)(current_target_pixel.position(j*3).get & 0xFF) + 1
          targetHistInt(1)(current_target_pixel.position(j*3+1).get & 0xFF) =
            targetHistInt(1)(current_target_pixel.position(j*3+1).get & 0xFF) + 1
          targetHistInt(2)(current_target_pixel.position(j*3+2).get & 0xFF) =
            targetHistInt(2)(current_target_pixel.position(j*3+2).get & 0xFF) + 1
        }
      }
    }

    // Calc CDF
    for (i <- 1 until 256) {
      sourceHistInt(0)(i) = sourceHistInt(0)(i) + sourceHistInt(0)(i - 1)
      sourceHistInt(1)(i) = sourceHistInt(1)(i) + sourceHistInt(1)(i - 1)
      sourceHistInt(2)(i) = sourceHistInt(2)(i) + sourceHistInt(2)(i - 1)

      targetHistInt(0)(i) = targetHistInt(0)(i) + targetHistInt(0)(i - 1)
      targetHistInt(1)(i) = targetHistInt(1)(i) + targetHistInt(1)(i - 1)
      targetHistInt(2)(i) = targetHistInt(2)(i) + targetHistInt(2)(i - 1)
    }

    // Normalize CDF
    for (i <- 0 until 256) {
      sourceHistogram(0)(i) =
        if (sourceHistInt(0)(255) != 0) sourceHistInt(0)(i).toFloat / sourceHistInt(0)(255) else 0
      sourceHistogram(1)(i) =
        if (sourceHistInt(1)(255) != 0) sourceHistInt(1)(i).toFloat / sourceHistInt(1)(255) else 0
      sourceHistogram(2)(i) =
        if (sourceHistInt(2)(255) != 0) sourceHistInt(2)(i).toFloat / sourceHistInt(2)(255) else 0

      targetHistogram(0)(i) =
        if (targetHistInt(0)(255) != 0) targetHistInt(0)(i).toFloat / targetHistInt(0)(255) else 0
      targetHistogram(1)(i) =
        if (targetHistInt(1)(255) != 0) targetHistInt(1)(i).toFloat / targetHistInt(1)(255) else 0
      targetHistogram(2)(i) =
        if (targetHistInt(2)(255) != 0) targetHistInt(2)(i).toFloat / targetHistInt(2)(255) else 0
    }

    // Create lookup table
    def binary_search(needle: Float, haystack: Array[Float]): Int = {
      var l = 0
      var r = 255
      var m = 0
      while (l < r) {
        m = (l + r) / 2
        if (needle > haystack(m)) {
          l = m + 1
        } else {
          r = m - 1
        }
      }
      // TODO check closest value
      m
    }

    val lutRow = lut.row(0).data()
    for (i <- 0 until 256) {
      lutRow.position(i*3).put(binary_search(targetHistogram(0)(i), sourceHistogram(0)).toByte)
      lutRow.position(i*3+1).put(binary_search(targetHistogram(1)(i), sourceHistogram(1)).toByte)
      lutRow.position(i*3+2).put(binary_search(targetHistogram(2)(i), sourceHistogram(2)).toByte)
    }
    LUT(targetImage, lut, targetImage)
  }
}
