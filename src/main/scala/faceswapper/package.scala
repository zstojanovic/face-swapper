import org.bytedeco.opencv.opencv_core.{Point, Point2f, Rect, Size}

package object faceswapper {

  val FaceCount = 2

  def doRectsIntersect(a: Rect, b: Rect): Boolean = {
    if ((a.x + a.width) < b.x || a.x > (b.x + b.width)) {
      false
    } else if ((a.y + a.height) < b.y || a.y > (b.y + b.height)) {
      false
    } else {
      true
    }
  }

  def rectMinusPoint(a: Rect, p: Point): Rect = {
    val r = new Rect(a)
    r.x(a.x - p.x)
    r.y(a.y - p.y)
    r
  }

  def rectPlusSize(a: Rect, s: Size): Rect = {
    val r = new Rect(a)
    r.width(a.width + s.width)
    r.height(a.height + s.height)
    r
  }

  def pointNorm(a: Point): Double = {
    scala.math.sqrt(a.x * a.x + a.y * a.y)
  }

  def pointMinus(a: Point2f, b: Point2f): Point2f = {
    new Point2f(a.x - b.x, a.y - b.y)
  }

  def pointMinus(a: Point, b: Point): Point = {
    new Point(a.x - b.x, a.y - b.y)
  }

  def pointPlus(a: Point2f, b: Point2f): Point2f = {
    new Point2f(a.x + b.x, a.y + b.y)
  }

  def rectIntersect(a: Rect, b: Rect): Rect = {
    val r = new Rect(a)
    val x1 = a.x.max(b.x)
    val y1 = a.y.max(b.y)
    r.width((a.x + a.width).min(b.x + b.width) - x1)
    r.height((a.y + a.height).min(b.y + b.height) - y1)
    r.x(x1)
    r.y(y1)
    if (r.width <= 0 || r.height <= 0) {
      new Rect()
    } else {
      r
    }
  }

  def rectUnion(a: Rect, b: Rect): Rect = {
    val xs = Set(a.x, b.x, a.x + a.width, b.x + b.width)
    val ys = Set(a.y, b.y, a.y + a.height, b.y + b.height)
    val x = xs.min
    val y = ys.min
    new Rect(x, y, xs.max - x, ys.max - y)
  }
}
