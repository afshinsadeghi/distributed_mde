import java.lang.{ Integer => JavaInt, Long => JavaLong, String => JavaString, Double => JavaFloat }
import scala.io.Source
import java.util
import java.util.Map.Entry
import java.util.Timer
import javax.cache.processor.{ EntryProcessor, MutableEntry }
import org.apache.ignite.cache.query.SqlFieldsQuery
import org.apache.ignite.internal.util.scala.impl
import org.apache.ignite.scalar.scalar
import org.apache.ignite.scalar.scalar._
import org.apache.ignite.stream.StreamReceiver
import org.apache.ignite.{ IgniteCache, IgniteException }
import scala.util.Random
import scala.io.Source._
import scala.collection.JavaConverters._
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
import java.util.Calendar;
import java.io.BufferedWriter
import java.io._
import scala.collection.mutable.ListBuffer
import org.apache.commons.math3.analysis.function.Sqrt
import scala.util.control.Breaks._
import java.util.Collection
import org.apache.ignite.lang.IgniteCallable
import org.apache.ignite.lang.IgniteReducer
import java.util.ArrayList
import org.apache.ignite.Ignite
import org.apache.ignite.Ignition
import org.apache.ignite.compute.{ ComputeJob, ComputeJobResult, ComputeTaskSplitAdapter }
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map

class ComputeInsideNode(entityCache: Map[Int, DenseMatrix[Double]], relationCache: Map[Int, DenseMatrix[Double]], variables: Map[String, String]) {

  var dimNum = Integer.parseInt(variables("dim_num"))
  var batchsize = Integer.parseInt(variables("batchsize"))
  var margin = variables("margin").toDouble
  var lr_rate = variables("lr_rate").toDouble
  var batchesPerNode = Integer.parseInt(variables("batchesPerNode"))
  var updateGamma = variables("updateGamma").toBoolean
  var gamma_p = variables("gamma_p").toDouble
  var gamma_n = variables("gamma_n").toDouble
  var delta_p = variables("delta_p").toDouble
  var delta_n = variables("delta_n").toDouble
  var beta1: Double = variables("beta1").toDouble
  var beta2: Double = variables("beta2").toDouble

  var iteration =  Integer.parseInt(variables("iteration"))

  // Initialize optimizer
  val numerical_stability = .00000001
  //val optimizer_name = "sgd" // "adam"
  val optimizer_name = "adam" // "adam"
  /**
    * Initialize momentum term, exponential weight for moving average of the gradient
    */
  var optimizer_s: Double = .9

  /**
    * Initialize exponential weight for moving average of the history of gradient sizes
    */
  var optimizer_r: Double = .999

  var h_moment1:DenseMatrix[Double] = DenseMatrix.zeros(8,dimNum)
  //var h_moment1: DenseMatrix[Double] = null
  /**
    * The second moment captures a moving average of the squared gradient, or the magnitude of the
    * gradient. It is applied with the Adam optimizer.
    */
  var h_moment2:DenseMatrix[Double] = DenseMatrix.zeros(8,dimNum)
  //var h_moment2:DenseMatrix[Double] = null

  var r_moment1:DenseMatrix[Double] = DenseMatrix.zeros(8,dimNum)
  var r_moment2:DenseMatrix[Double] = DenseMatrix.zeros(8,dimNum)
  var t_moment1:DenseMatrix[Double] = DenseMatrix.zeros(8,dimNum)
  var t_moment2:DenseMatrix[Double] = DenseMatrix.zeros(8,dimNum)

  /** Type alias. */
  type Cache = IgniteCache[JavaString, JavaInt]
  type CacheFilter = IgniteCache[TripleID, Int]
  type TriplesCache = IgniteCache[Int, TripleID]

  def rand_max(x: Int): Int = {
    var res = (scala.util.Random.nextInt() * scala.util.Random.nextInt()) % x
    while (res < 0)
      res += x
    return res
  }



  def norm_l2(a: DenseVector[Double]): Double = {
    var sum: Double = 0.0
    for (i <- 0 until a.size)
      sum = sum + (a(i) * a(i))
    sum = sqrt(sum)
    sum
  }

  def calc_score(triple: TripleID): Double = {
    var score: Double = 0.0
    var psi: Double = 1.2

    var head = entityCache(triple.head)
    var tail = entityCache(triple.tail)
    var rel = relationCache(triple.rel)

    var row0 = head(0, ::).t + rel(0, ::).t - tail(0, ::).t
    var row4 = head(4, ::).t + rel(4, ::).t - tail(4, ::).t
    var a = (norm_l2(row0) + norm_l2(row4)) / 2.0

    var row1 = head(1, ::).t + tail(1, ::).t - rel(1, ::).t
    var row5 = head(5, ::).t + tail(5, ::).t - rel(5, ::).t
    var b = (norm_l2(row1) + norm_l2(row5)) / 2.0

    var row2 = rel(2, ::).t + tail(2, ::).t - head(2, ::).t
    var row6 = rel(6, ::).t + tail(6, ::).t - head(6, ::).t
    var c = (norm_l2(row2) + norm_l2(row6)) / 2.0

    var row3 = head(3, ::).t - rel(3, ::).t * tail(3, ::).t
    var row7 = head(7, ::).t - rel(7, ::).t * tail(7, ::).t
    var d = (norm_l2(row3) + norm_l2(row7)) / 2.0

    score = ((1.5 * a + 3 * b + 1.5 * c + 3 * d) / 9.0 - psi)
    //score = norm_l2(row0) //
    //println(score)
    score
  }

  def gradA(h: DenseVector[Double], t: DenseVector[Double], r: DenseVector[Double]): DenseVector[Double] = {
    val grad: DenseVector[Double] = ((h + r - t ) / norm_l2(h + r - t )) * 1.5 / 9.0
    grad

  }

  def gradB(h: DenseVector[Double], t: DenseVector[Double], r: DenseVector[Double]): DenseVector[Double] = {
    val grad: DenseVector[Double] = ((h + t - r ) / norm_l2(h + t - r )) * 3.0 / 9.0
    grad

  }

  def gradC(h: DenseVector[Double], t: DenseVector[Double], r: DenseVector[Double]): DenseVector[Double] = {

    val grad: DenseVector[Double] = ((t + r - h ) / norm_l2(t + r - h )) * 1.5 / 9.0
    grad

  }

  def gradD(h: DenseVector[Double], t: DenseVector[Double], r: DenseVector[Double]): Tuple3[DenseVector[Double], DenseVector[Double], DenseVector[Double]] = {

    val gradH: DenseVector[Double] = ((h - r * t) /norm_l2(h - r * t ))  * 3.0 / 9.0
    val gradR: DenseVector[Double] = ((h - r * t) /norm_l2(h - r * t )) * (t) * 3.0 / 9.0
    val gradT: DenseVector[Double] = ((h - r * t) /norm_l2(h - r * t )) * (-r) * 3.0 / 9.0
    (gradH, gradR, gradT)

  }

  def gradient(triple: TripleID, trueTriple: Boolean, beta: Double, entity_tmp: Map[Int, DenseMatrix[Double]], relation_tmp: Map[Int, DenseMatrix[Double]]) {
    var head = entityCache(triple.head)
    //var test = entityCache(3).t
    //println(test(0,0))
    var tail = entityCache(triple.tail)
    var rel = relationCache(triple.rel)
    
    var gradHead :DenseMatrix[Double] = DenseMatrix.zeros(8, dimNum)
    var gradRel :DenseMatrix[Double] = DenseMatrix.zeros(8, dimNum)
    var gradTail :DenseMatrix[Double] = DenseMatrix.zeros(8, dimNum)

    var gA0 = gradA(head(0, ::).t, tail(0, ::).t, rel(0, ::).t)
    var gA4 = gradA(head(4, ::).t, tail(4, ::).t, rel(4, ::).t)
    var gB1 = gradB(head(1, ::).t, tail(1, ::).t, rel(1, ::).t)
    var gB5 = gradB(head(5, ::).t, tail(5, ::).t, rel(5, ::).t)
    var gC2 = gradC(head(2, ::).t, tail(2, ::).t, rel(2, ::).t)
    var gC6 = gradC(head(6, ::).t, tail(6, ::).t, rel(6, ::).t)
    var gD3 = gradD(head(3, ::).t, tail(3, ::).t, rel(3, ::).t)
    var gD7 = gradD(head(7, ::).t, tail(7, ::).t, rel(7, ::).t)

    gradHead(0, ::) := gA0.t * beta
    gradHead(4, ::) := gA4.t * beta
    gradHead(1, ::) := gB1.t * beta
    gradHead(5, ::) := gB5.t * beta
    gradHead(2, ::) := gC2.t * beta * (-1.0)
    gradHead(6, ::) := gC6.t * beta * (-1.0)
    gradHead(3, ::) := gD3._1.t * beta
    gradHead(7, ::) := gD7._1.t * beta

    gradRel(0, ::) := gA0.t * beta
    gradRel(4, ::) := gA4.t * beta
    gradRel(1, ::) := gB1.t * beta * (-1.0)
    gradRel(5, ::) := gB5.t * beta * (-1.0)
    gradRel(2, ::) := gC2.t * beta
    gradRel(6, ::) := gC6.t * beta
    gradRel(3, ::) := gD3._2.t * beta
    gradRel(7, ::) := gD7._2.t * beta

    gradTail(0, ::) := gA0.t * beta * (-1.0)
    gradTail(4, ::) := gA4.t * beta * (-1.0)
    gradTail(1, ::) := gB1.t * beta
    gradTail(5, ::) := gB5.t * beta
    gradTail(2, ::) := gC2.t * beta
    gradTail(6, ::) := gC6.t * beta
    gradTail(3, ::) := gD3._3.t * beta
    gradTail(7, ::) := gD7._3.t * beta

    //println("triple head"+ triple.head)
    // println("contains:"+entity_tmp.contains(triple.head))
    if (!entity_tmp.contains(triple.head))
      entity_tmp.put(triple.head, DenseMatrix.zeros[Double](8, dimNum)) // entity_tmp.put(triple.head, head)
    if (!entity_tmp.contains(triple.tail))
      entity_tmp.put(triple.tail, DenseMatrix.zeros[Double](8, dimNum)) // entity_tmp.put(triple.tail, tail)
    if (!relation_tmp.contains(triple.rel))
      relation_tmp.put(triple.rel, DenseMatrix.zeros[Double](8, dimNum)) //relation_tmp.put(triple.rel, rel)

    //Check this
    if (!trueTriple) lr_rate = lr_rate * -1.0
    //    println("lr_rate" + lr_rate)
    //println("gradients ga0 for head:" + gradHead(0, ::))
    //println("gradients ga0 for tail:" + gradTail(0, ::))
    //println("gradients ga0 for rel:" + gradRel(0, ::) )

    //println("gradients ga1 for head:" + gradHead(1, ::))
    //println("gradients ga1 for tail:" + gradTail(1, ::))
    //println("gradients ga1 for rel:" + gradRel(1, ::) )
    //println("gradients for head:" + gradHead)
    //    println("gradients for tail:" + gradTail)
    //    println("gradients for rel:" + gradRel )

    var h = entity_tmp(triple.head)
    var t = entity_tmp(triple.tail)
    var r = relation_tmp(triple.rel)

    //println("h  head:" + h(0, ::))
    if (optimizer_name == "sgd") { //layer.weights -= lr * gradient
      var hNew = h + gradHead * lr_rate   //not sure if sign of lr is right, test it later
      //println("h new  head:" + hNew)
      entity_tmp.put(triple.head, hNew)


      var tNew = t - gradTail * lr_rate
      entity_tmp.put(triple.tail, tNew)


      var rNew = r + gradRel * lr_rate
      relation_tmp.put(triple.rel, rNew)
    }
    else if (optimizer_name == "adam") { // from https//github.com/JeremyNixon/sparkdl/blob/master/src/main/scala/org/apache/spark/ml/sparkdl/dl.scala

      h_moment1 = optimizer_s * h_moment1 + (1- optimizer_s) * gradHead
      h_moment2 = optimizer_r * h_moment2 + (1- optimizer_r) * (gradHead *:* gradHead)
      var m1_unbiased = h_moment1 /:/ (1 - (math.pow(optimizer_s, iteration + 1)))
      var m2_unbiased = h_moment2 /:/ (1- (math.pow(optimizer_r, iteration + 1)))
      val hNew  = h - ( lr_rate * m1_unbiased /:/ (sqrt(m2_unbiased) + numerical_stability) )


      r_moment1 = optimizer_s * r_moment1 + (1- optimizer_s) * gradHead
      r_moment2 = optimizer_r * r_moment2 + (1- optimizer_r) * (gradHead *:* gradHead)
      m1_unbiased = r_moment1 /:/ (1 - (math.pow(optimizer_s, iteration + 1)))
      m2_unbiased = r_moment2 /:/ (1- (math.pow(optimizer_r, iteration + 1)))
      val rNew  = r - ( lr_rate * m1_unbiased /:/ (sqrt(m2_unbiased) + numerical_stability) )

      t_moment1 = optimizer_s * t_moment1 + (1- optimizer_s) * gradHead
      t_moment2 = optimizer_r * t_moment2 + (1- optimizer_r) * (gradHead *:* gradHead)
      m1_unbiased = h_moment1 /:/ (1 - (math.pow(optimizer_s, iteration + 1)))
      m2_unbiased = h_moment2 /:/ (1- (math.pow(optimizer_r, iteration + 1)))
      val tNew  = t - ( lr_rate * m1_unbiased /:/ (sqrt(m2_unbiased) + numerical_stability) )


      entity_tmp.put(triple.head, hNew)
      entity_tmp.put(triple.tail, tNew)
      relation_tmp.put(triple.rel, rNew)
    }

  }

  def loss(posScore: Double, negScore: Double): Tuple3[Double, Double, Double] = {
    var lambda_pos: Double = gamma_p - delta_p
    var lambda_neg: Double = gamma_n - delta_n
    //println("positive score " + posScore)
    //println("positive loss before " + ((posScore - lambda_pos) * (-1.0) + margin))

    var pos_loss: Double = max(0.0, ((posScore - lambda_pos) * (-1.0) + margin))
    //println("positive loss " + pos_loss)

    var neg_loss: Double = max(0.0, ((negScore - lambda_neg)  + margin))
    var loss: Double = beta1 * pos_loss + beta2 * neg_loss

    (loss, pos_loss, neg_loss)
  }


  /**
   * Not in Use
   * @param posScore
   * @param negScore
   * @return
   */
  def loss_old(posScore: Double, negScore: Double): Tuple3[Double, Double, Double] = {
    var loss: Double = max(0.0, ((posScore - negScore + margin) * (-1.0)))
    var neg_loss: Double = 0.0
    (loss, loss, neg_loss)
  }

  def train(trueTriple: TripleID, falseTriple: TripleID, entity_tmp: Map[Int, DenseMatrix[Double]], relation_tmp: Map[Int, DenseMatrix[Double]]): Tuple3[Double, Double, Double] = {
    var res: Double = 0
    var resP: Double = 0
    var resN: Double = 0
    var posScore = calc_score(trueTriple)
    //println("positive score " + posScore)
    var negScore = calc_score(falseTriple)
    var (totLoss, posLoss, negLoss) = loss(posScore, negScore)
    //println("positive loss " + posLoss)
    if (posLoss != 0)
      gradient(trueTriple, true, totLoss, entity_tmp, relation_tmp)
    if (negLoss != 0)
      gradient(falseTriple, false, totLoss, entity_tmp, relation_tmp)
    //if (posLoss - negLoss + 1 > 0)
    //  gradient(trueTriple, true, totLoss, entity_tmp, relation_tmp)
    //  gradient(falseTriple, false, totLoss, entity_tmp, relation_tmp)
    res += totLoss
    resP += posLoss
    resN += negLoss

    (res, resP, resN)
  }

  def computeInsideNode(): Tuple5[Double, Map[Int, DenseMatrix[Double]], Map[Int, DenseMatrix[Double]], Double, Double] = {
    println("iteration: "+iteration)
    val remoteIgnite = Ignition.localIgnite()
    var trainingTriplesCache: IgniteCache[JavaInt, TripleID] = remoteIgnite.cache("tripleIde");
    var existingTriplesCache: IgniteCache[TripleID, JavaInt] = remoteIgnite.cache("allTriplesCache");
    var entity_tmp: Map[Int, DenseMatrix[Double]] = Map.empty[Int, DenseMatrix[Double]]
    var relation_tmp: Map[Int, DenseMatrix[Double]] = Map.empty[Int, DenseMatrix[Double]]
    var entity_num = entityCache.size
    var relation_num = relationCache.size
    var triple_num = trainingTriplesCache.size()

    var totRes: Double = 0
    var totResPos: Double = 0
    var totResNeg: Double = 0
    for (batch <- 0 until batchesPerNode) {
      for (i <- 0 until batchsize) {
        var id = rand_max(triple_num)
        var randomEntity = rand_max(entity_num)
        var triple: TripleID = trainingTriplesCache.get(id)

        if (scala.util.Random.nextInt() % 1000 < 500) {
          while (randomEntity == triple.tail || existingTriplesCache.get(new TripleID(triple.head, randomEntity, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          if (!entity_tmp.contains(randomEntity)) {
            val temp = train(triple, new TripleID(triple.head, randomEntity, triple.rel), entity_tmp, relation_tmp)
            totRes += temp._1
            totResPos += temp._2
            totResNeg += temp._3
          }
        } else {
          while (randomEntity == triple.head || existingTriplesCache.get(new TripleID(randomEntity, triple.tail, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          if (!entity_tmp.contains(randomEntity)) {
            val temp = train(triple, new TripleID(randomEntity, triple.tail, triple.rel), entity_tmp, relation_tmp)
            totRes += temp._1
            totResPos += temp._2
            totResNeg += temp._3
          }
        }
      }
    }
    (totRes, entity_tmp, relation_tmp, totResPos, totResNeg)
  }

}