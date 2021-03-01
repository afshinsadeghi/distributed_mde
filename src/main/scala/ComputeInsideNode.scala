import java.lang.{Double => JavaFloat, Integer => JavaInt, Long => JavaLong, String => JavaString}

import scala.io.Source
import java.util
import java.util.Map.Entry
import java.util.Timer

import javax.cache.processor.{EntryProcessor, MutableEntry}
import org.apache.ignite.cache.query.SqlFieldsQuery
import org.apache.ignite.internal.util.scala.impl
import org.apache.ignite.scalar.scalar
import org.apache.ignite.scalar.scalar._
import org.apache.ignite.stream.StreamReceiver
import org.apache.ignite.{IgniteCache, IgniteException}

import scala.util.Random
import scala.io.Source._
import scala.collection.JavaConverters._
//import breeze.linalg._
//import breeze.numerics._
//import breeze.stats.distributions._
import java.util.Calendar
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
import org.apache.ignite.compute.{ComputeJob, ComputeJobResult, ComputeTaskSplitAdapter}

import scala.collection.mutable.ListBuffer
import scala.collection.mutable.Map
import org.apache.spark.sql._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
//import scala.math._

class ComputeInsideNode(entityCache: Tensor[Float], relationCache: Tensor[Float], variables: Map[String, String]) {

  var dimNum = Integer.parseInt(variables("dim_num"))
  var batchsize = Integer.parseInt(variables("batchsize"))
  var margin = variables("margin").toFloat
  var lr_rate_ = variables("lr_rate").toFloat
  var lr_rate = 0.1
  //val learningRate:Float = 0.001f //will get the correct value later
  val learningRate: Float = 0.01f
  var batchesPerNode = Integer.parseInt(variables("batchesPerNode"))
  var updateGamma = variables("updateGamma").toBoolean
  var gamma_p = variables("gamma_p").toFloat
  var gamma_n = variables("gamma_n").toFloat
  var delta_p = variables("delta_p").toFloat
  var delta_n = variables("delta_n").toFloat
  var beta1: Float = variables("beta1").toFloat
  var beta2: Float = variables("beta2").toFloat

  var iteration = Integer.parseInt(variables("iteration"))


  //val optim = new Adadelta(learningRate)
  val optim = new SGD(learningRate)
  //val optimizer_name = "sgd"
  val optimizer_name = "adadelta"

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

  def norm_l2(a: Tensor[Float]): Float = {
    a.norm(2)
  }

  def calc_score2(tensorTotal: Tensor[Float], isPostive: Boolean): Tensor[Float] = {

    val batchSize = tensorTotal.size(1) / 6
    val scores: Tensor[Float] = Tensor(batchSize)

    var startPosition = 1;
    if (isPostive) {
      startPosition = 1;
    } else {
      startPosition = 3 * batchSize + 1
    }

    var index = 1
    for (i <- startPosition to (startPosition + batchSize - 1)) {
      var psi: Float = 1.2.toFloat

      var head = tensorTotal(i)
      var rel = tensorTotal(i + batchSize)
      var tail = tensorTotal(i + 2 * batchSize)

      var row0: Tensor[Float] = head.select(1, 1) + rel.select(1, 1) - tail.select(1, 1)
      var row4 = head.select(1, 5) + rel.select(1, 5) - tail.select(1, 5)
      var a = (norm_l2(row0) + norm_l2(row4)) / 2.0

      var row1 = head.select(1, 2) + tail.select(1, 2) - rel.select(1, 2)
      var row5 = head.select(1, 6) + tail.select(1, 6) - rel.select(1, 6)
      var b = (norm_l2(row1) + norm_l2(row5)) / 2.0


      var row2 = rel.select(1, 3) + tail.select(1, 3) - head.select(1, 3)
      var row6 = rel.select(1, 7) + tail.select(1, 7) - head.select(1, 7)
      var c = (norm_l2(row2) + norm_l2(row6)) / 2.0

      var row3 = head.select(1, 4) - rel.select(1, 4).cmul(tail.select(1, 4))
      var row7 = head.select(1, 8) - rel.select(1, 8).cmul(tail.select(1, 8))
      var d = (norm_l2(row3) + norm_l2(row7)) / 2.0

      var score: Float = ((1.5 * a + 3 * b + 1.5 * c + 3 * d) / 9.0 - psi).toFloat
      //score  = score - tanh(score)

      score
      scores.update(index, score)
      index = index + 1
    }
    scores
  }


  def TransE_Score(triple: TripleID, entityEmbedding: Tensor[Float], relationEmbedding: Tensor[Float]): Float = { //(Tensor[Float])
    // val sampleDistance = Tensor(sample.size, 1)
    //    var count = 1
    //    triple.map(row => {

    val subTensor = entityEmbedding.select(1, triple.head) //.reshape(Array(k,1))
    val predTensor = relationEmbedding.select(1, triple.rel) //.reshape(Array(k,1))
    val objTensor = entityEmbedding.select(1, triple.tail) //.reshape(Array(k,1))

    val dist = (subTensor + predTensor - objTensor).norm(2)

    //      sampleDistance.update(count,dist)
    //      count += 1
    //    })
    //    (sampleDistance)
    (dist)
  }

  def loss(posScore: Tensor[Float], negScore: Tensor[Float]): Tuple3[Float, Float, Float] = {
    var lambda_pos: Float = 1.0.toFloat // gamma_p - delta_p
    var lambda_neg: Float = 1.0.toFloat //gamma_n - delta_n

    var pos_loss = (ReLU().forward(((posScore - lambda_pos) * -1.0.toFloat) + margin)).sum
    var neg_loss = (ReLU().forward((negScore - lambda_neg) + margin)).sum

    var loss = beta1 * pos_loss + beta2 * neg_loss

    (loss, pos_loss, neg_loss)
  }


  def forward2(positiveTriples: List[TripleID], negativeTriples: List[TripleID], entityEmbedding: Tensor[Float], relationEmbedding: Tensor[Float]): Tuple4[Float, Float, Float, Tensor[Float]] = {
    var res: Float = 0
    var resP: Float = 0
    var resN: Float = 0

    val positiveHeadTensor: Tensor[Float] = Tensor(positiveTriples.size, 8, dimNum)
    val positiveTailTensor: Tensor[Float] = Tensor(positiveTriples.size, 8, dimNum)
    val positiveRelTensor: Tensor[Float] = Tensor(positiveTriples.size, 8, dimNum)
    var index = 1
    for (id <- positiveTriples) {
      positiveHeadTensor.update(index, entityEmbedding(id.head + 1))
      positiveRelTensor.update(index, relationEmbedding(id.rel + 1))
      positiveTailTensor.update(index, entityEmbedding(id.tail + 1))
      index = index + 1
    }
    index = 1;
    val negativeHeadTensor: Tensor[Float] = Tensor(negativeTriples.size, 8, dimNum)
    val negativeTailTensor: Tensor[Float] = Tensor(negativeTriples.size, 8, dimNum)
    val negativeRelTensor: Tensor[Float] = Tensor(negativeTriples.size, 8, dimNum)
    for (id <- negativeTriples) {
      negativeHeadTensor.update(index, entityEmbedding(id.head + 1))
      negativeRelTensor.update(index, relationEmbedding(id.rel + 1))
      negativeTailTensor.update(index, entityEmbedding(id.tail + 1))
      index = index + 1
    }

    val tensorTotal: Tensor[Float] = Tensor(6 * negativeTriples.size, 8, dimNum)

    val batchSize = positiveTriples.size

    for (i <- 1 to batchSize) {
      tensorTotal.update(i, positiveHeadTensor(i))
      tensorTotal.update(i + 1 * batchSize, positiveRelTensor(i))
      tensorTotal.update(i + 2 * batchSize, positiveTailTensor(i))
      tensorTotal.update(i + 3 * batchSize, negativeHeadTensor(i))
      tensorTotal.update(i + 4 * batchSize, negativeRelTensor(i))
      tensorTotal.update(i + 5 * batchSize, negativeTailTensor(i))
    }


    /**
     * I think update function should get List[TripleID] because then we can calculte each individual cal_score
     * But then optimize function does not work
     *
     * @param x
     * @return
     */

    def Update(x: Tensor[Float]) = {

      val posScore = calc_score2(x, true)
      val negScore = calc_score2(x, false)
      val (totLoss, posLoss, negLoss) = loss(posScore, negScore)
      res += totLoss
      resP += posLoss
      resN += negLoss

      (totLoss, derivetive(x))
    }

    def derivetive(x: Tensor[Float]): Tensor[Float] = {

      val batchSize = x.size(1) / 6
      val scores: Tensor[Float] = Tensor(6 * batchSize, 8, dimNum)
      for (i <- 1 to batchSize) {
        var head = tensorTotal(i)
        var rel = tensorTotal(i + batchSize)
        var tail = tensorTotal(i + 2 * batchSize)


        val score = gradTotal(head, tail, rel)


        scores.update(i + 0, score.select(1, 1))
        scores.update(i + 1 * batchSize, score.select(1, 2))
        scores.update(i + 2 * batchSize, score.select(1, 3))
        scores.update(i + 3 * batchSize, score.select(1, 4))
        scores.update(i + 4 * batchSize, score.select(1, 5))
        scores.update(i + 5 * batchSize, score.select(1, 6))
      }
      scores
    }

    def gradTotal(h: Tensor[Float], t: Tensor[Float], r: Tensor[Float]): Tensor[Float] = {
      val scores: Tensor[Float] = Tensor(6,8, dimNum)
      //val scores: Tensor[Float] = Tensor(8, dimNum)

      val g1A = gradA(h.select(1, 1), t.select(1, 1), r.select(1, 1))
      val g5A = gradA(h.select(1, 5), t.select(1, 5), r.select(1, 5))

      val g2B = gradB(h.select(1, 2), t.select(1, 2), r.select(1, 2))
      val g6B = gradB(h.select(1, 6), t.select(1, 6), r.select(1, 6))

      val g3c = gradC(h.select(1, 3), t.select(1, 3), r.select(1, 3))
      val g7c = gradC(h.select(1, 7), t.select(1, 7), r.select(1, 7))

      val g4d = gradD(h.select(1, 4), t.select(1, 4), r.select(1, 4))
      val g8d = gradD(h.select(1, 8), t.select(1, 8), r.select(1, 8))

      for(k <-1 to 6) {
        for (i <- 1 to 8) {
          val tmp: Tensor[Float] = Tensor(8, dimNum)
          tmp.update(i, g1A(k))
          tmp.update(i, g2B(k))
          tmp.update(i, g3c(k))
          tmp.update(i, g4d(k))
          tmp.update(i, g5A(k))
          tmp.update(i, g6B(k))
          tmp.update(i, g7c(k))
          tmp.update(i, g8d(k))
          scores.update(k, tmp)
        }
      }

      scores

    }

    def gradA(h: Tensor[Float], t: Tensor[Float], r: Tensor[Float]): Tensor[Float] = {
      val scores: Tensor[Float] = Tensor(6, dimNum)

      val gradHeadPositive = ((h + r - t) / norm_l2(h + r - t)) * 1.5.toFloat / 9.0.toFloat
      val gradRelPositive = gradHeadPositive
      val gradTailPositive = gradHeadPositive * -1.0.toFloat

      val gradHeadNegative = gradHeadPositive * -1.0.toFloat
      val gradTailNegative = gradTailPositive * -1.0.toFloat
      val gradRelNegative = gradRelPositive * -1.0.toFloat

      scores.update(1, gradHeadPositive)
      scores.update(2, gradRelPositive)
      scores.update(3, gradTailPositive)

      scores.update(4, gradHeadNegative)
      scores.update(5, gradRelNegative)
      scores.update(6, gradTailNegative)

      scores
    }

    def gradB(h: Tensor[Float], t: Tensor[Float], r: Tensor[Float]): Tensor[Float] = {
      val scores: Tensor[Float] = Tensor(6, dimNum)

      val gradHeadPositive = ((h + t - r) / norm_l2(h + t - r)) * 3.0.toFloat / 9.0.toFloat
      val gradRelPositive = gradHeadPositive
      val gradTailPositive = gradHeadPositive * -1.0.toFloat

      val gradHeadNegative = gradHeadPositive * -1.0.toFloat
      val gradTailNegative = gradTailPositive * -1.0.toFloat
      val gradRelNegative = gradRelPositive * -1.0.toFloat

      scores.update(1, gradHeadPositive)
      scores.update(2, gradRelPositive)
      scores.update(3, gradTailPositive)

      scores.update(4, gradHeadNegative)
      scores.update(5, gradRelNegative)
      scores.update(6, gradTailNegative)

      scores
    }

    def gradC(h: Tensor[Float], t: Tensor[Float], r: Tensor[Float]): Tensor[Float] = {
      val scores: Tensor[Float] = Tensor(6, dimNum)

      val gradHeadPositive = ((t + r - h) / norm_l2(t + r - h)) * 3.0.toFloat / 9.0.toFloat
      val gradRelPositive = gradHeadPositive
      val gradTailPositive = gradHeadPositive * -1.0.toFloat

      val gradHeadNegative = gradHeadPositive * -1.0.toFloat
      val gradTailNegative = gradTailPositive * -1.0.toFloat
      val gradRelNegative = gradRelPositive * -1.0.toFloat

      scores.update(1, gradHeadPositive)
      scores.update(2, gradRelPositive)
      scores.update(3, gradTailPositive)

      scores.update(4, gradHeadNegative)
      scores.update(5, gradRelNegative)
      scores.update(6, gradTailNegative)

      scores


    }

    def gradD(h: Tensor[Float], t: Tensor[Float], r: Tensor[Float]): Tensor[Float]= {
    //def gradD(h: Tensor[Float], t: Tensor[Float], r: Tensor[Float]): Tuple3[Tensor[Float], Tensor[Float], Tensor[Float]] = {
      val scoresH: Tensor[Float] = Tensor(6, dimNum)

      val gradHeadPositive = ((h - r.cmul(t)) / norm_l2(h - r.cmul(t))) * 3.0.toFloat / 9.0.toFloat
      val gradRelPositive = gradHeadPositive
      val gradTailPositive = gradHeadPositive * -1.0.toFloat

      val gradHeadNegative = gradHeadPositive * -1.0.toFloat
      val gradTailNegative = gradTailPositive * -1.0.toFloat
      val gradRelNegative = gradRelPositive * -1.0.toFloat

      scoresH.update(1, gradHeadPositive)
      scoresH.update(2, gradRelPositive)
      scoresH.update(3, gradTailPositive)

      scoresH.update(4, gradHeadNegative)
      scoresH.update(5, gradRelNegative)
      scoresH.update(6, gradTailNegative)


//      val scoresR: Tensor[Float] = Tensor(6, 8, dimNum)
//
//      val gradHeadPositive1 = ((h - r * t) / norm_l2(h - r * t)) * t * 3.0.toFloat / 9.0.toFloat
//      val gradRelPositive1 = gradHeadPositive1
//      val gradTailPositive1 = gradHeadPositive1 * -1.0.toFloat
//
//      val gradHeadNegative1 = gradHeadPositive1 * -1.0.toFloat
//      val gradTailNegative1 = gradTailPositive1 * -1.0.toFloat
//      val gradRelNegative1 = gradRelPositive1 * -1.0.toFloat
//
//      scoresR.update(1, gradHeadPositive1)
//      scoresR.update(2, gradRelPositive1)
//      scoresR.update(3, gradTailPositive1)
//
//      scoresR.update(4, gradHeadNegative1)
//      scoresR.update(5, gradRelNegative1)
//      scoresR.update(6, gradTailNegative1)
//
//
//      val scoresT: Tensor[Float] = Tensor(6, 8, dimNum)
//
//      val gradHeadPositive2 = ((h - r * t) / norm_l2(h - r * t)) * (r * -1.0.toFloat) * 3.0.toFloat / 9.0.toFloat
//      val gradRelPositive2 = gradHeadPositive2
//      val gradTailPositive2 = gradHeadPositive2 * -1.0.toFloat
//
//      val gradHeadNegative2 = gradHeadPositive2 * -1.0.toFloat
//      val gradTailNegative2 = gradTailPositive2 * -1.0.toFloat
//      val gradRelNegative2 = gradRelPositive2 * -1.0.toFloat
//
//      scoresT.update(1, gradHeadPositive2)
//      scoresT.update(2, gradRelPositive2)
//      scoresT.update(3, gradTailPositive2)
//
//      scoresT.update(4, gradHeadNegative2)
//      scoresT.update(5, gradRelNegative2)
//      scoresT.update(6, gradTailNegative2)
//
//      (scoresH, scoresR, scoresT)
      scoresH
    }
    

    val result = optim.optimize(Update, tensorTotal) //line 12  of the main algorithm
    val updateResult: Tensor[Float] = result._1

    (res, resP, resN, updateResult)
  }

  def computeInsideNode(): Tuple5[Float, Tensor[Float], Tensor[Float], Float, Float] = {
    //    println("iteration: " + iteration)

    val remoteIgnite = Ignition.localIgnite()
    val trainingTriplesCache: IgniteCache[JavaInt, TripleID] = remoteIgnite.cache("tripleIde");
    val existingTriplesCache: IgniteCache[TripleID, JavaInt] = remoteIgnite.cache("allTriplesCache")

    val entity_tmp: Tensor[Float] = entityCache.clone()
    val relation_tmp: Tensor[Float] = relationCache.clone()

    val entity_num = entityCache.size()(0)
    var relation_num = relationCache.size()(0)
    val triple_num = trainingTriplesCache.size()

    var totRes: Float = 0
    var totResPos: Float = 0
    var totResNeg: Float = 0

    var oldUpdate: Tensor[Float] = null

    for (batch <- 0 until batchesPerNode) {

      var positiveTriples: List[TripleID] = List()
      var negativeTriples: List[TripleID] = List()


      for (i <- 0 until batchsize) {
        val id = rand_max(triple_num)
        var randomEntity = rand_max(entity_num)
        val triple: TripleID = trainingTriplesCache.get(id)

        if (scala.util.Random.nextInt() % 1000 < 500) {
          while (randomEntity == triple.tail || existingTriplesCache.get(new TripleID(triple.head, randomEntity, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          positiveTriples = triple :: positiveTriples
          negativeTriples = new TripleID(triple.head, randomEntity, triple.rel) :: negativeTriples
        } else {
          while (randomEntity == triple.head || existingTriplesCache.get(new TripleID(randomEntity, triple.tail, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          positiveTriples = triple :: positiveTriples
          negativeTriples = new TripleID(randomEntity, triple.tail, triple.rel) :: negativeTriples
        }
      }

      val temp = forward2(positiveTriples, negativeTriples, entity_tmp, relation_tmp)
      totRes += temp._1
      totResPos += temp._2
      totResNeg += temp._3

      val updatedResult: Tensor[Float] = temp._4

      //      if(oldUpdate==null){
      //oldUpdate = entity_tmp.clone()
      //      }else{
      //println((entity_tmp.sub(oldUpdate)).sum())
      //println("-----------------------------")
      //println((entity_tmp-entity_tmp).sum())
      //println("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
      //oldUpdate = entity_tmp.clone()
      //      }


      //println(entity_tmp.select(1, 100)(1))

      for (i <- 1 until batchsize) {
        val triple: TripleID = positiveTriples(i - 1)
        val tensorHead = updatedResult(i)
        val tensorRel = updatedResult(i + 1 * batchsize)
        val tensorTail = updatedResult(i + 2 * batchsize)

        entity_tmp.update(triple.head + 1, tensorHead)
        entity_tmp.update(triple.tail + 1, tensorTail)
        relation_tmp.update(triple.rel + 1, tensorRel)


      }

    }
    (totRes, entity_tmp, relation_tmp, totResPos, totResNeg)
  }

}