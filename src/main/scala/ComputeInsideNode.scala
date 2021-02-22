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
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions._
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
  val learningRate:Float = 10.0f
  var batchesPerNode = Integer.parseInt(variables("batchesPerNode"))
  var updateGamma = variables("updateGamma").toBoolean
  var gamma_p = variables("gamma_p").toFloat
  var gamma_n = variables("gamma_n").toFloat
  var delta_p = variables("delta_p").toFloat
  var delta_n = variables("delta_n").toFloat
  var beta1: Float = variables("beta1").toFloat
  var beta2: Float = variables("beta2").toFloat

  var iteration =  Integer.parseInt(variables("iteration"))


  val optim = new Adadelta(learningRate)
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

  def calc_score(triple: TripleID): Float = {

    var psi: Float = 1.2.toFloat

    var head = entityCache(triple.head)
    var tail = entityCache(triple.tail)
    var rel = relationCache(triple.rel)


    var row0 = head.select(1,1) + rel.select(1,1) - tail.select(1,1)
    var row4 = head.select(1,5)+ rel.select(1,5) - tail.select(1,5)
    var a = (norm_l2(row0) + norm_l2(row4)) / 2.0

    var row1 = head.select(1,2) + tail.select(1,2) - rel.select(1,2)
    var row5 = head.select(1,6) + tail.select(1,6) - rel.select(1,6)
    var b = (norm_l2(row1) + norm_l2(row5)) / 2.0


    var row2 = rel.select(1,3) + tail.select(1,3) - head.select(1,3)
    var row6 = rel.select(1,7) + tail.select(1,7) - head.select(1,7)
    var c = (norm_l2(row2) + norm_l2(row6)) / 2.0

    var row3 = head.select(1,4) - rel.select(1,4) * tail.select(1,4)
    var row7 = head.select(1,8) - rel.select(1,8) * tail.select(1,8)
    var d = (norm_l2(row3) + norm_l2(row7)) / 2.0

    var score = ((1.5 * a + 3 * b + 1.5 * c + 3 * d) / 9.0 - psi).toFloat
    //score  = score - tanh(score)
    //println(score)
    score
  }

  def TransE_Score(triple: TripleID, entityEmbedding:Tensor[Float], relationEmbedding:Tensor[Float]): Float = { //(Tensor[Float])
   // val sampleDistance = Tensor(sample.size, 1)
//    var count = 1
//    triple.map(row => {

    val subTensor  = entityEmbedding.select  (1, triple.head)//.reshape(Array(k,1))
    val predTensor = relationEmbedding.select(1, triple.rel)//.reshape(Array(k,1))
    val objTensor  = entityEmbedding.select  (1, triple.tail)//.reshape(Array(k,1))

    val dist = (subTensor + predTensor - objTensor).norm(2)

//      sampleDistance.update(count,dist)
//      count += 1
//    })
//    (sampleDistance)
    (dist)
  }

  def loss(posScore: Tensor[Float], negScore: Tensor[Float]): Tuple3[Float, Float, Float] = {
    var lambda_pos: Float = 1.0.toFloat// gamma_p - delta_p
    var lambda_neg: Float = 1.0.toFloat//gamma_n - delta_n
    //println("positive score " + posScore)
    //println("positive loss before " + ((posScore - lambda_pos) * (-1.0) + margin))

    println("class: " + lambda_pos.getClass)


    var pos_loss = (ReLU().forward(((posScore- lambda_pos)* -1.0.toFloat)+ margin)).sum
    var neg_loss = (ReLU().forward((negScore - lambda_neg)+ margin)).sum

    //var pos_loss: Double = max(0.0, ((posScore - lambda_pos) * (-1.0) + margin))
    //println("positive loss " + pos_loss)

    //var neg_loss: Double = max(0.0, ((negScore - lambda_neg)  + margin))
    var loss = beta1 * pos_loss + beta2 * neg_loss

    (loss, pos_loss, neg_loss)
  }


  def forward(trueTriple: TripleID, falseTriple: TripleID, entityEmbedding: Tensor[Float], relationEmbedding:Tensor[Float]): Tuple3[Float, Float, Float] = {
    var res: Float = 0
    var resP: Float = 0
    var resN: Float = 0
    //var posScore = calc_score(trueTriple) //println("positive score " + posScore)

    //var negScore = calc_score(falseTriple)

    //var posScore = TransE_Score(trueTriple, entityEmbedding, relationEmbedding)
    //println("positive score " + posScore)
    //var negScore = TransE_Score(falseTriple, entityEmbedding, relationEmbedding)



    //println("positive loss " + posLoss)
    //if (posLoss != 0)
    //gradient(trueTriple, true, totLoss, entity_tmp, relation_tmp)
    //if (negLoss != 0)
    //  gradient(falseTriple, false, totLoss, entity_tmp, relation_tmp)
    //if (posLoss - negLoss + 1 > 0)
    //  gradient(trueTriple, true, totLoss, entity_tmp, relation_tmp)
    //  gradient(falseTriple, false, totLoss, entity_tmp, relation_tmp)


    //println("Epoch: " + epoch)
    //Function for updating the embedding
    def Update(x: Tensor[Float]) = {
      var posScore = calc_score(trueTriple)
      //println("positive score " + posScore)
      var negScore = calc_score(falseTriple)
      var (totLoss, posLoss, negLoss) = loss(posScore, negScore)
      res = totLoss
      resP = posLoss
      resN = negLoss
      (totLoss , x)
    }

    //if((positiveDistance.sum * gamma) > negativeDistance.sum){
    optim.optimize(Update, entityEmbedding) //line 12  of the main algorithm

    (res, resP, resN)
  }

  def computeInsideNode(): Tuple5[Float,Tensor[Float], Tensor[Float], Float, Float] = {
    println("iteration: "+iteration)
    val remoteIgnite = Ignition.localIgnite()
    var trainingTriplesCache: IgniteCache[JavaInt, TripleID] = remoteIgnite.cache("tripleIde");
    var existingTriplesCache: IgniteCache[TripleID, JavaInt] = remoteIgnite.cache("allTriplesCache")

    var entity_tmp: Tensor[Float] = entityCache.clone() //Map.empty[Int, DenseMatrix[Float]] /

    var relation_tmp: Tensor[Float] = relationCache.clone()// Map.empty[Int, DenseMatrix[Double]]

    //var entity_tmp: Map[Int, DenseMatrix[Double]] = entityCache.clone() //Map.empty[Int, DenseMatrix[Double]] /
    //var relation_tmp: Map[Int, DenseMatrix[Double]] = relationCache.clone()// Map.empty[Int, DenseMatrix[Double]]
    //println(entity_tmp(1))




    var entity_num = entityCache.size()(0)
    var relation_num = relationCache.size()(0)
    var triple_num = trainingTriplesCache.size()

    var totRes: Float = 0
    var totResPos: Float = 0
    var totResNeg: Float = 0
    for (batch <- 0 until batchesPerNode) {
      for (i <- 0 until batchsize) {
        var id = rand_max(triple_num)
        //trainingTriplesCache //todo:here get a batch instead of only one item

        var randomEntity = rand_max(entity_num)
        var triple: TripleID = trainingTriplesCache.get(id)

        if (scala.util.Random.nextInt() % 1000 < 500) {
          while (randomEntity == triple.tail || existingTriplesCache.get(new TripleID(triple.head, randomEntity, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          //if (!entity_tmp.contains(randomEntity)) {
            val temp = forward(triple, new TripleID(triple.head, randomEntity, triple.rel), entity_tmp, relation_tmp)
            totRes += temp._1
            totResPos += temp._2
            totResNeg += temp._3
          //}
        } else {
          while (randomEntity == triple.head || existingTriplesCache.get(new TripleID(randomEntity, triple.tail, triple.rel)) != null) {
            randomEntity = rand_max(entity_num)
          }
          //if (!entity_tmp.contains(randomEntity)) {
            val temp = forward(triple, new TripleID(randomEntity, triple.tail, triple.rel), entity_tmp, relation_tmp)
            totRes += temp._1
            totResPos += temp._2
            totResNeg += temp._3
          //}
        }
      //} for the batch loop
    }
    //println(entity_tmp(1))
    (totRes, entity_tmp, relation_tmp, totResPos, totResNeg)
  }

}