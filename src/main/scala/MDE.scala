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
import java.nio.file.Paths

import org.apache.ignite.IgniteSystemProperties
import jdk.nashorn.internal.ir.ForNode
import java.util.UUID
import org.apache.spark.mllib.linalg.Matrix

import com.intel.analytics.bigdl.tensor.Tensor

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.utils.T

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat


import org.apache.ignite.messaging.MessagingListenActor

import scala.collection.JavaConversions._
import org.apache.ignite.configuration.CacheConfiguration
import org.apache.ignite.cache.CacheMode
import org.apache.ignite.configuration.IgniteConfiguration

import scala.collection.mutable.Map

object MDE extends App {
  private val CONFIG = "src/main/scala/example-ignite.xml"
  // Ignition.setClientMode(true)

  /** Cache names. */
  private final val NAME = MDE.getClass.getSimpleName
  private final val entityCache = "Entity2IDcachee"
  private final val relationCache = "Relation2IDcachee"
  private final val tripleIDs = "tripleIde"
  private final val filterCache = "allTriplesCache"
  private final val testCache = "testTriplecachee"

  /** Type alias. */
  type Cache = IgniteCache[JavaString, JavaInt]
  type CacheFilter = IgniteCache[TripleID, Int]
  type TriplesCache = IgniteCache[Int, TripleID]

  val dimension = 20
  val margin = 1.0.toFloat
  val method = 0
  val nbatches = 50 //100
  val nepoch = 2000
  //var lr_rate = 0.0000001//0.00001//0.001//0.0001//0.1// 0.01 too big
  //var lr_rate = 0.001//0.01 // for adam learning rate 0.001
  var lr_rate =  0.01.toFloat //for sgd
  var entity_num = 0
  var relation_num = 0
  var testTriples = 0

  var gamma_p: Float = 1.9.toFloat//1.9   // wn18 50 1,9 1,9 2 1    fb15 200 10 13 1 1   wn18rr
  var gamma_n: Float = 1.9.toFloat//1.9  //FB15k-237  100, 9, 9, 1   // WN18RR 50, 2, 2, 5 and 1
  var delta_p: Float = 0.0.toFloat
  var delta_n: Float = 0.0.toFloat
  var beta1: Float = 1.0.toFloat
  //var beta1: Float = 0.9
  var beta2: Float = 1.0.toFloat
  //var beta2: Float = 0.999
  val update_gamma = false
  var hitAt = 10



  scalar(CONFIG) {
    //println(MaxValue)
    println("----------------------------------------------------")
    var cfgTriples = new CacheConfiguration[Int, TripleID]
    cfgTriples.setName(tripleIDs)
    cfgTriples.setCacheMode(CacheMode.REPLICATED)
    var tripleIdCache = Ignition.ignite().getOrCreateCache(cfgTriples)

    var cfg_all_Triples = new CacheConfiguration[TripleID, Int]
    cfg_all_Triples.setName(filterCache)
    cfg_all_Triples.setCacheMode(CacheMode.REPLICATED)
    var filtercache = Ignition.ignite().getOrCreateCache(cfg_all_Triples)

    var cfg_test_Triples = new CacheConfiguration[Int, TripleID]
    cfg_test_Triples.setName(testCache)
    cfg_test_Triples.setCacheMode(CacheMode.REPLICATED)
    var testTripleIDsCache = Ignition.ignite().getOrCreateCache(cfg_test_Triples)

    val entitycache = createCache$[JavaString, JavaInt](entityCache, indexedTypes = Seq(classOf[JavaString],
      classOf[JavaInt]))
    val relationcache = createCache$[JavaString, JavaInt](relationCache, indexedTypes = Seq(classOf[JavaString],
      classOf[JavaInt]))

    println(java.time.LocalDateTime.now())

    //define call as an ignite variable that is callable
    var calls: Collection[IgniteCallable[Int]] = new ArrayList

    calls.add(() => {
      val smtr = dataStreamer$[JavaString, Int](entityCache, 2048)
      var enity_num: Int = 0
      val stream: InputStream = getClass.getResourceAsStream("/data/WN18/entity2id.txt")
      for (line <- Source.fromInputStream(stream).getLines()) {
        enity_num += 1
        val res = line.split("\t")
        smtr.addData(res(0), Integer.valueOf(res(1)))
      }
      smtr.close(false)

      println("entity number is " + enity_num)
      enity_num
    })


    calls.add(() => {
      val smtr = dataStreamer$[JavaString, Int](relationCache, 2048)
      var rel_number: Int = 0
      val stream: InputStream = getClass.getResourceAsStream("/data/WN18/relation2id.txt")
      for (line <- Source.fromInputStream(stream).getLines()) {
        rel_number += 1
        val res = line.split("\t")
        smtr.addData(res(0), Integer.valueOf(res(1)))
      }
      smtr.close(false)
      println("rel number is " + rel_number)
      rel_number
    })
    println("epochs number is " + nepoch)
    var compute1 = Ignition.ignite().compute().call(calls)
    println("Calls 1 computed")
    println(java.time.LocalDateTime.now())
    entity_num = entitycache.size()
    relation_num = relationcache.size()


    //var entityEmbed = scala.collection.mutable.Map[Int, DenseMatrix[Double]]()
    //var relationEmbed = scala.collection.mutable.Map[Int, DenseMatrix[Double]]()

    val (entityEmbed, relationEmbed) = CreateEmbedding(entity_num, relation_num, dimension)

    //createEmbedMatrixMap(entityEmbed, entity_num)
    //createEmbedMatrixMap(relationEmbed, relation_num)




    println("embeddings initiated " + java.time.LocalDateTime.now())
    val tripleNr = populateTrainingTriplesCache("/data/WN18/train.txt", tripleIDs, filterCache, entitycache,
      relationcache)
    println(" Train triples read")
    val validTriples = populateTrainingTriplesCache("/data/WN18/valid.txt", tripleIDs, filterCache, entitycache,
      relationcache)
    println(" Valid triples read")
    testTriples = populateTestCache("/data/WN18/test.txt", testCache, entitycache, relationcache)
    println(" Test triples read")
    println("Triple cache created: " + java.time.LocalDateTime.now())
    entitycache.destroy()
    relationcache.destroy()
    println("rdim nur " + dimension)
    println("lr_rate is " + lr_rate)
    println("margin is " + margin)

    val startTime = java.time.LocalDateTime.now()
    train(filtercache, tripleIdCache.size(), entityEmbed, relationEmbed)
    println("Training started at: " + startTime + " and ended at: " + java.time.LocalDateTime.now())
    //entityEmbed.clear()
    //relationEmbed.clear()
    val testStart = java.time.LocalDateTime.now()

    // computeTest(entityEmbed, relationEmbed, testTripleIDsCache, filtercache)
    println("Test started at: " + testStart + " and ended at:" + java.time.LocalDateTime.now())

  }

  private def triplesCache[K, V]: IgniteCache[K, V] = cache$[K, V](tripleIDs).get

  private def filteringCache[K, V]: IgniteCache[K, V] = cache$[K, V](filterCache).get

  def writeFile(filename: String, embeddings: Map[Int, DenseVector[Float]], eSize: Int): Unit = {
    val bw = new BufferedWriter(new FileWriter(new File(filename)))
    for (rowId <- 0 until eSize) {
      var current = embeddings(rowId)
      for (i <- 0 until dimension)
        bw.write(current(i) + " ")
      bw.write("\n")
    }
    bw.close()
  }



  def rand(min: Float, max: Float): Float = {
    val RAND_MAX = 0x7fffffff
    var r = min + (max - min) * scala.util.Random.nextInt() / (RAND_MAX + 1.0)
    r.toFloat
  }

  def normal(x: Float, miu: Float, sigma: Float): Float = {
    var r = 1.0 / sqrt(2 * math.Pi) / sigma * exp(-1 * (x - miu) * (x - miu) / (2 * sigma * sigma))
    r.toFloat
  }

  def randn(miu: Float, sigma: Float, min: Float, max: Float): Float = {
    var x, y, dScope: Float = 0.0.toFloat
    do {
      x = rand(min, max)
      y = normal(x, miu, sigma)
      dScope = rand(0.0.toFloat, normal(miu, miu, sigma))
    } while (dScope > y)
    return x
  }

  def vec_len(a: DenseVector[Float]): Float = {
    var res: Float = 0
    for (i <- 0 until a.size)
      res += a(i) * a(i)
    res = sqrt(res)
    return res;
  }

  def norm(a: DenseVector[Float]): DenseVector[Float] = {
    var x: Float = vec_len(a)
    if (x > 1)
      for (i <- 0 until a.size)
        a(i) /= x
    return a
  }

  def norm_l2(a: DenseVector[Float]): Float = {
    var sum: Float = 0.0.toFloat
    for (i <- 0 until a.size)
      sum = sum + (a(i) * a(i))
    sum = sqrt(sum)
    return sum
  }

  def renorm_l2(a: DenseVector[Float]): DenseVector[Float] = {
    var sum: Float = 0.0.toFloat
    for (i <- 0 until a.size)
      sum = sum + (a(i) * a(i))
    sum = sqrt(sum)
    for (i <- 0 until a.size)
      a(i) /= sum
    return a
  }


  def populateTrainingTriplesCache(filePath: String, Name1: String, Name2: String, ecache: Cache, rcache: Cache): Int
  = {
    var lineNr: Int = 0
    val str1 = dataStreamer$[Int, TripleID](Name1, 2048)
    val str2 = dataStreamer$[TripleID, Int](Name2, 2048)
    val stream: InputStream = getClass.getResourceAsStream(filePath)
    for (line <- Source.fromInputStream(stream).getLines()) {
      val res = line.split("\t")
      var h = ecache.get(res(0))
      var t = ecache.get(res(1))
      var r = rcache.get(res(2))

      str1.perNodeParallelOperations()
      str1.addData(lineNr, new TripleID(h, t, r))
      str2.perNodeParallelOperations()
      str2.addData(new TripleID(h, t, r), 1)
      lineNr += 1
    }
    str1.close(false)
    str2.close(false)
    lineNr
  }

  def populateTestCache(filePath: String, Name: String, ecache: Cache, rcache: Cache): Int = {
    val smtr = dataStreamer$[JavaInt, TripleID](Name, 2048)
    println(filePath)
    var lineNr = 0
    val stream: InputStream = getClass.getResourceAsStream(filePath)
    var lines: Iterator[String] = Source.fromInputStream(stream).getLines()
    for (line <- lines) {
      val res = line.split("\t")
      var h = ecache.get(res(0))
      var t = ecache.get(res(1))
      var r = rcache.get(res(2))

      smtr.addData(lineNr, new TripleID(h, t, r))
      lineNr += 1
    }
    smtr.close(false)
    return lineNr
  }

  def update_limits(loss_p_per_triple: Float, loss_n_per_triple: Float) {
    if (loss_p_per_triple * beta1 < 0.1 && delta_p < gamma_p)
      delta_p += 0.1.toFloat
    if (loss_n_per_triple > 0.05 && delta_n < gamma_n - 0.1)
      delta_n += 0.1.toFloat
    else if (loss_n_per_triple * beta2 < 0.1 && delta_n > 0.1 && delta_n <= delta_p - 0.1)
      delta_n -= 0.1.toFloat
  }

  /**
   *
   *
   * @param cacheF seems not in use
   * @param tripleSize
   * @param entityEmbed
   * @param relationEmbed
   */
  def train(cacheF: CacheFilter, tripleSize: Int, entityEmbed: Tensor[Float], relationEmbed: Tensor[Float]) {

    import scalax.chart.api._
    val series = new XYSeries("Loss vs Epoch")
    val chart = XYLineChart(series)
    chart.show()

    val ignite = Ignition.ignite()
    val clusterSize = ignite.cluster().nodes().size()
    var batchesPerNode = 0
    if (clusterSize < 10) {
      batchesPerNode = Math.ceil(10 / clusterSize).toInt
    } else {
      batchesPerNode = 1
    }
    //println("cluster size is : " + clusterSize)
    val batchsize: Int = tripleSize / nbatches
    var variables = Map[String, String]()
    variables += ("batchsize" -> batchsize.toString())
    variables += ("dim_num" -> dimension.toString())
    variables += ("lr_rate" -> lr_rate.toString())
    variables += ("margin" -> margin.toString())
    variables += ("batchesPerNode" -> batchesPerNode.toString())
    variables += ("updateGamma" -> update_gamma.toString())
    variables += ("gamma_p" -> gamma_p.toString())
    variables += ("gamma_n" -> gamma_n.toString())
    variables += ("delta_p" -> delta_p.toString())
    variables += ("delta_n" -> delta_n.toString())
    variables += ("beta1" -> beta1.toString())
    variables += ("beta2" -> beta2.toString())


    var res: Float = 0
    //var resPos: Float = 0
    //var resNeg: Float = 0
    var tmpLoss: Float = 0

    val bw = new BufferedWriter(new FileWriter("trainingResults.txt"))
    bw.write("dimSize = " + dimension + "\n")
    bw.write("lr_rate = " + lr_rate + "\n")
    bw.write("marign = " + margin + "\n")
    bw.write("nBatches = " + nbatches + "\n")
    bw.write("~~~~~~~~~~~~~~~~~~~~~~~~ Training results ~~~~~~~~~~~~~~~~~~~~~~~~" + "\n")
    bw.write("Starting training at  " + java.time.LocalDateTime.now())
    for (epoch <- 0 until nepoch) {
      variables += ("iteration" -> String.valueOf(epoch))

      //if (epoch == 5)
      //  lr_rate =lr_rate / 10

      //if (epoch == 20)
      //  lr_rate = lr_rate / 10

      //if (epoch == 50)
      //  lr_rate = lr_rate / 10

      bw.write("iteration = " + epoch + "\n")
      res = 0
      var resPos: Float = 0
      var resNeg: Float = 0
      for (batch <- 0 until 10) {
        var calls: Collection[IgniteCallable[Tuple5[Float, Tensor[Float], Tensor[Float], Float, Float]]] = new ArrayList
        for (i <- 0 until clusterSize) {
          calls.add(() => {
            //val nodeid = ignite$.cluster().localNode.id
            val inNode = new ComputeInsideNode(entityEmbed, relationEmbed, variables)
            val singlebatchRes = inNode.computeInsideNode()
            singlebatchRes
          })
        }
        var batchCompute = ignite.compute().call(calls)
        calls.clear()
        for (batchLoss <- batchCompute.asScala) {
          res += batchLoss._1
          var entityEmbed_ =  (batchLoss._2).clone() //entity gradients
          var relationEmbed_ =  (batchLoss._3).clone() //relation gradients
          resPos += batchLoss._4
          resNeg += batchLoss._5
        }

      }
      var loss_per_triple = res / tripleSize
      var loss_p_per_triple = resPos / tripleSize
      var loss_n_per_triple = resNeg / tripleSize

      if (update_gamma) {
        update_limits(loss_p_per_triple, loss_n_per_triple)
      }
      bw.write("epoch " + epoch + " loss: " + loss_per_triple + "\n")
      println("epoch " + epoch + " loss: " + res)
      println("epoch " + epoch + " loss_per_triple: " + loss_per_triple)
      println("epoch " + epoch + " loss_positive_per_triple: " + loss_p_per_triple)
      println("epoch " + epoch + " loss_negative_per_triple: " + loss_n_per_triple)
      println("______________________________________________________")

      if (epoch > 950 && tmpLoss > res) {
        tmpLoss = res
        //        writeFile("finalEntEmbed.txt", entityEmbed, entity_num)
        //        writeFile("finalRelEmbed.txt", relationEmbed, relation_num)
      }



        swing.Swing onEDT {
          series.add(epoch,loss_per_triple)
        }


    }
    bw.write("Finished training at  " + java.time.LocalDateTime.now())
    bw.close()
    println()
    println("embeddings saved at loss: " + tmpLoss)
    println()

  }


  def CreateEmbedding (entityListLength:Int, relationListLength:Int, dimension:Int) : (Tensor[Float], Tensor[Float]) = {

    val entityEmbedding   = Tensor(entityListLength,8,dimension).rand(-6 / Math.sqrt(dimension), 6 / Math.sqrt(dimension))
    //val entityEmbedding   = Tensor(entityListLength,8,dimension).rand(-6 / Math.sqrt(dimension), 6 / Math.sqrt(dimension))
    val relationEmbedding = Tensor(relationListLength,8, dimension).rand(-6 / Math.sqrt(dimension), 6 / Math.sqrt(dimension))
    (entityEmbedding, relationEmbedding)
  }



  //  def computeTest(entityCache: Map[Int, DenseMatrix[Float]], relationCache: Map[Int, DenseMatrix[Float]],
  //  testT: TriplesCache, filterC: CacheFilter) {
  //    var sumRank4head = 0
  //    var sumRank4headFiltered = 0
  //    var sumRank4tail = 0
  //    var sumRank4tailFiltered = 0
  //    var countHits4head = 0
  //    var countHits4tail = 0
  //    var countHits4headFiltered = 0
  //    var countHits4tailFiltered = 0
  //    var variables = Map[String, String]()
  //    variables += ("dim_num" -> n.toString())
  //    variables += ("hit" -> hitAt.toString())
  //    var start = java.time.LocalDateTime.now()
  //    var testcalls: Collection[IgniteCallable[Map[String, Int]]] = new ArrayList
  //    var testChunks = (0 until testTriples).toList.grouped(1000)
  //    testChunks.foreach(chunk => {
  //      testcalls.add(() => {
  //        val nodeid = ignite$.cluster().localNode.id
  //        // println("computing triples at node: " + nodeid)
  //        var inNode = new Test(entityCache, relationCache, variables)
  //        var singletestRes = inNode.testSingleTriple(chunk)
  //        singletestRes
  //      })
  //    })
  //    var testRes = Ignition.ignite().compute().call(testcalls)
  //    for (testresult <- testRes.asScala) {
  //      countHits4head += testresult("countHits4head")
  //      countHits4headFiltered += testresult("countHits4headFiltered")
  //      sumRank4head += testresult("sumRank4head")
  //      sumRank4headFiltered += testresult("sumRank4headFiltered")
  //      countHits4tail += testresult("countHits4tail")
  //      countHits4tailFiltered += testresult("countHits4tailFiltered")
  //      sumRank4tail += testresult("sumRank4tail")
  //      sumRank4tailFiltered += testresult("sumRank4tailFiltered")
  //    }
  //    var finish = java.time.LocalDateTime.now()
  //    println("start: " + start + " end:" + finish)
  //    val bw = new BufferedWriter(new FileWriter("testResults.txt"))
  //    bw.write("dim " + n + ", lr_rate " + lr_rate + ", margin " + margin)
  //    bw.write("Hit@" + hitAt + " raw for head is: " + countHits4head * 100 / testTriples + "% \n")
  //    bw.write("Hit@" + hitAt + " raw for tail is: " + countHits4tail * 100 / testTriples + "% \n")
  //    bw.write("Hit@" + hitAt + " filtered for head is: " + countHits4headFiltered * 100 / testTriples + "% \n")
  //    bw.write("Hit@" + hitAt + " filtered for tail is: " + countHits4tailFiltered * 100 / testTriples + "% \n")
  //    bw.write("*************************************************************************")
  //    bw.write(" MR for head: " + sumRank4head / testTriples + " \n")
  //    bw.write(" MR for tail: " + sumRank4tail / testTriples + " \n")
  //    bw.write(" MR for head filtered: " + sumRank4headFiltered / testTriples + " \n")
  //    bw.write(" MR for tail filtered: " + sumRank4tailFiltered / testTriples + " \n")
  //    bw.write("*************************************************************************")
  //    bw.close()
  //
  //    println("dim " + n + ", lr_rate " + lr_rate + ", margin " + margin)
  //    println("Hit@" + hitAt + " raw for head is: " + Math.round(countHits4head * 100 / testTriples) + "% ")
  //    println("Hit@" + hitAt + " raw for tail is: " + Math.round(countHits4tail * 100 / testTriples) + "% ")
  //    println("Hit@" + hitAt + " filtered for head is: " + Math.round(countHits4headFiltered * 100 / testTriples) +
  //    "% ")
  //    println("Hit@" + hitAt + " filtered for tail is: " + Math.round(countHits4tailFiltered * 100 / testTriples) +
  //    "% ")
  //    println("*************************************************************************")
  //    println(" MR for head: " + Math.round(sumRank4head / testTriples))
  //    println(" MR for tail: " + Math.round(sumRank4tail / testTriples))
  //    println(" MR for head filtered: " + Math.round(sumRank4headFiltered / testTriples))
  //    println(" MR for tail filtered: " + Math.round(sumRank4tailFiltered / testTriples))
  //    println("*************************************************************************")
  //  }
}

class TripleID(var head: Int, var tail: Int, var rel: Int)

class readDataset() {

}

