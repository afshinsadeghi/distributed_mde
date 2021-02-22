package Tryouts

import org.apache.log4j._ // for log messages
import java.io._
import scala.util._ //for sort
import scala.util.Random
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.rdd.RDD //for RDD
import com.intel.analytics.bigdl.nn._ //for ReLU
import com.intel.analytics.bigdl.optim._ //for Optimizer
import com.intel.analytics.bigdl.tensor.Tensor // for Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat //for Optimizer(learning rate), Tensor[Float]
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.intel.analytics.bigdl.utils.T //for T
import scala.collection.mutable

object EvaluateTransEVersion2 {
  
  def Distance(row:Row, entityListLength: Int, embedding:Array[Array[Float]], indices: Array[Array[Int]], shape: Array[Int], L:Int): (Float) = {
    val subTensor  = Tensor.dense(Tensor.sparse(indices, embedding(row(0).toString.toInt-1), shape))
    val predTensor = Tensor.dense(Tensor.sparse(indices, embedding(row(1).toString.toInt + entityListLength-1), shape))
    val objTensor  = Tensor.dense(Tensor.sparse(indices, embedding(row(2).toString.toInt-1), shape))
    
    L match {
      case 1 => ((subTensor + predTensor - objTensor).abs).sum //L1
      case _ => ((subTensor.pow(2) + predTensor.pow(2) - objTensor.pow(2)).sqrt.abs).sum //L2    
    }     
  }
  
  def Rank(row: Row, score:Float, changedSample:Array[Row], trainSet: Array[Row], entityListLength: Int, embedding:Array[Array[Float]], indices: Array[Array[Int]], shape: Array[Int], L:Int)
         : (Float) = {
    
    //val filterNegativeSample = (changedSample).diff(trainSet) :+ row 
      
    val rawScore = changedSample.map(row => {Distance(row, entityListLength, embedding, indices, shape, L)}) //for raw hit@10
    //val filterScore = filterNegativeSample.map(row => {Distance(row, entityListLength, embedding, L)}) //for filter hit@10
     
    Sorting.quickSort(rawScore)
    //Sorting.quickSort(filterScore)
      
    val rawRank = rawScore.indexOf(score)
    //val filterRank = filterScore.indexOf(score)
    
    (rawRank)
  }
  
  def main (args: Array[String]){
    val t1 = System.nanoTime
   
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 

    val datasetName = "Kingship" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val spark = SparkSession.builder.master("local").appName("Evaluation").config("spark.some.config.option","some-value").getOrCreate()
    val sc = spark.sparkContext  
    
    val trainSet     = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd.collect
    val testSet      = spark.read.option("sep", "\t").csv(resoiurceDir + "test.tsv").rdd.collect
    val embedding    = spark.read.option("sep", "\t").csv(resoiurceDir + "trainedEmbedding.tsv").rdd.map(r1 => {r1.toSeq.toArray.map(_.asInstanceOf[String]).map(r2 => {(r2.toFloat)})}).collect
    val entityRDD   = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[String].toInt)    
    val summary      = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    val k = embedding(0).length    
    
    val L = 1
    val rows = Array.fill(k){0}
    val cols = rows.zipWithIndex.map(elem => elem._2)
    val indices = Array(rows, cols)    
    val shape = Array(1, k)
    
    var hit = 0
    
    testSet.foreach(row => {
      val score = Distance(row, entityListLength, embedding, indices, shape, L)
      
      val changeHead = entityRDD.map(r => Row(r, row(1), row(2))).collect //for raw hit@10 with head corruption      
      val changeTail = entityRDD.map(r => Row(row(0), row(1), r)).collect //for raw hit@10 with tail corruption
      
      val rawRankHead = Rank(row, score, changeHead, trainSet, entityListLength, embedding, indices, shape, L)
      val rawRankTail = Rank(row, score, changeTail, trainSet, entityListLength, embedding, indices, shape, L)
      
      val avgRank = (rawRankHead + rawRankTail)/2
      
      if(avgRank <= 100){
        hit += 1
      }
      
      println(avgRank)
    })
    println("hit@100 : " + hit + " out of " + testSet.length + " numbers of test dataset.")
    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}