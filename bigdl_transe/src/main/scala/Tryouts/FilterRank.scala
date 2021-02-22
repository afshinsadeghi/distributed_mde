package Tryouts

import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import scala.util._
import org.apache.spark.sql._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import org.apache.spark.sql.types._
import scala.BigDecimal
import scala.collection.Seq

object FilterRank {
  def main (args: Array[String]){
    val t1 = System.nanoTime
    val L  = 1
    
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 
    
    val datasetName  = "Tasneem" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val schema =  new StructType().add("_c0",StringType,true).add("_c1",StringType,true).add("_c2",StringType,true)
    
    val spark = SparkSession.builder.master("local").appName("Evaluation").config("spark.some.config.option","some-value").getOrCreate()
    val sc    = spark.sparkContext  
    
    val trainSet          = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv")//.rdd.collect
    val testSet           = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd
    val entityEmbedding   = spark.read.option("sep", "\t").csv(resoiurceDir + "entityEmbedding.tsv").rdd.map(r1 => {r1.toSeq.toArray.map(_.asInstanceOf[String]).map(r2 => {BigDecimal(r2.toDouble)})}).collect
    val relationEmbedding = spark.read.option("sep", "\t").csv(resoiurceDir + "relationEmbedding.tsv").rdd.map(r1 => {r1.toSeq.toArray.map(_.asInstanceOf[String]).map(r2 => {BigDecimal(r2.toDouble)})}).collect
    val entityRDD         = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[String].toInt)    
    val summary           = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    
    trainSet.show
    println()
    
    testSet.take(1).foreach(row => {
      println("Row: " + row + "\n")
      val changeHeadRDD = entityRDD.map(r => Row(row(0).toString, row(1).toString, r.toString))
      
      val changeHeadDF = spark.createDataFrame(changeHeadRDD, schema)
      changeHeadDF.show
      println()
      val changeHead = changeHeadDF.exceptAll(trainSet).rdd.collect :+ row
      changeHead.foreach(println)
      
      val tsrdd = trainSet.rdd
      val chrdd = changeHeadRDD
      val finalrdd = (changeHeadRDD.subtract(tsrdd)).union(sc.parallelize(Seq(row)))
      println()
      finalrdd.foreach(println)

    })
    
    
    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s") 
  }
  
}