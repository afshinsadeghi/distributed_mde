package Tryouts

import org.apache.log4j._
import java.io._
import scala.util.Random
import org.apache.spark.sql._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.sql.types._ //for variable type

object MiniBatch {
  
  
  
  def main (args: Array[String]){
    val t1 = System.nanoTime
      
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 

    val datasetName = "Tasneem" //Folder name of the dataset
    val resourceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val schema = new StructType().add("_c0",StringType,true).add("_c1",StringType,true).add("_c2",StringType,true)
    
    val batchSize = 5
    
    val spark = SparkSession.builder.master("local").appName("TrainTransE").config("spark.some.config.option","some-value").getOrCreate()
    val sc    = spark.sparkContext
    
    val trainSet   = spark.read.option("sep", "\t").csv(resourceDir + "originalTrain.tsv").rdd.collect
    val trainSize  = trainSet.length

    var main  = trainSet
    var batch: Array[Row]  = Array()
    
    var batchLoop = 0    
    (trainSize%batchSize) match {
      case 0 => batchLoop = (trainSize/batchSize)
      case _ => batchLoop = (trainSize/batchSize) + 1
    }
    
    for(count <- 1 to batchLoop){
      println("\nIn count: " + count)      
      batch = main.take(batchSize)
      main.foreach(println)
      println()
      batch.foreach(println)
      main = main.diff(batch)
    }

    /*
     * https://stackoverflow.com/questions/32932229/how-to-randomly-sample-from-a-scala-list-or-array
     * for shuffle array
     */
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}