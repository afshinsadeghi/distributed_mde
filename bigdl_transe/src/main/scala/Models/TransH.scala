package Models

import org.apache.log4j._
import java.io._
import scala.util.Random
import org.apache.spark.sql._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import scala.math._

object TransH {
  
  def main (args: Array[String]){
    val t1 = System.nanoTime
   
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 

    val datasetName = "Tasneem" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val spark = SparkSession.builder.master("local").appName("TrainTransE").config("spark.some.config.option","some-value").getOrCreate()
    
    val dataset   = spark.read.option("sep", "\t").csv(resoiurceDir + "test.tsv").withColumnRenamed("_c0", "head").withColumnRenamed("_c1", "relation").withColumnRenamed("_c2", "tail")
    dataset.show
    
    import spark.implicits._    
    
    val df1 = dataset.drop("relation")
    
    val df2 = df1.groupBy("head").count().sort($"count".desc).withColumnRenamed("count", "hpt")
    df2.show
    
    val df3 = df1.groupBy("tail").count().sort($"count".desc).withColumnRenamed("count", "tph")
    df3.show
    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}