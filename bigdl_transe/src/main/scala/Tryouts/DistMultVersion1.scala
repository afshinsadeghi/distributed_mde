package Tryouts

import org.apache.log4j._ //for log messages
import java.io._ //for Printwriter
import scala.util.Random
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.rdd.RDD //for RDD
import com.intel.analytics.bigdl.nn._ //for ReLU
import com.intel.analytics.bigdl.optim._ //for Optimizer
import com.intel.analytics.bigdl.utils.{Shape, T, Table}
import com.intel.analytics.bigdl.tensor._ // for Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat //for Optimizer(learning rate), Tensor[Float]
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.nn.keras.{Sequential, Merge, InputLayer}

object DistMultVersion1 {
  
  def main (args: Array[String]){
    
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 
    
    val t1 = System.nanoTime
   
    val datasetName = "WN18" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"

    val spark = SparkSession.builder.master("local").appName("TrainTransE").config("spark.some.config.option","some-value").getOrCreate()

    val tensor1 = Tensor(3,4).fill(2)
    val tensor2 = Tensor(T(T(5,6,7,8),
                           T(1,2,3,4),
                           T(0,1,0,1)))
    val tensor3 = Tensor(T(T(2,1,3,1),
                           T(1,2,3,1),
                           T(-1,1,-1,1)))
    
    
    println(tensor1)
    //println(tensor2)
    println(tensor3)
    
    val score = tensor1.map(tensor3, (a, b) => a*b)//.map(tensor3, (c, d) => c*d)
   println(score)
    
   /*val mlp = DotProduct()
   println("input:")
   println(tensor1)
   println(tensor3)
   println("output:")
   println(mlp.forward(T(tensor1, tensor3))) */
   
   println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}