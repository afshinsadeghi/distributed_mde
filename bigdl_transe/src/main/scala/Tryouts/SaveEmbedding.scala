package Tryouts

import org.apache.log4j._ // for log messages
import org.apache.spark.sql._ //for Row, SparkSession
import com.intel.analytics.bigdl.tensor.Tensor // for Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat //for Optimizer(learning rate), Tensor[Float]
import java.io._

object SaveEmbedding {
  
  def main (args: Array[String]){
    
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 

    val t1 = System.nanoTime
    val entityLength = 10
    val relationLength = 5
    val k = 5
    
    val datasetName = "Kingship" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    val fileDir = resoiurceDir + "trainedEmbedding.tsv"
    
    val embedding = Tensor((entityLength+relationLength),k).rand(-5, 5)
    println(embedding)
    
    val printWriter = new PrintWriter(new File(fileDir))    
    
    for(index <- 1 to embedding.size(1)){
      printWriter.write(embedding.apply(index).toArray.mkString("\t")+"\n")
    }    
    printWriter.close
    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }  
}