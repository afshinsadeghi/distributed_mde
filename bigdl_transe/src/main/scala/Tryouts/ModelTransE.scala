package Tryouts

import scala.util.Random
import org.apache.log4j._ //for log messages
import org.apache.spark._ //for SparkContext
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.rdd.RDD //for RDD
import com.intel.analytics.bigdl.nn._ //for layers
import com.intel.analytics.bigdl.utils._ //for T, Engine
import com.intel.analytics.bigdl.optim._ //for optimizer
import com.intel.analytics.bigdl.tensor._ //for Tensor
import com.intel.analytics.bigdl.dataset.Sample //for Sample
import com.intel.analytics.bigdl.numeric.NumericFloat


object ModelTransE {

    def NegativeSample(row:Row, entityList:Array[Object], entityListLength: Int): (Row) = {
    println(row)
    val sub  = row.get(0).asInstanceOf[Integer]
    val pred = row.get(1).asInstanceOf[Integer] + entityListLength
    val obj  = row.get(2).asInstanceOf[Integer]    
    
    if(Random.nextInt(2) == 0){ //change head
      val subList = entityList.diff(Array(sub)) //without the subject
      //val subList = entityList //entire entityList
      val negSub =  subList(Random.nextInt(subList.length))
      (Row(sub, pred, obj, negSub, pred, obj))
    }else{
      val objList = entityList.diff(Array(obj)) //without the object
      //val objList = entityList //entire entityList
      val negObj =  objList(Random.nextInt(objList.length))
      (Row(sub, pred, obj, sub, pred, negObj))
    }
  }

  def ToSample(row: Row): (Sample[Float]) = {
    val label = 1f
    val feature = Tensor(T(row(0).asInstanceOf[String].toInt, row(1).asInstanceOf[String].toInt, row(2).asInstanceOf[String].toInt, 
                           row(3).asInstanceOf[String].toInt, row(4).asInstanceOf[String].toInt, row(5).asInstanceOf[String].toInt))
    (Sample(feature, label))    
  }
  
  def main (args: Array[String]){
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 
        
    val t1 = System.nanoTime
    val L = 1
    val k = 4
    val datasetName = "Tasneem" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val conf = Engine.createSparkConf().setAppName("Train").setMaster("local[1]")
    val sc = new SparkContext(conf) 
    val spark = SparkSession.builder.config(conf).getOrCreate()
    Engine.init    
    
    val dataset     = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd
    val entityList = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[Object]).collect
    val summary    = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)    
    //val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    val entityListLength = 3
    val relationListLength = 2
    val totalLength = entityListLength + relationListLength
    
    val samples = dataset.map(row => {NegativeSample(row, entityList, entityListLength)})    
    val trainData = samples.map(row => ToSample(row))
    
    val row = Tensor(Storage(Array(1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 3.0f)), 1, Array(6))
    //val row = trainData.take(1)
    println("Row:\n" + row)
    
    val reshape1 = Reshape(Array(6)).forward(row)
    println("Reshape[6]:\n" + reshape1)
    
    val embedding = LookupTable(totalLength, k).setInitMethod(RandomUniform(-6/Math.sqrt(k), 6/Math.sqrt(k)))
    
    println("\nEmbedding:\n" + embedding.weight)
    
    val rowToEmb = embedding.forward(reshape1)
    println("\nRow to embedding:\n" + rowToEmb)
    
    val reshape2 = Reshape(Array(2,3,1,k)).forward(rowToEmb)
    println("Reshape[2,3,1,emb_dim] :\n" + reshape2)
    
    val reshape22 = Reshape(Array(2,3,k)).forward(rowToEmb)
    println("Reshape[2,3,emb_dim] version2 :\n" + reshape22)
    
    val squeeze = Squeeze(1).forward(reshape2)
    println("Squeeze(1):\n" + reshape2)    

    val posSub  = Select(1, 1).forward(rowToEmb)
    val posPred = Select(1, 2).forward(rowToEmb)
    val posObj  = Select(1, 3).forward(rowToEmb)
    val negSub  = Select(1, 4).forward(rowToEmb)
    val negPred = Select(1, 5).forward(rowToEmb)
    val negObj  = Select(1, 6).forward(rowToEmb)
    println("Posive subject:\n"+ posSub +"\nPosive predicate:\n"+ posPred +"\nPosive object:\n"+ posObj +"\nNegative subject:\n"+ negSub +"\nNegative predicate:\n"+ negPred +"\nNegative object:\n" + negObj)

    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }  
}
