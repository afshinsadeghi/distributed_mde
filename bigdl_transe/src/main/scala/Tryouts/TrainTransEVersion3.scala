package Tryouts
/*
 * This version of Training TransE runs the epochs with fixed negative sample over the entire training dataset.
 */

import org.apache.log4j._ // for log messages
import java.io._
import scala.util.Random
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.rdd.RDD //for RDD
import com.intel.analytics.bigdl.nn._ //for ReLU
import com.intel.analytics.bigdl.optim._ //for Optimizer
import com.intel.analytics.bigdl.tensor.Tensor // for Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat //for Optimizer(learning rate), Tensor[Float]
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.intel.analytics.bigdl.utils.Engine

object TrainTransEVersion3 {
  
  //Creates the embedding
  def CreateEmbedding (entityListLength:Int, relationListLength:Int, k:Int) : (Tensor[Float]) = {    
    (Tensor(entityListLength + relationListLength, k).rand(6 / Math.sqrt(k), -6 / Math.sqrt(k))) //create the total embedding that contains both entity and relation embedding
  }
  
    
  //Creates negative sample of the positive sample
  def NegativeSample(row:Row, entityList:Array[Object]): (Row) = {
    val sub  = row.get(0).asInstanceOf[Object]
    val pred = row.get(1).asInstanceOf[Object]
    val obj  = row.get(2).asInstanceOf[Object]    
    
    if(Random.nextInt(2) == 0){ //change head
      //val subList = entityList.diff(Array(sub)) //without the subject
      val subList = entityList //entire entityList
      val negSub =  subList(Random.nextInt(subList.length))
      (RowFactory.create(negSub, pred, obj))
    }else{
      //val objList = entityList.diff(Array(obj)) //without the object
      val objList = entityList //entire entityList
      val negObj =  objList(Random.nextInt(objList.length))
      (RowFactory.create(sub, pred, negObj))
    }
  }
  
  //Calculates the distance between positive and negative triple as Tensor of given dimension
  def Distance(sample:Array[Row], entityListLength:Integer, embedding:Tensor[Float], L:Int): (Tensor[Float]) = {      
    val sampleDistance = Tensor(sample.size, embedding.size(2))
    var count = 1    
    sample.map(row => { 
      val subTensor  = embedding.select(1, row.get(0).toString.toInt)
      val predTensor = embedding.select(1, row.get(1).toString.toInt + entityListLength)
      val objTensor  = embedding.select(1, row.get(2).toString.toInt)          
      val dist = L_p_norm(subTensor, predTensor, objTensor, L)
      sampleDistance.update(count,dist)
      count += 1
    })    
    (sampleDistance)
  }
  
  //L1 and L2 norm. 
  def L_p_norm(SubTensor:Tensor[Float], predTensor:Tensor[Float], objTensor:Tensor[Float], L:Int): (Tensor[Float]) = {
    L match {
      case 1 => ((SubTensor + predTensor - objTensor).abs) //L1
      case _ => ((SubTensor.pow(2) + predTensor.pow(2) - objTensor.pow(2)).sqrt.abs) //L2    
    }           
  }
  
  //Calculates the score of the positive and negative sample
  def ScoreFunction(posDist:Tensor[Float], negDist:Tensor[Float], gamma:Float, k:Int): (Float) = {
    val loss = (ReLU().forward(posDist - negDist + gamma)).sum //margin rank 
    println(loss/k)
    (loss)
  }
  
  
  //Trains the TransE
  def TrainTransE(positiveSample: Array[Row], negativeSample: Array[Row], entityList:Array[Object], entityListLength:Int, relationListLength:Int, embedding: Tensor[Float], L:Int, gamma:Float, k: Int): (Float) = {
    
    Normalize(1).forward(embedding) // Normalization of embedding
    
    val positiveDistance = Distance(positiveSample, entityListLength, embedding, L) //line 10 of the main algorithm
    val negativeDistance = Distance(negativeSample, entityListLength, embedding, L) //line 10 of the main algorithm    
    
    (ScoreFunction(positiveDistance, negativeDistance, gamma, k))
  }
  
    
  def main (args: Array[String]){
    val t1 = System.nanoTime
   
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 

    val datasetName = "UML" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val k = 200
    val gamma:Float = 1f
    val learningRate:Float = 0.001f
    val L = 1
    val nEpoch = 1000
    val optim = new Adam(learningRate = learningRate)
    
    val spark = SparkSession.builder.master("local").appName("TrainTransE").config("spark.some.config.option","some-value").getOrCreate()
    
    val trainSet   = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd
    val summary    = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    val entityList = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[Object]).collect
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    
    val embedding = CreateEmbedding(entityListLength, relationListLength, k)
    
    val positiveSample = trainSet.collect //.takeSample(false, batchSize) //line 6 of the main algorithm
    val negativeSample = positiveSample.map(row => {NegativeSample(row,entityList)}) //line 8-10 of the main algorithm
   
    for(epoch <- 1 to nEpoch){ //line 4 of the main algorithm
      
      //Function for updating the embedding
      def Update(x: Tensor[Float]) = {
        
        (TrainTransE(positiveSample, negativeSample , entityList, entityListLength, relationListLength, embedding, L, gamma, k), x) 
      }

      optim.optimize(Update, embedding) //line 12  of the main algorithm
    }    
    println((System.nanoTime - t1) / 1e9d  + "s")
    //println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}