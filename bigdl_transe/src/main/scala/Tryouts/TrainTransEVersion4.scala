package Tryouts

import org.apache.log4j._ //for log messages
import java.io._ //for Printwriter
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

object TrainTransEVersion4 {
  
  //Creates the embedding
  def CreateEmbedding (entityListLength:Int, relationListLength:Int, k:Int) : (Tensor[Float], Tensor[Float]) = {    
    val entityEmbedding   = Tensor(entityListLength, k).rand(-6 / Math.sqrt(k), 6 / Math.sqrt(k))
    val relationEmbedding = Tensor(relationListLength, k).rand(-6 / Math.sqrt(k), 6 / Math.sqrt(k))
    (entityEmbedding, relationEmbedding) //create the total embedding that contains both entity and relation embedding
  }
  
    
  //Creates negative sample of the positive sample
  def NegativeSample(row:Row, entityList:Array[Object]): (Row) = {
    //println(row)
    val sub  = row.get(0).asInstanceOf[Object]
    val pred = row.get(1).asInstanceOf[Object]
    val obj  = row.get(2).asInstanceOf[Object]    
    
    if(Random.nextInt(2) == 0){ //change head
      val subList = entityList.diff(Array(sub)) //without the subject
      //val subList = entityList //entire entityList
      val negSub =  subList(Random.nextInt(subList.length))
      (RowFactory.create(negSub, pred, obj))
    }else{
      val objList = entityList.diff(Array(obj)) //without the object
      //val objList = entityList //entire entityList
      val negObj =  objList(Random.nextInt(objList.length))
      (RowFactory.create(sub, pred, negObj))
    }
  }
  
  //Calculates the distance between positive and negative triple as Tensor of given dimension
  def Distance(sample:Array[Row], entityEmbedding:Tensor[Float], relationEmbedding:Tensor[Float], L:Int): (Tensor[Float]) = {      
    val sampleDistance = Tensor(sample.size, entityEmbedding.size(2))
    var count = 1    
    sample.map(row => { 
      val subTensor  = entityEmbedding.select(1, row.get(0).toString.toInt)
      val predTensor = relationEmbedding.select(1, row.get(1).toString.toInt)
      val objTensor  = entityEmbedding.select(1, row.get(2).toString.toInt)          
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
  def ScoreFunction(posDist:Tensor[Float], negDist:Tensor[Float], gamma:Float, k:Int, trainSize: Long): (Float) = {
    val loss = (ReLU().forward(posDist - negDist + gamma)).sum //margin rank 
    //println((loss/k)/trainSize)
    (loss)
  }
  
  
  //Trains the TransE
  def TrainTransE(trainSet: RDD[Row], entityList:Array[Object], entityListLength:Int, entityEmbedding: Tensor[Float], relationEmbedding: Tensor[Float], L:Int, gamma:Float, k: Int, trainSize: Long): (Float) = {
    
    //Normalize(1).forward(embedding) // Normalization of embedding
    
    val positiveSample = trainSet.collect //.takeSample(false, batchSize) //line 6 of the main algorithm
    val negativeSample = positiveSample.map(row => {NegativeSample(row,entityList)}) //line 8-10 of the main algorithm
    
    val positiveDistance = Distance(positiveSample, entityEmbedding, relationEmbedding, L) //line 10 of the main algorithm
    val negativeDistance = Distance(negativeSample, entityEmbedding, relationEmbedding, L) //line 10 of the main algorithm    
    
    (ScoreFunction(positiveDistance, negativeDistance, gamma, k, trainSize))
  }
  
  //Saves the embedding to a tsv file
  def SaveTrainedEmbedding(embedding:Tensor[Float], fileDir:String){
    val printWriter = new PrintWriter(new File(fileDir))        
    for(index <- 1 to embedding.size(1)){
      printWriter.write(embedding.apply(index).toArray.mkString("\t")+"\n")
    }    
    printWriter.close
  }
  
  
  def main (args: Array[String]){
    val t1 = System.nanoTime
   
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 

    val datasetName = "Kingship" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val k = 20
    val gamma:Float = 1f
    val learningRate:Float = 0.001f
    val L = 1
    val nEpoch = 1000
    val optim = new Adagrad(learningRate = learningRate)
    /*
    val conf = Engine.createSparkConf().setAppName("Train").setMaster("local[1]")
    val sc = new SparkContext(conf) 
    val spark = SparkSession.builder.config(conf).getOrCreate()
    Engine.init
    */
    val spark = SparkSession.builder.master("local").appName("TrainTransE").config("spark.some.config.option","some-value").getOrCreate()
    val trainSet   = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd
    val summary    = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    val entityList = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[Object]).collect
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    val trainSize = trainSet.count
    
    val (entityEmbedding, relationEmbedding) = CreateEmbedding(entityListLength, relationListLength, k)
   
    //SaveTrainedEmbedding(embedding, resoiurceDir + "embeddingBeforeTraining.tsv" )
    
    for(epoch <- 1 to nEpoch){ //line 4 of the main algorithm
      println("Epoch: " + epoch)
      val positiveSample = trainSet.collect //.takeSample(false, batchSize) //line 6 of the main algorithm
      val negativeSample = positiveSample.map(row => {NegativeSample(row,entityList)}) //line 8-10 of the main algorithm
    
      val positiveDistance = Distance(positiveSample, entityEmbedding, relationEmbedding, L) //line 10 of the main algorithm
      val negativeDistance = Distance(negativeSample, entityEmbedding, relationEmbedding, L) //line 10 of the main algorithm             
      //Function for updating the embedding
      def Update(x: Tensor[Float]) = {
        
        (ScoreFunction(positiveDistance, negativeDistance, gamma, k, trainSize), x) 
      }

      
      optim.optimize(Update, entityEmbedding) //line 12  of the main algorithm
      optim.optimize(Update, relationEmbedding)
    }    
    
    SaveTrainedEmbedding(entityEmbedding, resoiurceDir + "entityEmbedding.tsv" )
    SaveTrainedEmbedding(relationEmbedding, resoiurceDir + "relationEmbedding.tsv" )
    
    println((System.nanoTime - t1) / 1e9d  + "s")
    //println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}