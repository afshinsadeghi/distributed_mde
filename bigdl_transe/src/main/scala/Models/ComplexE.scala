package Models

import org.apache.log4j._
import java.io._
import scala.util.Random
import org.apache.spark.sql._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

object ComplexE {
  
  //Creates the embedding
  def CreateEmbedding (entityListLength:Int, relationListLength:Int, k:Int) : (Tensor[Float], Tensor[Float]) = {  
    val entityEmbedding   = Tensor(entityListLength, k).rand(-6 / Math.sqrt(k), 6 / Math.sqrt(k))
    val relationEmbedding = Tensor(relationListLength, k).rand(-6 / Math.sqrt(k), 6 / Math.sqrt(k))
    
    (entityEmbedding, relationEmbedding) //create the total embedding that contains both entity and relation embedding
  }
   
  //Creates negative sample of the positive sample
  def NegativeSample(row:Row, entityList:Array[Object]): (Row) = {
    val sub  = row.get(0).asInstanceOf[Object]
    val pred = row.get(1).asInstanceOf[Object]
    val obj  = row.get(2).asInstanceOf[Object]    
    
    if(Random.nextInt(2) == 0){ //change head
      val subList = entityList.diff(Array(sub)) //without the subject
      val negSub =  subList(Random.nextInt(subList.length))
      (RowFactory.create(negSub, pred, obj))
    }else{
      val objList = entityList.diff(Array(obj)) //without the object
      val negObj =  objList(Random.nextInt(objList.length))
      (RowFactory.create(sub, pred, negObj))
    }
  }
  
  //Calculates the distance between positive and negative triple as Tensor of given dimension
  def Distance(sample:Array[Row], entityListLength:Integer, entityEmbedding:Tensor[Float], relationEmbedding:Tensor[Float], k:Int): (Tensor[Float]) = {      
    val sampleDistance = Tensor(sample.size, 1)
    var count = 1    
    sample.map(row => {
      
      val subTensor  = entityEmbedding.select  (1, row.get(0).toString.toInt)
      val predTensor = relationEmbedding.select(1, row.get(1).toString.toInt)
      val objTensor  = entityEmbedding.select  (1, row.get(2).toString.toInt)          
      println("Subject:\n" + subTensor + "\nPredicate:\n" + predTensor + "\nObject:\n" + objTensor)
      val splitValue = k/2  
    
      var splitTensor = subTensor.split(splitValue,1) 
      val (realSub, imgSub) = (splitTensor(0), splitTensor(1)) 
      
      splitTensor = predTensor.split(splitValue,1) 
      val (realPred, imgPred) = (splitTensor(0), splitTensor(1)) 
    
      splitTensor = objTensor.split(splitValue,1) 
      val (realObj, imgObj) = (splitTensor(0), splitTensor(1)) 
    
      val realScore = (realSub * realPred - imgSub * imgPred).sum
      val imgScore  = (realSub * imgPred  + imgSub * realPred).sum
      val dist      = ((realObj * realScore) + (imgObj * realScore)).sum
      
      sampleDistance.update(count,dist)
      count += 1
    })    
    (sampleDistance)
  }
  
 
  //Calculates the score of the positive and negative sample
  def LossFunction(posDist:Tensor[Float], negDist:Tensor[Float], gamma:Float, k:Int, trainSize: Long): (Float) = {
    val loss = (ReLU().forward(posDist  + gamma - negDist)).sum /trainSize //margin rank
    println(loss)
    (loss)
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

    val datasetName = "Turzo" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val k = 6
    val gamma:Float = 2f
    val learningRate:Float = 0.01f
    val L = 1
    val nEpoch = 1
    val optim = new SGD(learningRate = learningRate)

    val spark = SparkSession.builder.master("local").appName("TrainTransE").config("spark.some.config.option","some-value").getOrCreate()
    val trainSet   = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd
    val summary    = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    val entityList = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[Object]).collect
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    val trainSize = trainSet.count
   
    val (entityEmbedding, relationEmbedding) = CreateEmbedding(entityListLength, relationListLength, k)   
    

    for(epoch <- 1 to nEpoch){ //line 4 of the main algorithm
      
      val positiveSample = trainSet.collect
      val negativeSample = positiveSample.map(row => {NegativeSample(row,entityList)}) 
      
      val positiveDistance = Distance(positiveSample, entityListLength, entityEmbedding, relationEmbedding, k) 
      val negativeDistance = Distance(negativeSample, entityListLength, entityEmbedding, relationEmbedding, k)
     
      println("Epoch: " + epoch)
      //Function for updating the embedding
      def Update(x: Tensor[Float]) = {        
        (LossFunction(positiveDistance, negativeDistance, gamma, k, trainSize), x) 
      }

      optim.optimize(Update, entityEmbedding) //line 12  of the main algorithm
           
    }    
    
    SaveTrainedEmbedding(entityEmbedding, resoiurceDir + "entityEmbedding.tsv" )
    SaveTrainedEmbedding(relationEmbedding, resoiurceDir + "relationEmbedding.tsv" )
    
    
    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}