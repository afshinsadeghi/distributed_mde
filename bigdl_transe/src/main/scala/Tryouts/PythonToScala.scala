package Tryouts

import org.apache.log4j._ // for log messages
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.rdd.RDD //for RDD
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T //for T
import com.intel.analytics.bigdl.numeric.NumericFloat
import scala.util.Random
import com.intel.analytics.bigdl.nn._ //for layers
import com.intel.analytics.bigdl.optim._ //for optimizer
import com.intel.analytics.bigdl.visualization.TrainSummary
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

object PythonToScala {
  
  def NegativeSample(row:Row, entityList:Array[Object]): (Row) = {
    println(row)
    val sub  = row.get(0).asInstanceOf[Object]
    val pred = row.get(1).asInstanceOf[Object]
    val obj  = row.get(2).asInstanceOf[Object]    
    
    if(Random.nextInt(2) == 0){ //change head
      val subList = entityList.diff(Array(sub)) //without the subject
      //val subList = entityList //entire entityList
      val negSub =  subList(Random.nextInt(subList.length))
      (RowFactory.create(sub, pred, obj, negSub, pred, obj))
    }else{
      val objList = entityList.diff(Array(obj)) //without the object
      //val objList = entityList //entire entityList
      val negObj =  objList(Random.nextInt(objList.length))
      (RowFactory.create(sub, pred, obj, sub, pred, negObj))
    }
  }
  
  def ToSample(row: Row): (Sample[Float]) = {
    val label = 1f
    val feature = Tensor(T(row(0).asInstanceOf[String].toInt, row(1).asInstanceOf[String].toInt, row(2).asInstanceOf[String].toInt, row(3).asInstanceOf[String].toInt, row(4).asInstanceOf[String].toInt, row(5).asInstanceOf[String].toInt))
//    val labels = Array(Tensor(1).fill(1), Tensor(1).fill(-1))
    (Sample(feature, label))    
  }
  
    def CreateModel(totalLength:Int, k:Int): (Sequential[Float]) = {
    
    val model = Sequential()
    model.add(Reshape(Array(6)))
    val embedding = LookupTable(totalLength, k).setInitMethod(RandomUniform(-6 / Math.sqrt(k), 6 / Math.sqrt(k)) )
    model.add(embedding)
    
    model.add(Reshape(Array(2, 3, 1, k))).add(Squeeze(1)) //Error: dimension out of range
    model.add(SplitTable(2))
    
    val branches = ParallelTable()
    val branch1 = Sequential()
    val pos_h_l = Sequential().add(ConcatTable().add(Select(2, 1)).add(Select(2, 2)))
    val pos_add = pos_h_l.add(CAddTable())
    val pos_t = Sequential().add(Select(2, 3)).add(MulConstant(-1.0))
    val triple_pos_score = Sequential().add(ConcatTable().add(pos_add).add(pos_t)).add(CAddTable()).add(Abs())
    branch1.add(triple_pos_score).add(Squeeze(3)).add(Squeeze(1)).add(Unsqueeze(2))

    val branch2 = Sequential()
    val neg_h_l = Sequential().add(ConcatTable().add(Select(2, 1)).add(Select(2, 2)))
    val neg_add = neg_h_l.add(CAddTable())
    val neg_t = Sequential().add(Select(2, 3)).add(MulConstant(-1.0))
    val triple_neg_score = Sequential().add(ConcatTable().add(neg_add).add(neg_t)).add(CAddTable()).add(Abs())
    println(triple_neg_score)
    branch2.add(triple_neg_score).add(Squeeze(3)).add(Squeeze(1)).add(Unsqueeze(2))

    branches.add(branch1).add(branch2)
    model.add(branches)
    
    (model) 
  }
    
   def Training(trainModel:Sequential[Float], trainData:RDD[Sample[Float]], nEpochs:Int){
     println(2)
    val optimizer = Optimizer(
            model = trainModel,            
            sampleRDD = trainData,
            criterion = MarginRankingCriterion[Float](),
            batchSize=2
            ).setEndWhen(Trigger.maxEpoch(nEpochs))
            .setOptimMethod(new SGD(learningRate = 0.05))
            println(3)
     val train_summary = TrainSummary("/home/tasneem/Desktop/Train", "TrainSummary")       
     optimizer.setTrainSummary(train_summary)
     val trained_model = optimizer.optimize()
     val loss = train_summary.readScalar("Loss")
     println(loss)
  }
  
  def main (args: Array[String]){
    val t1 = System.nanoTime
    
    val L = 1
    val k = 5
    
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 
    
    val datasetName = "Tasneem" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val conf = Engine.createSparkConf().setAppName("Train").setMaster("local[1]")
    val sc = new SparkContext(conf) 
    val spark = SparkSession.builder.config(conf).getOrCreate()
    Engine.init
    
    val dataset     = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd
    val entityList = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[Object]).collect
    val summary    = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
    
    val samples = dataset.map(row => {NegativeSample(row, entityList)})
    
    val trainData = samples.map(row => ToSample(row))
    //trainData.foreach(r => println(r))
    
    val model = CreateModel(entityListLength+relationListLength, k)
    println(model)
    //Training(model, trainData, 1)

    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }  
}
