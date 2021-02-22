package Tryouts

import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import scala.util._
import org.apache.spark.sql._ //for Row,SparkSession
import org.apache.spark.rdd.RDD
import com.intel.analytics.bigdl.nn._ //for ReLU
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

object EvaluateTransEVersion1 {
  def Distance(row:Row, entityListLength: Int, entityEmbedding:Array[Array[BigDecimal]], relationEmbedding:Array[Array[BigDecimal]], L:Int): (BigDecimal) = {
   
    val subEmbedding  = entityEmbedding(row(0).toString.toInt-1)
    val predEmbedding = relationEmbedding(row(1).toString.toInt-1)
    val objEmbedding  = entityEmbedding(row(2).toString.toInt-1)
    
    L match {
      case 1 => { ((subEmbedding, predEmbedding, objEmbedding).zipped.map(_+_-_).map(r => r.abs).sum) }
      case _ => { ((subEmbedding, predEmbedding, objEmbedding).zipped.map(_.pow(2) + _.pow(2) - _.pow(2)).map(r => BigDecimal(Math.sqrt(r.toDouble).abs)).sum) }
    }
  }
  
  def Rank(row: Row, score:BigDecimal, changedSample:Array[Row], trainSet: Array[Row], entityListLength: Int, entityEmbedding:Array[Array[BigDecimal]], relationEmbedding:Array[Array[BigDecimal]], L:Int)
         : (Long, Long) = {
    
    //val filterNegativeSample = (changedSample).diff(trainSet) //:+ row 
    
    val rawScore    = changedSample.map(row => {Distance(row, entityListLength, entityEmbedding, relationEmbedding, L)}) //for raw rank
    //val filterScore = filterNegativeSample.map(row => {Distance(row, entityListLength, entityEmbedding, relationEmbedding, L)}) //for filter rank
    
    Sorting.quickSort[BigDecimal](rawScore)(Ordering[BigDecimal].reverse)
    //Sorting.quickSort(rawScore) 
    
    //Sorting.quickSort(filterScore)
    
    val rawRank    = rawScore.indexOf(score)
    //val filterRank = filterScore.indexOf(score)
    val filterRank = rawRank
    (rawRank, filterRank)
  }
  
  def main (args: Array[String]){
    val t1 = System.nanoTime
    val L  = 1
    
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 
    
    val datasetName  = "Nations" //Folder name of the dataset
    val resoiurceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    
    val spark = SparkSession.builder.master("local").appName("Evaluation").config("spark.some.config.option","some-value").getOrCreate()
    val sc    = spark.sparkContext  
    
    val trainSet          = spark.read.option("sep", "\t").csv(resoiurceDir + "train.tsv").rdd.collect
    val testSet           = spark.read.option("sep", "\t").csv(resoiurceDir + "test.tsv").rdd.collect
    val entityEmbedding   = spark.read.option("sep", "\t").csv(resoiurceDir + "entityEmbedding.tsv").rdd.map(r1 => {r1.toSeq.toArray.map(_.asInstanceOf[String]).map(r2 => {BigDecimal(r2.toDouble)})}).collect
    val relationEmbedding = spark.read.option("sep", "\t").csv(resoiurceDir + "relationEmbedding.tsv").rdd.map(r1 => {r1.toSeq.toArray.map(_.asInstanceOf[String]).map(r2 => {BigDecimal(r2.toDouble)})}).collect
    val entityRDD         = spark.read.option("sep", "\t").csv(resoiurceDir + "entityToID.tsv").select("_c1").rdd.map(r => r(0).asInstanceOf[String].toInt)    
    val summary           = spark.read.option("sep", "\t").option("header", "true").csv(resoiurceDir + "summary.tsv").take(1)
    
    val (entityListLength, relationListLength) = (summary(0).get(0).toString.toInt, summary(0).get(1).toString.toInt)
     
    var hit1R    = 0
    var hit5R    = 0
    var hit10R   = 0
    var hit50R   = 0
    var hit100R  = 0
    var hit1F    = 0
    var hit5F    = 0
    var hit10F   = 0
    var hit50F   = 0
    var hit100F  = 0
    var rawMeanRank    = 0.0
    var filterMeanRank = 0.0

    testSet.foreach(row => {
      
      val score = Distance(row, entityListLength, entityEmbedding, relationEmbedding, L)
      
      val changeHead = entityRDD.map(r => Row(r, row(1), row(2))).collect //for raw hit@10 with head corruption
      val changeTail = entityRDD.map(r => Row(row(0), row(1), r)).collect //for raw hit@10 with tail corruption
      
      val (rawRankHead, filterRankHead) = Rank(row, score, changeHead, trainSet, entityListLength, entityEmbedding, relationEmbedding, L)
      val (rawRankTail, filterRankTail) = Rank(row, score, changeTail, trainSet, entityListLength, entityEmbedding, relationEmbedding, L)
      
      val rawRank    = (rawRankHead + rawRankTail)/2
      val filterRank = (filterRankHead + filterRankTail)/2
      
      //println("Raw rank for row    : " + row + "  in Test dataset:" + rawRank)
      //println("Filter rank for row : " + row + "  in Test dataset:" + filterRank)
      
      rawMeanRank += rawRank      
      if(rawRank <= 1)  {hit1R   += 1}      
      if(rawRank <= 5)  {hit5R   += 1}      
      if(rawRank <= 10) {hit10R  += 1}
      if(rawRank <= 50) {hit50R  += 1}
      if(rawRank <= 100){hit100R += 1}   
      
      filterMeanRank += filterRank      
      if(filterRank <= 1)  {hit1F   += 1}      
      if(filterRank <= 5)  {hit5F   += 1}      
      if(filterRank <= 10) {hit10F  += 1}
      if(filterRank <= 50) {hit50F  += 1}
      if(filterRank <= 100){hit100F += 1}  
      
    })
    
    println("Raw mean Rank   : " + (rawMeanRank/testSet.length))
    println("hit@1   : " + (hit1R   / testSet.length.toFloat * 100.0) + " %")
    println("hit@5   : " + (hit5R   / testSet.length.toFloat * 100.0) + " %")
    println("hit@10  : " + (hit10R  / testSet.length.toFloat * 100.0) + " %")
    println("hit@50  : " + (hit50R  / testSet.length.toFloat * 100.0) + " %")
    println("hit@100 : " + (hit100R / testSet.length.toFloat * 100.0) + " %")
    /*
    println("Filter mean Rank : " + (filterMeanRank/testSet.length))
    println("hit@1   : " + (hit1F   / testSet.length.toFloat * 100.0) + " %")
    println("hit@5   : " + (hit5F   / testSet.length.toFloat * 100.0) + " %")
    println("hit@10  : " + (hit10F  / testSet.length.toFloat * 100.0) + " %")
    println("hit@50  : " + (hit50F  / testSet.length.toFloat * 100.0) + " %")
    println("hit@100 : " + (hit100F / testSet.length.toFloat * 100.0) + " %")
    */
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }  
}