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
import java.io._
import scala.math.BigDecimal
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable._
import org.apache.spark.sql.expressions.Window

object PreProcessingForNegativeSample {
  
  
  def SubJoinedObj(relation:Integer, sample:DataFrame, finalDF:DataFrame): (DataFrame) = {
    
   val localDF = sample.select("_c0", "_c2").where(sample("_c1") === relation).withColumn("ID",row_number.over(Window.orderBy("_c0")))
                                            .withColumnRenamed("_c0", "s:"+relation).withColumnRenamed("_c2", "o:"+relation)
    if(finalDF.isEmpty == true){
      (localDF)
    }else{
      (finalDF.join(localDF, localDF("ID") === finalDF("ID"), "full").drop("ID").withColumn("ID",row_number.over(Window.orderBy("s:"+relation))))      
    }   
  }
  
  //Gets the list of files from the directory of the folder
  def getListOfFiles(dir: File, extensions: List[String]): (List[File]) = {
      dir.listFiles.filter(_.isFile).toList.filter { file => extensions.exists(file.getName.endsWith(_))}
  }
  
    
  //Empties the given folder and deletes it in recursion
  def deleteFolder(file:File){
    //to end the recursive loop
    if (!file.exists()){
      return
    }
    // if directory, go inside and call recursively
    if (file.isDirectory()) {
      val fileList = file.listFiles
      fileList.foreach(f => deleteFolder(f))
    }
    //call delete to delete files and empty directory
    file.delete()
    //System.out.println("Deleted file/folder: "+file.getAbsolutePath())       
  }
  
  
  //Renames the file while created from the writing the dataframe to tsv file, moves it and delete the folder
  def renameFile(oldFolderName:String, newFileName:String){
    try{
    //Rename the parquet file      
      val oldFileFolder = new File(oldFolderName)    
      val newFile = new File(newFileName)    
      val file = getListOfFiles(oldFileFolder, List("csv"))        
      var flag = file.isEmpty        
    
      if(!flag){            
        val oldFile = new File (oldFileFolder + "/" + file(0))      
        flag = file(0).renameTo(newFile) //rename and move      
        if(flag){        
          deleteFolder(oldFileFolder)      
        }else{        
          println("Error: Renaming file was unsuccessful for " + oldFolderName + " !")     
        }    
      }else{      
        println("Error: File doen't exist for " + oldFolderName + " !")    
      }       
    }catch{
      case _: Throwable => println("Error: Folder doesn't exist for " + oldFolderName + " !")    
    }
  }
  
  
  //Writes the dataframe to the corresponding file
  def writeDataframeToTSVFile(df:DataFrame, dir:String, fileName:String){
    val dirForTempFOlder = dir + "/temp" //Directory of the temp folder which will be changed
    df.repartition(1).write.option("delimiter","\t").format("com.databricks.spark.csv").option("header","true").save((dirForTempFOlder))
    renameFile(dirForTempFOlder, (dir+fileName))    
  }
  
  
  def main (args: Array[String]){
    
    val t1 = System.nanoTime
    
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages 
    
    val datasetName = "Countries/Countries_S1" //Folder name of the dataset
    val resourceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"
    
    val schema = new StructType().add("_c0",StringType,true).add("_c0",StringType,true).add("_c1",IntegerType,true)
    
    val spark = SparkSession.builder.master("local").appName("SamplePreProcessing").config("spark.some.config.option","some-value").getOrCreate()    
    val sc = spark.sparkContext
    
    import spark.implicits._
    
    val sample = spark.read.option("sep", "\t").csv(resourceDir + "train.tsv")
    
    val sort = sample.groupBy("_c1").count().sort($"count".desc)
    
    val relation = sort.select("_c1").rdd.map(r => r(0).toString).collect
    val maxLength = sort.select("count").take(1).map(r => r(0).asInstanceOf[Long]).apply(0).toInt
    println("NUMBER OF PREDICATES: " + relation.length)
    println("EXPECTED DF SIZE: " + maxLength)
    
    var finalDF = spark.emptyDataFrame    
    var flag = true
    
    val array = Array.ofDim[String](maxLength)
    
    def returnRow(r1:String, r2:Int, sub:Array[String], subLength:Int, obj:Array[String], objLength:Int): (Row) ={
      r2 match {
        case r2 if (r2.<(subLength) && r2.<(objLength)) => (Row(sub(r2), obj(r2), r2))        
        case r2 if (r2.<(subLength) && r2.>=(objLength))=> (Row(sub(r2), r1, r2))
        case r2 if (r2.>=(subLength)&& r2.<(objLength)) => (Row(r1, obj(r2), r2))
        case _ => (Row(r1, r1, r2))
      }     
    }
    
    var step = 1
    val t2 = System.nanoTime
    relation.foreach(r => {      
      println("START OF STEP :" + step)
      val sub = sample.select("_c0").where( $"_c1" === r).distinct.rdd.map(r => r(0).toString).collect    
      val obj = sample.select("_c2").where( $"_c1" === r).distinct.rdd.map(r => r(0).toString).collect
      val subLength = sub.length
      val objLength = obj.length
      val RDD = sc.parallelize(array.zipWithIndex.map(r => returnRow(r._1, r._2, sub, subLength, obj, objLength)))
      val localDF = spark.createDataFrame(RDD, schema).toDF("sub:"+ r, "obj:"+ r, "ID")
      flag match {
        case true => { finalDF = localDF
                       flag = false }
        case _ => finalDF = finalDF.join(localDF, "ID")
      }     
      step += 1
    })
    
    finalDF = finalDF.drop("ID").na.drop(how = "all") //removes rows where all the columns are null
    println("TOTAL RUNTIME AFTER ALL THE STEPS: " + (System.nanoTime - t2) / 1e9d  + "s")
    println("FINAL DF SIZE: " + finalDF.count)
    
    writeDataframeToTSVFile(finalDF, resourceDir, "table.tsv")  
    
    /* With ".withColumn("ID", monotonically_increasing_id)"
    //-----------------------------------------------------------------------------------------------------------------------------------------------
    relation.take(2).foreach(r =>{
      val sub = sample.select("_c0").where( $"_c1" === r).distinct.withColumnRenamed("_c0", "sub:"+r).withColumn("ID", monotonically_increasing_id)    
      val obj = sample.select("_c2").where( $"_c1" === r).distinct.withColumnRenamed("_c2", "obj:"+r).withColumn("ID", monotonically_increasing_id)       
      val localDF = sub.join(obj, sub("ID") === obj("ID"), "full").drop("ID").withColumn("ID", monotonically_increasing_id)
      finalDF = finalDF.withColumn("ID", monotonically_increasing_id)
      if(flag){
        finalDF = localDF.drop("ID")
        flag = false
      }else{
        finalDF = finalDF.join(localDF, localDF("ID") === finalDF("ID"), "full").drop("ID")
      }
      
      //finalDF = finalDF.na.drop(how = "all")
    })     
    println(finalDF.count)
    */
    
    /* With "row_number.over(Window.orderBy())"
    //-----------------------------------------------------------------------------------------------------------------------------------------------
    def SubJoinedObj(relation:Integer, sample:DataFrame, finalDF:DataFrame): (DataFrame) = {
       val localDF = sample.select("_c0", "_c2").where(sample("_c1") === relation).withColumn("ID",row_number.over(Window.orderBy("_c0"))).withColumnRenamed("_c0", "s:"+relation).withColumnRenamed("_c2", "o:"+relation)
    	 if(finalDF.isEmpty == true){
      		(localDF)
    	 }else{
      	  (finalDF.join(localDF, localDF("ID") === finalDF("ID"), "full").drop("ID").withColumn("ID",row_number.over(Window.orderBy("s:"+relation))))      
       }   
    }   
    var finalDF = spark.emptyDataFrame    
    relation.foreach(r => {finalDF = SubJoinedObj(r, sample, finalDF)})
    finalDF.show
    //-----------------------------------------------------------------------------------------------------------------------------------------------
    */
     
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  } 
}