package Preprocessing
/*
 * This code processes the original dataset and creates the following files:
 * 	1. Entity   to ID file
 * 	2. Relation to ID file
 * 	3. Original dataset to ID dataset
 * 	4. Summary of the dataset
 */

import org.apache.log4j._ //for log messages
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.sql.types._ //for variable type
import java.nio.file.{Files, Paths, StandardCopyOption}
import java.io.File //for File
import com.intel.analytics.bigdl.tensor.Tensor // for Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat //for Optimizer(learning rate), Tensor[Float]

object PreprocessingOriginalDataset {
  
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
  
  
  
  /*
   * Renames the file while created from the writing the dataframe to tsv file
   * Moves the renamed file 
   * Deletes the folder 
   */
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
    fileName match {
      case fileName if fileName.equals("summary.tsv") => df.repartition(1).write.option("delimiter","\t").option("header", "true").format("com.databricks.spark.csv").save((dirForTempFOlder))
      case _ => df.repartition(1).write.option("delimiter","\t").format("com.databricks.spark.csv").save((dirForTempFOlder))
    }    
    renameFile(dirForTempFOlder, (dir+fileName))    
  }

  
  def main (args: Array[String]){
    val t1 = System.nanoTime //For calculating run time
   
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages
    
    
    /* "train.tsv" for dataset "Tasneem" is kept as: (resourceDir + datasetName + "/originalTrain.tsv") [so is valid.tsv, test.tsv]
     * "entityToID.tsv" will be kept as: (resourceDir + datasetName + "/entityToTD.tsv") [so will relationToID.tsv]
     * converted "train.tsv" wrt to the dictionary of entity and relation will be kept as:(resourceDir + datasetName + "/train.tsv")
     */ 
    val datasetName = "Lab" //Folder name of the dataset
    val resourceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"//Parent directory of the resources 
           
    val spark = SparkSession.builder.master("local").appName("Preprocessing of original dataset")
                                    .config("spark.some.config.option","some-value").getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._ //for creating dataframe
        
    val schema1 = new StructType().add("_c0",IntegerType,true).add("_c1",IntegerType,true).add("_c2",IntegerType,true)
    val schema2 = new StructType().add("_c0",StringType,true).add("_c1",LongType,true)
    val schema3 = new StructType().add("Total entity", IntegerType, true).add("Total relation", IntegerType, true)
    
    val trainDF = spark.read.option("sep", "\t").csv(resourceDir + "originalTrain.tsv")
    val validDF = spark.read.option("sep", "\t").csv(resourceDir + "originalValid.tsv")
    val testDF  = spark.read.option("sep", "\t").csv(resourceDir + "originalTest.tsv") 
    val totalDF = trainDF.union(validDF.union(testDF))    
    
    val entityArray   = totalDF.select("_c0").withColumnRenamed("_c0", "_c2").union(totalDF.select("_c2"))
                               .distinct.rdd.map(r => r(0).asInstanceOf[String]).collect
    val relationArray = totalDF.select("_c1").distinct.rdd.map(r => r(0).asInstanceOf[String]).collect    
            
    val trainRDD    = trainDF.rdd.map(r => {Row(entityArray.indexOf(r(0))+1, relationArray.indexOf(r(1))+1, entityArray.indexOf(r(2))+1)})
    val validRDD    = validDF.rdd.map(r => {Row(entityArray.indexOf(r(0))+1, relationArray.indexOf(r(1))+1, entityArray.indexOf(r(2))+1)})
    val testRDD     =  testDF.rdd.map(r => {Row(entityArray.indexOf(r(0))+1, relationArray.indexOf(r(1))+1, entityArray.indexOf(r(2))+1)})
    val entityRDD   = sc.parallelize(entityArray).zipWithIndex.map(r => {Row(r._1, r._2+1)})
    val relationRDD = sc.parallelize(relationArray).zipWithIndex.map(r => {Row(r._1, r._2+1)})
    val summaryRDD  = sc.parallelize(Seq(Array(entityArray.length, relationArray.length))).map(r => {Row(r(0), r(1))})
        
    val newTrainDF = spark.createDataFrame(trainRDD,   schema1)
    val newValidDF = spark.createDataFrame(validRDD,   schema1)
    val newTestDF  = spark.createDataFrame(testRDD,    schema1)
    val entityDF   = spark.createDataFrame(entityRDD,  schema2)
    val relationDF = spark.createDataFrame(relationRDD,schema2)
    val summaryDF  = spark.createDataFrame(summaryRDD, schema3)
        
    writeDataframeToTSVFile(newTrainDF, resourceDir, "train.tsv")
    writeDataframeToTSVFile(newValidDF, resourceDir, "valid.tsv")
    writeDataframeToTSVFile(newTestDF,  resourceDir, "test.tsv")
    writeDataframeToTSVFile(entityDF,   resourceDir, "entityToID.tsv")
    writeDataframeToTSVFile(relationDF, resourceDir, "relationToID.tsv") 
    writeDataframeToTSVFile(summaryDF, resourceDir,  "summary.tsv") 
    val time = (System.nanoTime - t1) / 1e9d
    println(trainDF.count)
    println(validDF.count)
    println(testDF.count)
    println(totalDF.count)
    println(time)  
    println(entityArray.length)
    println(relationArray.length)
  }

}