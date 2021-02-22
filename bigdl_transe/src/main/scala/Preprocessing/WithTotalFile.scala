package Preprocessing

import org.apache.log4j._ //for log messages
import org.apache.spark.sql._ //for Row, SparkSession
import org.apache.spark.sql.types._ //for variable type
import java.io.File //for File


object WithTotalFile {
  
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
    val t1 = System.nanoTime
   
    Logger.getLogger("org").setLevel(Level.OFF)  //to get rid of logger messages 
    Logger.getLogger("akka").setLevel(Level.OFF) //to get rid of logger messages
    
    
    val datasetName = "Sample/Sample1v2" //Folder name of the dataset
    val resourceDir = "/home/tasneem/Desktop/eclipse/workspace/Thesis/src/main/resources/" + datasetName + "/"//Parent directory of the resources 
           
    val spark = SparkSession.builder.master("local").appName("Preprocessing of original dataset")
                                    .config("spark.some.config.option","some-value").getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._ //for creating dataframe
        
    val schema1 = new StructType().add("_c0",IntegerType,true).add("_c1",IntegerType,true).add("_c2",IntegerType,true)
    val schema2 = new StructType().add("_c0",StringType,true).add("_c1",LongType,true)
    val schema3 = new StructType().add("Total entity",   IntegerType, true)
                                  .add("Total relation", IntegerType, true)
                                  .add("Total data",     IntegerType, true)
                                  .add("Total train",    IntegerType, true)
                                  .add("Total test",     IntegerType, true)
    
    val total = spark.read.option("sep", "\t").csv(resourceDir + "total.tsv")
    
    val array = total.randomSplit(Array(0.75, 0.25), 11L)    
    val train = array(0).toDF("_c0", "_c1", "_c2")
    val test  = array(1).toDF("_c0", "_c1", "_c2")

    val entityArray   = total.select("_c0").withColumnRenamed("_c0", "_c2").union(total.select("_c2"))
                             .distinct.rdd.map(r => r(0).asInstanceOf[String]).collect
    val relationArray = total.select("_c1").distinct.rdd.map(r => r(0).asInstanceOf[String]).collect    
    
    val entityRDD   = sc.parallelize(entityArray).zipWithIndex.map(r => {Row(r._1, r._2+1)})
    val relationRDD = sc.parallelize(relationArray).zipWithIndex.map(r => {Row(r._1, r._2+1)})
    val summaryRDD  = sc.parallelize(Seq(Array(entityArray.length, relationArray.length, total.count.toInt, train.count.toInt, test.count.toInt)))
                        .map(r => {Row(r(0), r(1), r(2), r(3), r(4))})
    
    val entityDF   = spark.createDataFrame(entityRDD,  schema2)
    val relationDF = spark.createDataFrame(relationRDD,schema2)
    val summaryDF  = spark.createDataFrame(summaryRDD, schema3)
    
    writeDataframeToTSVFile(train,      resourceDir, "train.tsv")
    writeDataframeToTSVFile(test,       resourceDir, "test.tsv")
    writeDataframeToTSVFile(entityDF,   resourceDir, "entityToID.tsv")
    writeDataframeToTSVFile(relationDF, resourceDir, "relationToID.tsv") 
    writeDataframeToTSVFile(summaryDF,  resourceDir, "summary.tsv") 
    
    println("TOTAL RUNTIME: " + (System.nanoTime - t1) / 1e9d  + "s")
  }
}