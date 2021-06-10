package vlp.tmi

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions._

import java.util.regex.Pattern
import java.util.regex.Matcher
import java.text.SimpleDateFormat
import java.util.Locale

import Utilities._

object StructuredStreaming {
  
   // Case class defining structured data for a line of Apache access log data
   case class LogEntry(ip:String, client:String, user:String, dateTime:String, request:String, status:String, bytes:String, referer:String, agent:String)
   
   val logPattern = apacheLogPattern()
   val datePattern = Pattern.compile("\\[(.*?) .+]")
      
   // Function to convert Apache log times to what Spark/SQL expects
   def parseDateField(field: String): Option[String] = {
      val dateMatcher = datePattern.matcher(field)
      if (dateMatcher.find) {
              val dateString = dateMatcher.group(1)
              val dateFormat = new SimpleDateFormat("dd/MMM/yyyy:HH:mm:ss", Locale.ENGLISH)
              val date = (dateFormat.parse(dateString))
              val timestamp = new java.sql.Timestamp(date.getTime());
              return Option(timestamp.toString())
          } else {
          None
      }
   }
   
   // Convert a raw line of Apache access log data to a structured LogEntry object (or None if line is corrupt)
   def parseLog(x:Row) : Option[LogEntry] = {     
     val matcher: Matcher = logPattern.matcher(x.getString(0)); 
     if (matcher.matches()) {
       val timeString = matcher.group(4)
       return Some(LogEntry(
           matcher.group(1),
           matcher.group(2),
           matcher.group(3),
           parseDateField(matcher.group(4)).getOrElse(""),
           matcher.group(5),
           matcher.group(6),
           matcher.group(7),
           matcher.group(8),
           matcher.group(9)
           ))
     } else {
       return None
     }
   }
  
   def main(args: Array[String]) {
      // Use new SparkSession interface in Spark 2.0
      val spark = SparkSession
        .builder
        .appName("StructuredStreaming")
        .master("local[*]")
        .config("spark.sql.streaming.checkpointLocation", "file:///tmp/checkpoint")
        .getOrCreate()
        
      setupLogging()
        
      // Create a stream of text files dumped into the logs directory
      val rawData = spark.readStream.text("/Users/phuonglh/Downloads/logs/")
            
      // Must import spark.implicits for conversion to DataSet to work!
      import spark.implicits._
            
      // Convert our raw text into a DataSet of LogEntry rows, then just select the two columns we care about
      val structuredData = rawData.flatMap(parseLog).select("status", "dateTime")
    
      // Group by status code, with a one-hour window.
      val windowed = structuredData.groupBy($"status", window($"dateTime", "1 hour")).count().orderBy("window")
      
      // Start the streaming query, dumping results to the console. Use "complete" output mode because we are aggregating
      // (instead of "append").
      val query = windowed.writeStream.outputMode("complete").format("console").start()
      
      // Keep going until we're stopped.
      query.awaitTermination()
      
      spark.stop()
   }
}