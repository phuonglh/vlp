// Kafka setup instructions for Windows: https://dzone.com/articles/running-apache-kafka-on-windows-os

package com.sundogsoftware.sparkstreaming

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.spark.storage.StorageLevel

import java.util.regex.Pattern
import java.util.regex.Matcher

import Utilities._

import org.apache.spark.streaming.kafka._
import kafka.serializer.StringDecoder

/** Working example of listening for log data from Kafka's testLogs topic on port 9092. */
object KafkaExample {
  
  def main(args: Array[String]) {

    // Create the context with a 1 second batch size
    val ssc = new StreamingContext("local[*]", "vlp.tmi.KafkaExample", Seconds(1))
    
    setupLogging()
    
    // Construct a regular expression (regex) to extract fields from raw Apache log lines
    val pattern = apacheLogPattern()

    // hostname:port for Kafka brokers, not Zookeeper
    val kafkaParams = Map("metadata.broker.list" -> "localhost:9092")
    // List of topics you want to listen for from Kafka
    val topics = List("testLogs").toSet
    // Create our Kafka stream, which will contain (topic,message) pairs. We tack a 
    // map(_._2) at the end in order to only get the messages, which contain individual
    // lines of data.
    val lines = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder](
      ssc, kafkaParams, topics).map(_._2)
     
    // Extract the request field from each log line
    val requests = lines.map(x => {val matcher:Matcher = pattern.matcher(x); if (matcher.matches()) matcher.group(5)})
    
    // Extract the URL from the request
    val urls = requests.map(x => {val arr = x.toString().split(" "); if (arr.size == 3) arr(1) else "[error]"})
    
    // Reduce by URL over a 5-minute window sliding every second
    val urlCounts = urls.map(x => (x, 1)).reduceByKeyAndWindow(_ + _, _ - _, Seconds(300), Seconds(1))
    
    // Sort and print the results
    val sortedResults = urlCounts.transform(rdd => rdd.sortBy(x => x._2, false))
    sortedResults.print()
    
    // Kick it off
    ssc.checkpoint("/tmp/checkpoint/")
    ssc.start()
    ssc.awaitTermination()
  }
}

