package vlp.tmi

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.streaming._
import org.apache.spark.streaming.twitter._
import org.apache.spark.streaming.StreamingContext._
import Utilities._

/** Listens to a stream of Tweets and keeps track of the most popular
 *  hashtags over a 5 minute window.
 */
object PopularHashtags {
  
  /** Our main function where the action happens */
  def main(args: Array[String]) {

    // Configure Twitter credentials using twitter.txt
    setupTwitter()
    
    // Set up a Spark streaming context named "PopularHashtags" that runs locally using
    // all CPU cores and one-second batches of data
    val ssc = new StreamingContext("local[*]", "vlp.tmi.PopularHashtags", Seconds(1))
    
    // Get rid of log spam (should be called after the context is set up)
    setupLogging()

    // Create a DStream from Twitter using our streaming context
    val tweets = TwitterUtils.createStream(ssc, None)
    
    // Now extract the text of each status update into DStreams using map()
    val statuses = tweets.map(status => status.getText())
    
    // Blow out each word into a new DStream
    val tweetwords = statuses.flatMap(tweetText => tweetText.split(" "))
    
    // Now eliminate anything that's not a hashtag
    val hashtags = tweetwords.filter(word => word.startsWith("#"))
    
    // Map each hashtag to a key/value pair of (hashtag, 1) so we can count them up by adding up the values
    val hashtagKeyValues = hashtags.map(hashtag => (hashtag, 1))
    
    // Now count them up over a 5 minute window sliding every one second
    val hashtagCounts = hashtagKeyValues.reduceByKeyAndWindow((x,y) => x + y, (x,y) => x - y, Seconds(300), Seconds(1))
    //  You will often see this written in the following shorthand:
    //val hashtagCounts = hashtagKeyValues.reduceByKeyAndWindow( _ + _, _ -_, Seconds(300), Seconds(1))
    
    // Sort the results by the count values
    val sortedResults = hashtagCounts.transform(rdd => rdd.sortBy(x => x._2, false))
    
    // Print the top 10
    sortedResults.print
    
    // Set a checkpoint directory, and kick it all off
    // I could watch this all day!
    ssc.checkpoint("/tmp/checkpoint/")
    ssc.start()
    ssc.awaitTermination()
  }  
}
