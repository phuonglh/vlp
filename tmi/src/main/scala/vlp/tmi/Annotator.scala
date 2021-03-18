package vlp.tmi

import java.{util => ju}
import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.consumer.KafkaConsumer
import java.time.Duration
import scala.util.parsing.json.JSONObject
import java.nio.file.{Paths, Files, StandardOpenOption}
import java.text.SimpleDateFormat
import java.util.Date

/**
  * phuonglh@gmail.com
  * 
  * March, 2021
  * 
  * An utility that consumes media news from an Apache Kafka server, analyse and classify 
  * texts into either "positive" (true) or "negative" (false) category. 
  * 
  */
abstract class Classifier {
    def classify(text: String): Boolean
}

class RuleBasedClassifier(keywords: Seq[String], threshold: Float = 0.3f) extends Classifier {

    def classify(text: String): Boolean = {
        val numMatches = keywords.map(kw => if (text.indexOf(kw) > 0) 1 else 0).sum
        val score = numMatches.toFloat/keywords.size
        println(score)
        if (score >= threshold) true else false
    }
}

object Annotator {
    def run(bootstrapServers: String, date: String): Unit = {
        val props = new ju.Properties()
        props.setProperty("bootstrap.servers", bootstrapServers)
        props.setProperty("group.id", Kafka.GROUP_ID) 
        props.setProperty("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
        props.setProperty("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
        props.setProperty("enable.auto.commit", "true");
        props.setProperty("auto.commit.interval.ms", "1000");
        val consumer = new KafkaConsumer[String, String](props)
        consumer.subscribe(ju.Arrays.asList(Kafka.GROUP_ID))
        import scala.collection.JavaConversions._

        val currentThread = Thread.currentThread()
        class Shutdown extends Thread {
            override def run() {
                println("Starting exit...")
                consumer.wakeup()
                try {
                    currentThread.join()
                } catch {
                    case e: Throwable => e.printStackTrace()
                }
            }
        }
        Runtime.getRuntime().addShutdownHook(new Shutdown())
        val positive = collection.mutable.Map[String, String]()
        val negative = collection.mutable.Map[String, String]()
        val keywords = Seq("covid", "dương tính", "sars-cov-2", "truy vết", "cách ly", "cdc", "astrazeneca", "đại dịch", "vắc xin", "vaccine")
        val classifier = new RuleBasedClassifier(keywords)
        try {
            while (true) {
                val records = consumer.poll(Duration.ofMillis(100))
                for (record <- records) {
                    println(s"ofsset = ${record.offset}, key = ${record.key}")
                    val url = record.key
                    val text = record.value().toLowerCase()
                    if (classifier.classify(text))
                        positive += (url -> text)
                    else
                        negative += (url -> text)
                }
            }
        } catch {
            case wakeupException: Throwable => 
        } finally {
            consumer.close()
            println("Closed consumer and we are done.")
        }
        val jsonPositive = new JSONObject(positive.toMap)
        val jsonNegative = new JSONObject(negative.toMap)
        Files.write(Paths.get(System.getProperty("user.dir"), "dat", date + "-kafka-pos.json"), jsonPositive.toString().getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
        Files.write(Paths.get(System.getProperty("user.dir"), "dat", date + "-kafka-neg.json"), jsonNegative.toString().getBytes, StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    }

  def main(args: Array[String]): Unit = {
      val dateFormat = new SimpleDateFormat("yyyyMMdd")
      val currentDate = dateFormat.format(new Date())
      run(Kafka.SERVERS, currentDate)
  }
}