package vlp.idx

import java.{util => ju}
import org.apache.kafka.clients.producer.KafkaProducer
import org.apache.kafka.clients.consumer.KafkaConsumer
import java.time.Duration

/**
  * Kafka producer and consumer for sending/receving extracted articles.
  * <p>
  * phuonglh
  */
object Kafka {

  def createProducer(bootstrapServers: String): KafkaProducer[String, String] = {
    val props = new ju.Properties()
    props.setProperty("bootstrap.servers", bootstrapServers)
    props.setProperty("acks", "all")
    props.setProperty("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.setProperty("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    new KafkaProducer[String, String](props)
  }

  def consume(bootstrapServers: String): Unit = {
    val props = new ju.Properties()
    props.setProperty("bootstrap.servers", bootstrapServers)
    props.setProperty("group.id", "news")
    props.setProperty("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
    props.setProperty("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer")
    props.setProperty("enable.auto.commit", "true");
    props.setProperty("auto.commit.interval.ms", "1000");
    val consumer = new KafkaConsumer[String, String](props)
    consumer.subscribe(ju.Arrays.asList("news"))
    import scala.collection.JavaConversions._
    while (true) {
      val records = consumer.poll(Duration.ofMillis(100))
      for (record <- records) {
        println(s"ofsset = ${record.offset}, key = ${record.key}")
      }
    }
  }

  def main(args: Array[String]): Unit = {
    consume("http://vlp.group:9092")
  }
}
