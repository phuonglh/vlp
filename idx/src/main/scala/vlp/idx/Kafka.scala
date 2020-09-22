package vlp.idx

import java.{util => ju}
import org.apache.kafka.clients.producer.KafkaProducer

/**
  * Kafka producer and consumer for sending/receving extracted articles.
  * <p>
  * phuonglh
  */
object Kafka {
  def createProducer(bootstrapServers: String) = {
    val props = new ju.Properties()
    props.put("bootstrap.servers", bootstrapServers)
    props.put("acks", "all")
    props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer")
    new KafkaProducer[String, String](props)
  }
}
