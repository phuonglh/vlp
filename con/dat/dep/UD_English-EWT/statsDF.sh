val df = spark.read.parquet("trainDF")
val ef = df.select("offsets").flatMap(row => row.getSeq[String](0))
ef.show()
val gf = ef.groupBy("value").count()
gf.show()

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
val hf = gf.withColumn("i", col("value").cast(IntegerType))
hf.show()
val sf = hf.sort("i")
sf.show()

gf.select("i", "count").repartition(1).write.csv("vlp/con/dat/dep/eng/2.7/trainDF-stat")
