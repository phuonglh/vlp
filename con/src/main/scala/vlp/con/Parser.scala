package vlp.con

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.dllib.keras.{Model, Sequential}
import com.intel.analytics.bigdl.dllib.keras.models.Models
import com.intel.analytics.bigdl.dllib.keras.optimizers.Adam
import com.intel.analytics.bigdl.dllib.nn.ClassNLLCriterion
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.layers._
import com.intel.analytics.bigdl.numeric.NumericFloat

import org.apache.spark.sql.SQLContext
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.Row
import com.intel.analytics.bigdl.dllib.nnframes.NNImageReader
import org.apache.hadoop.fs.Path
import com.intel.analytics.bigdl.dllib.utils.Engine
import org.apache.log4j.{Logger, Level}
import com.intel.analytics.bigdl.dllib.feature.image.ImageChannelNormalize
import com.intel.analytics.bigdl.dllib.keras.objectives.BinaryCrossEntropy
import com.intel.analytics.bigdl.dllib.optim.Top1Accuracy
import com.intel.analytics.bigdl.dllib.feature.image.RowToImageFeature
import com.intel.analytics.bigdl.dllib.feature.image.ImageResize
import com.intel.analytics.bigdl.dllib.feature.image.ImageCenterCrop
import com.intel.analytics.bigdl.dllib.feature.image.ImageMatToTensor
import com.intel.analytics.bigdl.dllib.feature.image.ImageFeatureToTensor
import com.intel.analytics.bigdl.dllib.feature.common.ChainedPreprocessing


object Parser {
  
 def buildMode(inputShape: Shape): Sequential[Float] = {
    val model = Sequential()
    model.add(Conv2D(32, 3, 3, inputShape = inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(poolSize = (2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)
    val conf = Engine.createSparkConf().setAppName("Parser").setMaster("local[*]")
      .set("spark.executor.cores", "1")
      .set("spark.cores.max", "8")
      .set("spark.executor.memory", "8g")
      .set("spark.driver.memory", "8g")
    val sparkContext = new SparkContext(conf)
    Engine.init

    val sc = NNContext.initNNContext("vlp.con.Parser")
    // read the "cats_dogs/train" folder and create labeled images: cat -> 1, dog -> 2
    val createLabel = udf { row: Row => 
      if (new Path(row.getString(0)).getName.contains("cat")) 1 else 2 
    }
    val imagePath = "dat/cats_dogs/demo/"
    val imageDF = NNImageReader.readImages(imagePath, sc, resizeH = 256, resizeW = 256)
    val df = imageDF.withColumn("label", createLabel(col("image")))
    df.printSchema()
    df.show()
    // train/test split
    val Array(trainingDF, validationDF) = df.randomSplit(Array(0.8, 0.2), seed = 80L)

    val model = buildMode(Shape(3, 256, 256))
    model.compile(optimizer = new Adam(), loss = BinaryCrossEntropy(), metrics = List(new Top1Accuracy()))

    // preprocess the images before training
    // val transformers = ChainedPreprocessing(Array(RowToImageFeature(), ImageResize(256, 256), 
    //   ImageCenterCrop(224, 224), ImageChannelNormalize(123, 117, 104), ImageMatToTensor(), ImageFeatureToTensor()))
    val transformers = ImageChannelNormalize(123, 117, 104)
    model.fit(trainingDF, batchSize = 64, nbEpoch = 5, labelCols = Array("label"), 
      transform = transformers, valX = validationDF)

    sc.stop()
  }
}
