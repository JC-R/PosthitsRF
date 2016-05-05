// scalastyle:off println
package edu.nyu.tandon

import java.io.{FileOutputStream, ObjectOutputStream}

import org.apache.spark.examples.mllib.AbstractParams
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import scopt.OptionParser

object PostHitsRFRegression {

  // runtime parameters
  case class Params(
     inputPath: String = null,
     modelName: String = null,
     nFolds: Int = 5,
     numTrees: Int = 50,
     maxDepth: Int = 25,
     maxBins: Int = 32,
     minInstancesPerNode: Int = 1,
     minInfoGain: Double = 0.0,
     featureSubsetStrategy: String = "auto",
     fracTest: Double = 0.3,
     cacheNodeIds: Boolean = false,
     checkpointDir: Option[String] = None,
     checkpointInterval: Int = 10) extends AbstractParams[Params]

  def main(args: Array[String]): Unit = {

    val defaultParams = Params()

    val parser = new OptionParser[Params]("PostHitsRFRegression") {
      head("PostHitsRFRegression: create a random forest regression model for posthits topk.")
      opt[Int]("nFolds")
        .text(s"n fold cross validation, default: ${defaultParams.nFolds}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("maxDepth")
        .text(s"max depth of the tree, default: ${defaultParams.maxDepth}")
        .action((x, c) => c.copy(maxDepth = x))
      opt[Int]("maxBins")
        .text(s"max number of bins, default: ${defaultParams.maxBins}")
        .action((x, c) => c.copy(maxBins = x))
      opt[Int]("minInstancesPerNode")
        .text(s"min number of instances required at child nodes to create the parent split," +
          s" default: ${defaultParams.minInstancesPerNode}")
        .action((x, c) => c.copy(minInstancesPerNode = x))
      opt[Double]("minInfoGain")
        .text(s"min info gain required to create a split, default: ${defaultParams.minInfoGain}")
        .action((x, c) => c.copy(minInfoGain = x))
      opt[Int]("numTrees")
        .text(s"number of trees in ensemble, default: ${defaultParams.numTrees}")
        .action((x, c) => c.copy(numTrees = x))
      opt[String]("featureSubsetStrategy")
        .text(s"number of features to use per node (supported:" +
          s" ${RandomForestClassifier.supportedFeatureSubsetStrategies.mkString(",")})," +
          s" default: ${defaultParams.numTrees}")
        .action((x, c) => c.copy(featureSubsetStrategy = x))
      opt[Double]("fracTest")
        .text(s"fraction of data to hold out for testing, default: ${defaultParams.fracTest}")
        .action((x, c) => c.copy(fracTest = x))
      opt[Boolean]("cacheNodeIds")
        .text(s"whether to use node Id cache during training, " +
          s"default: ${defaultParams.cacheNodeIds}")
        .action((x, c) => c.copy(cacheNodeIds = x))
      opt[String]("checkpointDir")
        .text(s"checkpoint directory where intermediate node Id caches will be stored, " +
          s"default: ${
            defaultParams.checkpointDir match {
              case Some(strVal) => strVal
              case None => "None"
            }
          }")
        .action((x, c) => c.copy(checkpointDir = Some(x)))
      opt[Int]("checkpointInterval")
        .text(s"how often to checkpoint the node Id cache, " +
          s"default: ${defaultParams.checkpointInterval}")
        .action((x, c) => c.copy(checkpointInterval = x))
      opt[String]("modelName")
        .text(s"model name (required): what we are learning - top1k, etc")
        .required()
        .action((x, c) => c.copy(modelName = x))
      arg[String]("<inputPath>")
        .text("input directory (required)")
        .required()
        .action((x, c) => c.copy(inputPath = x))

      checkConfig { params =>
        if (params.fracTest < 0 || params.fracTest >= 1) {
          failure(s"fracTest ${params.fracTest} value incorrect; should be in [0,1).")
        } else {
          success
        }
      }
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      sys.exit(1)
    }
  }

  // main method
  def run(params: Params) {

    println(s"PostHits-RandomForestsRegressor with $params") // Print model params.

    val conf = new SparkConf().setAppName("PostHits-RandomForestsRegressor")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    val sqlContext = new SQLContext(sc)
    import sqlContext.implicits._

    val work_dir = params.inputPath
    val model_name = params.modelName

    // Load and parse the data file, converting it to a DataFrame.
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val data = sqlContext.read.parquet(work_dir + model_name + ".ph.train.parquet").
      na.fill(0.0).
      map(row => {
        val d = (for (i <- 1 to 55) yield row.getDouble(i)).toArray
        (row.getDouble(0), Vectors.dense(d))
      }).toDF("label", "features")

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(1.0 - params.fracTest, params.fracTest))

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setNumTrees(params.numTrees)
      .setMaxBins(params.maxBins)
      .setMaxDepth(params.maxDepth)
      .setFeatureSubsetStrategy("auto")

    // Chain indexer and forest in a Pipeline
//    val pipeline = new Pipeline()
//      .setStages(Array(featureIndexer, rf))

    // Select (prediction, true label) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val cv = new CrossValidator()
      .setNumFolds(params.nFolds)
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(new ParamGridBuilder().build())  // no grid search

    // Train model
    val model = cv.fit(trainingData)
    saveModel(work_dir + model_name + ".spark-cv-model",model)

    println("Cross validation best model: ")
    println(model.bestModel.explainParams())

    // Make predictions on held data
    val predictions = model.transform(testData)
    predictions.select("prediction", "label", "features").show(5)
    println("Root Mean Squared Error (RMSE) on test data = " + evaluator.evaluate(predictions))

    sc.stop()
  }

  def saveModel(name: String, model: CrossValidatorModel) = {
    val oos = new ObjectOutputStream(new FileOutputStream(s"$name"))
    oos.writeObject(model)
    oos.close
  }

  def loadModel(name: String): CrossValidatorModel = {
    import java.io.{ObjectInputStream,FileInputStream}
    val in = new FileInputStream(name)
    val reader = new ObjectInputStream(in)
    reader.readObject().asInstanceOf[CrossValidatorModel]
  }

}
