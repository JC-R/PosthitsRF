/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.regression.RegressionModel
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
// $example off$
import java.io._

object RandomForestRegressorExample {

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

  def main(args: Array[String]): Unit = {

    println(s"RandomForestsRegressor") // Print model params.
    System.setProperty("hadoop.home.dir", "D:\\usr\\local\\winutils\\hadoop-2.6.0")

    val conf = new SparkConf().setAppName("RandomForestRegressorExample")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    sc.setLogLevel("ERROR")

    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    val data = sqlContext.read.format("libsvm").load("D:/sandbox/spark-1.6.1-bin-hadoop2.6/data/mllib/sample_libsvm_data.txt")

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestRegressor()
      .setLabelCol("label")
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setNumTrees(50)
      .setMaxBins(32)
      .setMaxDepth(30)
      .setFeatureSubsetStrategy("auto")

    // Chain indexer and forest in a Pipeline
    val pipeline = new Pipeline()
        .setStages(Array(featureIndexer,rf))

    // Select (prediction, true label) and compute test error
    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val paramGrid = new ParamGridBuilder()
        .addGrid(rf.maxDepth,Array(20,30))
        .addGrid(rf.numTrees,Array(25,50))
        .build() // No parameter search

    val cv = new CrossValidator()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(10)

    // Train model.  This also runs the indexer.
    val model = cv.fit(trainingData)
//    println(model.bestModel.asInstanceOf[RandomForestRegressionModel].toDebugString)

    saveModel("cv_drf.model",model)

    println("Cross validation best model: ")
    println(model.bestModel.explainParams())

    // Make predictions.
    val predictions = model.transform(testData)
    predictions.select("prediction", "label", "features").show(5)
    println("Root Mean Squared Error (RMSE) on test data = " + evaluator.evaluate(predictions))

    // reload and verify
    val newModel: CrossValidatorModel = loadModel("cv_drf.model")

    // Make predictions.
    val newPredictions = newModel.transform(testData)
    newPredictions.select("prediction", "label", "features").show(5)
    println("Root Mean Squared Error (RMSE) on test data = " + evaluator.evaluate(newPredictions))

    println("Learned regression forest model:\n")
//    val best: RandomForestModel = model.bestModel
//    val rfModel = model.bestModel.stages(1).asInstanceOf[RandomForestRegressionModel]
    // $example off$

    sc.stop()
  }
}
// scalastyle:on println
