#!/usr/bin/bash

 # run in spark cluster
 spark-submit --master yarn --executor-memory 8g --driver-memory 8g --class edu.nyu.tandon.PostHitsRFRegression --jars ~/toolkits/spark-1.6.1-bin-hadoop2.6/lib/spark-examples-1.6.1-hadoop2.6.0.jar --deploy-mode cluster SparkExample.jar --modelName top1k /user/juanr/data/
