package com.khalid.weka;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

import weka.*;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;

public class HelloWeka {
	public static void main(String[] args) throws Exception {

		Instances dataset = new Instances(new BufferedReader(new FileReader("/home/khalid/Desktop/arffData.arff")));
		System.out.println(dataset.toSummaryString());
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataset);
		saver.setFile(new File("/home/khalid/Desktop/newArffData.arff"));
		saver.writeBatch();
		int numFolds = 10;

		dataset.setClassIndex(dataset.numAttributes() - 1);
		RandomForest rf = new RandomForest();
		rf.setNumTrees(100);
		System.out.println(rf.getNumTrees()); //
		rf.buildClassifier(dataset);
		Evaluation evaluation = new Evaluation(dataset);
		System.out.println("num tree"); //
		evaluation.crossValidateModel(rf, dataset, numFolds, new Random(1));
		IBk classifier = new IBk(5);
		classifier.buildClassifier(dataset);

		/*
		 * System.out.println(evaluation.toSummaryString("\nResults\n======\n",
		 * true)); System.out.println(evaluation.toClassDetailsString());
		 * System.out.println("Results For Class -1- "); System.out.println(
		 * "Precision=  " + evaluation.precision(0)); System.out.println(
		 * "Recall=  " + evaluation.recall(0)); System.out.println(
		 * "F-measure=  " + evaluation.fMeasure(0)); System.out.println(
		 * "Results For Class -2- "); System.out.println("Precision=  " +
		 * evaluation.precision(1)); System.out.println("Recall=  " +
		 * evaluation.recall(1)); System.out.println("F-measure=  " +
		 * evaluation.fMeasure(1));
		 */
		// data without label
		Instance instanceValue1 = new SparseInstance(9);
		instanceValue1.setValue(0, 5);
		instanceValue1.setValue(1, 1);
		instanceValue1.setValue(2, 1);
		instanceValue1.setValue(3, 1);
		instanceValue1.setValue(4, 2);
		instanceValue1.setValue(5, 1);
		instanceValue1.setValue(6, 3);
		instanceValue1.setValue(7, 1);
		instanceValue1.setValue(8, 1);
		double[] prediction = classifier.distributionForInstance(instanceValue1);

		/*
		 * prediction[0] is benign and prediction[0] is malignant
		 */

		for (int i = 0; i < prediction.length; i++) {
			double d = prediction[i];
			System.out.println(d);
		}
		if (prediction[1] > prediction[0]) {
			System.out.println("malignant");
		} else {
			System.out.println("benign");
		}
	}
}
