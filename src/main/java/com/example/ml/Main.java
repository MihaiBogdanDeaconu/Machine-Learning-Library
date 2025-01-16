package com.example.ml;

import com.example.ml.data.DataLoader;
import com.example.ml.data.Dataset;
import com.example.ml.data.Instance;
import com.example.ml.model.DecisionTree;
import com.example.ml.model.LogisticRegression;
import com.example.ml.model.Perceptron;
import com.example.ml.utils.SplitResult;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import java.util.List;

public class Main {
    private static final Log log = LogFactory.getLog(Main.class);

    public static void main(String[] args){
        DataLoader<Double, Integer> loader = new DataLoader<>(
                Double::parseDouble,
                Integer::parseInt
        );

        List<Instance<Double, Integer>> data = loader.loadFromCsv("/Users/mihaibogdandeaconu/Documents/Facultate/MAP/ML/src/main/resources/com/example/ml/diabetes.csv");

        Dataset<Double, Integer> dataset = new Dataset<>(data, 8, Double::parseDouble);
        dataset.shuffle();

//        dataset.convertLabelsToMinusOne();

        SplitResult<Double, Integer> splitResultNotS = dataset.trainTestSplit(0.7, false);
        List<Instance<Double, Integer>> trainSetNotS = splitResultNotS.getTrainSet();
        List<Instance<Double, Integer>> validationSetNotS = splitResultNotS.getValidationSet();
        List<Instance<Double, Integer>> testSetNotS = splitResultNotS.getTestSet();

        SplitResult<Double, Integer> splitResult = dataset.trainTestSplit(0.7, true);
        List<Instance<Double, Integer>> trainSet = splitResult.getTrainSet();
        List<Instance<Double, Integer>> validationSet = splitResult.getValidationSet();
        List<Instance<Double, Integer>> testSet = splitResult.getTestSet();

//        System.out.println(trainSet.size());
//        System.out.println(validationSet.size());
//        System.out.println(testSet.size());

//        Perceptron<Double, Integer> perceptron = new Perceptron<>(8, 0.01, 50, 1);
//        perceptron.train(trainSet, validationSet);
//        perceptron.test(testSet);

//        LogisticRegression<Double, Integer> logisticRegression = new LogisticRegression<>(8, 0.01, 50, 16);
//        logisticRegression.train(trainSet, validationSet);
//        logisticRegression.test(testSet);
//
        DecisionTree<Double, Integer> tree = new DecisionTree<>(7, 20);
        tree.train(trainSetNotS, validationSetNotS);
        tree.test(testSetNotS);


    }
}
