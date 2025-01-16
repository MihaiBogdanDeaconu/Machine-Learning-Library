package com.example.ml.data;

import com.example.ml.utils.SplitResult;

import java.util.ArrayList;
import java.util.List;
import java.util.Collections;
import java.util.function.Function;

public class Dataset<F extends Number, L extends Number> {
    List<Instance<F, L>> instances;
    double[] means;
    double[] stdDevs;
    private final Function<String, F> featureParser;

    public Dataset(List<Instance<F, L>> instances, int inputSize, Function<String, F> featureParser) {
        this.instances = instances;
        this.means = new double[inputSize];
        this.stdDevs = new double[inputSize];
        this.featureParser = featureParser;
    }

    public List<Instance<F, L>> getInstances() {
        return instances;
    }

    public void convertLabelsToMinusOne() {
        for (Instance<F, L> instance : instances) {
            if (instance.getLabel().intValue() == 0) {
                instance.setLabel((L) Integer.valueOf(-1));
            }
        }
    }

    public void computeMeans(List<Instance<F, L>> trainInstances) {
        for(int j = 0; j < means.length; j++) {
            double sum = 0.0;
            for(int i = 0; i < trainInstances.size(); i++){
                List<F> features = trainInstances.get(i).getFeatures();
                sum += features.get(j).doubleValue();
            }
            means[j] = sum / trainInstances.size();
        }
    }

    public void computeStdDevs(List<Instance<F, L>> trainInstances){
        for (int j = 0; j < stdDevs.length; j++) {
            double sum = 0.0;
            for(int i = 0; i < trainInstances.size(); i++){
                double deviation = trainInstances.get(i).getFeatures().get(j).doubleValue() - this.means[j];
                deviation *= deviation;
                sum += deviation;
            }
            double variance = sum / trainInstances.size();
            stdDevs[j] = Math.sqrt(variance);
            if(stdDevs[j] == 0){
                stdDevs[j] = 1;
            }
        }
    }

    public void standardize(List<Instance<F, L>> dataSet){
        for(int i = 0; i < dataSet.size(); i++){
            Instance<F, L> instance = dataSet.get(i);
            List<F> features = instance.getFeatures();
            for(int j = 0; j <  features.size(); j++){
                double scaledFeature = (features.get(j).doubleValue() - this.means[j]) / this.stdDevs[j];
                F feature = featureParser.apply(Double.toString(scaledFeature));
                instance.setFeature(feature, j);
            }
        }
    }

    public SplitResult<F, L> trainTestSplit(double trainRatio, Boolean standardize){
        int trainSize = (int) (instances.size() * trainRatio);
        int validationSize = (int) (instances.size() * 0.15);

        List<Instance<F, L>> trainSet = new ArrayList<>(instances.subList(0, trainSize));
        List<Instance<F, L>> validationSet = new ArrayList<>(instances.subList(trainSize, trainSize + validationSize));
        List<Instance<F, L>> testSet = new ArrayList<>(instances.subList(trainSize + validationSize, instances.size()));

        computeMeans(trainSet);
        computeStdDevs(trainSet);

        if(standardize){
            computeMeans(trainSet);
            computeStdDevs(trainSet);
            standardize(trainSet);
            standardize(validationSet);
            standardize(testSet);
        }

//      Debugging purposes:
        double[] checkSums = new double[means.length];
        double[] checkSqSums = new double[means.length];
        for (Instance<F, L> inst : trainSet) {
            for (int j = 0; j < means.length; j++) {
                double val = inst.getFeatures().get(j).doubleValue();
                checkSums[j] += val;
                checkSqSums[j] += val * val;
            }
        }
        for (int j = 0; j < means.length; j++) {
            double mean = checkSums[j] / trainSet.size();
            double variance = checkSqSums[j] / trainSet.size() - mean * mean;
            double std = Math.sqrt(variance);
//            System.out.println("Feature " + j + ": Mean=" + mean + ", Std=" + std);
        }
        return new SplitResult<>(trainSet, validationSet, testSet);
    }

    public void shuffle() {
        Collections.shuffle(instances);
    }
}
