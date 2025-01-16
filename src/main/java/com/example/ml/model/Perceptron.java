package com.example.ml.model;

import com.example.ml.data.Instance;
import com.example.ml.evaluation.*;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import java.io.Serializable;
import java.util.*;

public class Perceptron<F extends Number, L extends Number> implements Model<F, L>, Serializable {
    private static final long serialVersionUID = 1L;

    double[] weights;
    double[] checkpointWeights;
    double bias;
    double checkpointBias;
    double learningRate;
    int maxEpochs;
    int batchSize;
    double bestAccuracy;

    public Perceptron(int inputSize, double learningRate, int maxEpochs, int batchSize) {
        this.weights = new double[inputSize];
        this.checkpointWeights = new double[inputSize];
        Random rand = new Random(42);
        for (int i = 0; i < inputSize; i++) {
            this.weights[i] = rand.nextDouble() * 0.02 - 0.01;
        }
        this.bias = rand.nextDouble() * 0.02 - 0.01;
        this.checkpointBias = 0.0;
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;
        this.batchSize = batchSize;
        this.bestAccuracy = 0.0;
    }

    @Override
    public void train(List<Instance<F, L>> trainSet, List<Instance<F, L>> validationSet) {
        List<Double> validationAccuracies = new ArrayList<>();

        for(int epoch = 0; epoch < this.maxEpochs; epoch++) {
            Collections.shuffle(trainSet);

            for(int i = 0; i < trainSet.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainSet.size());
                List<Instance<F, L>> batch = trainSet.subList(i, end);

                double[] weightsUpdate = new double[weights.length];
                double biasUpdate = 0.0;
                for(Instance<F, L> instance : batch) {
                    List<F> features = instance.getFeatures();
                    int label = instance.getLabel().intValue();
                    int prediction = predict(instance);
                    int error = (label - prediction) / 2;
                    for(int w = 0; w < weightsUpdate.length; w++) {
                        weightsUpdate[w] += learningRate * error * features.get(w).doubleValue();
                    }
                    biasUpdate += learningRate * error;
                }

                updateWeights(weightsUpdate, biasUpdate);
            }
            validate(validationSet, validationAccuracies, epoch);
        }
        plotValidationAccuracies(validationAccuracies);
    }

    @Override
    public EvaluationMetrics test(List<Instance<F, L>> testSet){
        // Use best checkpoint
        System.arraycopy(checkpointWeights, 0, weights, 0, checkpointWeights.length);
        bias = checkpointBias;

        List<L> testPredictions = getPredictions(testSet);
        Accuracy<F, L> accuracy = new Accuracy<>((L) Integer.valueOf(1), (L) Integer.valueOf(-1));
        Precision<F, L> precision = new Precision<>((L) Integer.valueOf(1), (L) Integer.valueOf(-1));
        Recall<F, L> recall = new Recall<>((L) Integer.valueOf(1), (L) Integer.valueOf(-1));
        F1Score<F, L> f1Score = new F1Score<>((L) Integer.valueOf(1), (L) Integer.valueOf(-1));
        double testAccuracy = accuracy.evaluate(testSet, testPredictions);
        double testPrecision = precision.evaluate(testSet, testPredictions);
        double testRecall = recall.evaluate(testSet, testPredictions);
        double testF1Score = f1Score.evaluate(testSet, testPredictions);

        // Return metrics instead of printing
        return new EvaluationMetrics(testAccuracy, testPrecision, testRecall, testF1Score);
    }

    public void validate(List<Instance<F, L>> validationSet, List<Double> validationAccuracies, int epoch) {
        List<L> valPredictions = getPredictions(validationSet);
        Accuracy<F, L> accuracy = new Accuracy<>((L) Integer.valueOf(1), (L) Integer.valueOf(-1));
        double valAccuracy = accuracy.evaluate(validationSet, valPredictions);
        if(valAccuracy > bestAccuracy){
            bestAccuracy = valAccuracy;
            System.arraycopy(weights, 0, checkpointWeights, 0, weights.length);
            checkpointBias = bias;
        }
        validationAccuracies.add(valAccuracy);
        System.out.println("Validation Accuracy after epoch " + (epoch + 1) + ": " + valAccuracy);
    }

    public List<L> getPredictions(List<Instance<F, L>> dataSet){
        List<L> predictions = new ArrayList<>();
        for (Instance<F, L> instance : dataSet) {
            int prediction = predict(instance);
            predictions.add((L)(Integer.valueOf(prediction)));
        }
        return predictions;
    }

    public int predict(Instance<F, L> instance) {
        List<F> features = instance.getFeatures();
        double output = this.bias;
        for(int i = 0; i < weights.length; i++) {
            output += features.get(i).doubleValue() * weights[i];
        }
        if(output >= 0) {
            return 1;
        }
        else{
            return -1;
        }
    }

    public void updateWeights(double[] weightsUpdate, double biasUpdate) {
        for(int i = 0; i < weights.length; i++) {
            weights[i] += weightsUpdate[i];
        }
        bias += biasUpdate;
    }

    private void plotValidationAccuracies(List<Double> validationAccuracies) {
        List<Integer> epochs = new ArrayList<>();
        for(int i = 1; i <= validationAccuracies.size(); i++) {
            epochs.add(i);
        }

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title("Validation Accuracy over Epochs")
                .xAxisTitle("Epoch")
                .yAxisTitle("Validation Accuracy")
                .build();

        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setYAxisDecimalPattern("#0.00");

        chart.addSeries("Validation Accuracy", epochs, validationAccuracies);

        new SwingWrapper<>(chart).displayChart();
    }
}
