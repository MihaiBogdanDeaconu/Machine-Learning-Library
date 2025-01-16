package com.example.ml.model;

import com.example.ml.data.Instance;
import com.example.ml.evaluation.*;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import org.knowm.xchart.XYChartBuilder;

import java.io.Serializable;
import java.util.*;

public class LogisticRegression<F extends Number, L extends Number> implements Model<F, L>, Serializable {
    private static final long serialVersionUID = 1L;

    double[] weights;
    double[] checkpointWeights;
    double bias;
    double checkpointBias;
    double learningRate;
    int maxEpochs;
    int batchSize;
    double bestAccuracy;

    public LogisticRegression(int inputSize, double learningRate, int maxEpochs, int batchSize) {
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

    public double sigmoid(double z) {
        return 1.0 / (1 + Math.exp(-z));
    }

    @Override
    public void train(List<Instance<F, L>> trainSet, List<Instance<F, L>> validationSet) {
        List<Double> validationAccuracies = new ArrayList<>();
        List<Double> losses = new ArrayList<>();

        for(int epoch = 0; epoch < this.maxEpochs; epoch++) {
            Collections.shuffle(trainSet);

            for(int i = 0; i < trainSet.size(); i += batchSize) {
                int end = Math.min(i + batchSize, trainSet.size());
                List<Instance<F, L>> batch = trainSet.subList(i, end);

                double[] weightsUpdate = new double[weights.length];
                double biasUpdate = 0.0;

                for(Instance<F, L> instance : batch) {
                    List<F> features = instance.getFeatures();
                    double label = instance.getLabel().doubleValue();
                    double prediction = predict(instance);
                    double error = prediction - label;
                    for(int w = 0; w < weightsUpdate.length; w++) {
                        weightsUpdate[w] += error * features.get(w).doubleValue();
                    }
                    biasUpdate += error;
                }
                int batchCount = end - i;
                for (int w = 0; w < weights.length; w++) {
                    weightsUpdate[w] /= batchCount;
                }
                biasUpdate /= batchCount;

                updateWeights(weightsUpdate, biasUpdate);
            }
            validate(validationSet, validationAccuracies, epoch);
            losses.add(computeLoss(validationSet));
            System.out.println("Loss: " + computeLoss(validationSet));
        }
        plot(validationAccuracies, "Validation Accuracies over epochs", "ValidationAccuracy");
        plot(losses, "Loss over epochs", "Loss");
    }

    @Override
    public EvaluationMetrics test(List<Instance<F, L>> testSet){
        // Use best checkpoint
        System.arraycopy(checkpointWeights, 0, weights, 0, checkpointWeights.length);
        bias = checkpointBias;

        List<L> testPredictions = getPredictions(testSet);
        Accuracy<F, L> accuracy = new Accuracy<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        Precision<F, L> precision = new Precision<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        Recall<F, L> recall = new Recall<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        F1Score<F, L> f1Score = new F1Score<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
        double testAccuracy = accuracy.evaluate(testSet, testPredictions);
        double testPrecision = precision.evaluate(testSet, testPredictions);
        double testRecall = recall.evaluate(testSet, testPredictions);
        double testF1Score = f1Score.evaluate(testSet, testPredictions);

        // Return metrics instead of printing
        return new EvaluationMetrics(testAccuracy, testPrecision, testRecall, testF1Score);
    }

    public double computeLoss(List<Instance<F, L>> dataset) {
        double totalLoss = 0.0;
        int m = dataset.size();

        for(Instance<F, L> instance : dataset) {
            double prediction = predict(instance);
            double y = instance.getLabel().doubleValue();
            double epsilon = 1e-15;

            prediction = Math.max(epsilon, prediction);
            prediction = Math.min(1 - epsilon, prediction);

            totalLoss += y * Math.log(prediction) + (1 - y) * Math.log(1 - prediction);
        }

        return - totalLoss / m ;
    }

    public void validate(List<Instance<F, L>> validationSet, List<Double> validationAccuracies, int epoch) {
        List<L> valPredictions = getPredictions(validationSet);
        Accuracy<F, L> accuracy = new Accuracy<>((L) Integer.valueOf(1), (L) Integer.valueOf(0));
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
        List<L> preds = new ArrayList<>();
        for (Instance<F, L> inst : dataSet) {
            double prediction = predict(inst);
            int predLabel = (prediction >= 0.5) ? 1 : 0;
            preds.add((L)(Integer.valueOf(predLabel)));
        }
        return preds;
    }

    public double predict(Instance<F, L> instance) {
        List<F> features = instance.getFeatures();
        double output = this.bias;
        for(int i = 0; i < weights.length; i++) {
            output += features.get(i).doubleValue() * weights[i];
        }
        return sigmoid(output);
    }

    public void updateWeights(double[] weightsUpdate, double biasUpdate) {
        for(int i = 0; i < weights.length; i++) {
            weights[i] -= learningRate * weightsUpdate[i];
        }
        bias -= learningRate * biasUpdate;
    }

    private void plot(List<Double> validationAccuracies, String title1, String title2) {
        List<Integer> epochs = new ArrayList<>();
        for(int i = 1; i <= validationAccuracies.size(); i++) {
            epochs.add(i);
        }

        XYChart chart = new XYChartBuilder()
                .width(800)
                .height(600)
                .title(title1)
                .xAxisTitle("Epoch")
                .yAxisTitle(title2)
                .build();

        chart.getStyler().setMarkerSize(6);
        chart.getStyler().setLegendVisible(false);
        chart.getStyler().setYAxisDecimalPattern("#0.00");

        chart.addSeries("Validation Accuracy", epochs, validationAccuracies);

        new SwingWrapper<>(chart).displayChart();
    }
}
