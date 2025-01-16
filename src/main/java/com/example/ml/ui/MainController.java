package com.example.ml.ui;

import com.example.ml.data.DataLoader;
import com.example.ml.data.Dataset;
import com.example.ml.data.Instance;
import com.example.ml.evaluation.EvaluationMetrics;
import com.example.ml.model.DecisionTree;
import com.example.ml.model.LogisticRegression;
import com.example.ml.model.Model;
import com.example.ml.model.Perceptron;
import com.example.ml.utils.SplitResult;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;

import java.io.File;
import java.util.List;

public class MainController {

    @FXML
    private TextField filePathField;

    @FXML
    private ComboBox<String> classifierComboBox;

    @FXML
    private VBox hyperparametersBox;

    @FXML
    private TextField trainSplitField;

    @FXML
    private TextField testSplitField;

    @FXML
    private Label accuracyLabel;

    @FXML
    private Label precisionLabel;

    @FXML
    private Label recallLabel;

    @FXML
    private Label f1ScoreLabel;

    private TextField learningRateField;
    private TextField maxEpochsField;
    private TextField batchSizeField;
    private TextField maxDepthField;
    private TextField minSamplesSplitField;

    @FXML
    public void initialize() {
        classifierComboBox.getItems().addAll("Perceptron", "Logistic Regression", "Decision Tree");
        classifierComboBox.getSelectionModel().selectFirst();
        handleClassifierSelection(null);
    }

    @FXML
    private void handleBrowse(ActionEvent event) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Select Input CSV File");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        File selectedFile = fileChooser.showOpenDialog(filePathField.getScene().getWindow());
        if (selectedFile != null) {
            filePathField.setText(selectedFile.getAbsolutePath());
        }
    }

    @FXML
    private void handleClassifierSelection(ActionEvent event) {
        hyperparametersBox.getChildren().clear();
        String selectedClassifier = classifierComboBox.getValue();

        switch (selectedClassifier) {
            case "Perceptron":
                addHyperparameterField("Learning Rate:", "learningRateField", "0.01");
                addHyperparameterField("Max Epochs:", "maxEpochsField", "50");
                addHyperparameterField("Batch Size:", "batchSizeField", "1");
                break;
            case "Logistic Regression":
                addHyperparameterField("Learning Rate:", "learningRateField", "0.01");
                addHyperparameterField("Max Epochs:", "maxEpochsField", "50");
                addHyperparameterField("Batch Size:", "batchSizeField", "16");
                break;
            case "Decision Tree":
                addHyperparameterField("Max Depth:", "maxDepthField", "7");
                addHyperparameterField("Min Samples Split:", "minSamplesSplitField", "20");
                break;
            default:
                break;
        }
    }

    private void addHyperparameterField(String labelText, String fxId, String defaultValue) {
        Label label = new Label(labelText);
        TextField textField = new TextField(defaultValue);
        textField.setId(fxId);
        hyperparametersBox.getChildren().addAll(label, textField);
    }

    @FXML
    private void handleTrain(ActionEvent event) {
        String filePath = filePathField.getText();
        if (filePath.isEmpty()) {
            showAlert(Alert.AlertType.ERROR, "Error", "Please select an input CSV file.");
            return;
        }

        String selectedClassifier = classifierComboBox.getValue();

        double trainRatio;
        double testRatio;

        try {
            trainRatio = Double.parseDouble(trainSplitField.getText()) / 100.0;
            testRatio = Double.parseDouble(testSplitField.getText()) / 100.0;

            if (trainRatio + testRatio > 1.0) {
                showAlert(Alert.AlertType.ERROR, "Error", "Train and Test splits exceed 100%.");
                return;
            }
        } catch (NumberFormatException e) {
            showAlert(Alert.AlertType.ERROR, "Error", "Invalid train-test split percentages.");
            return;
        }

        setUIEnabled(false);

        Task<Void> trainTask = new Task<>() {
            @Override
            protected Void call() {
                try {
                    DataLoader<Double, Integer> loader = new DataLoader<>(
                            Double::parseDouble,
                            Integer::parseInt
                    );
                    List<Instance<Double, Integer>> data = loader.loadFromCsv(filePath);

                    Dataset<Double, Integer> dataset = new Dataset<>(data, 8, Double::parseDouble);
                    dataset.shuffle();

                    Model<Double, Integer> model;
                    SplitResult<Double, Integer> splitResult;

                    EvaluationMetrics metrics;

                    switch (selectedClassifier) {
                        case "Perceptron":
                            learningRateField = (TextField) hyperparametersBox.lookup("#learningRateField");
                            maxEpochsField = (TextField) hyperparametersBox.lookup("#maxEpochsField");
                            batchSizeField = (TextField) hyperparametersBox.lookup("#batchSizeField");

                            double perceptronLearningRate = Double.parseDouble(learningRateField.getText());
                            int perceptronMaxEpochs = Integer.parseInt(maxEpochsField.getText());
                            int perceptronBatchSize = Integer.parseInt(batchSizeField.getText());

                            dataset.convertLabelsToMinusOne();

                            splitResult = dataset.trainTestSplit(trainRatio, true);
                            List<Instance<Double, Integer>> trainSet = splitResult.getTrainSet();
                            List<Instance<Double, Integer>> validationSet = splitResult.getValidationSet();
                            List<Instance<Double, Integer>> testSet = splitResult.getTestSet();

                            model = new Perceptron<>(8, perceptronLearningRate, perceptronMaxEpochs, perceptronBatchSize);
                            model.train(trainSet, validationSet);

                            metrics = model.test(testSet);
                            break;

                        case "Logistic Regression":
                            learningRateField = (TextField) hyperparametersBox.lookup("#learningRateField");
                            maxEpochsField = (TextField) hyperparametersBox.lookup("#maxEpochsField");
                            batchSizeField = (TextField) hyperparametersBox.lookup("#batchSizeField");

                            double lrLearningRate = Double.parseDouble(learningRateField.getText());
                            int lrMaxEpochs = Integer.parseInt(maxEpochsField.getText());
                            int lrBatchSize = Integer.parseInt(batchSizeField.getText());

                            splitResult = dataset.trainTestSplit(trainRatio, true);
                            List<Instance<Double, Integer>> trainSetLR = splitResult.getTrainSet();
                            List<Instance<Double, Integer>> validationSetLR = splitResult.getValidationSet();
                            List<Instance<Double, Integer>> testSetLR = splitResult.getTestSet();

                            model = new LogisticRegression<>(8, lrLearningRate, lrMaxEpochs, lrBatchSize);
                            model.train(trainSetLR, validationSetLR);

                            metrics = model.test(testSetLR);
                            break;

                        case "Decision Tree":
                            maxDepthField = (TextField) hyperparametersBox.lookup("#maxDepthField");
                            minSamplesSplitField = (TextField) hyperparametersBox.lookup("#minSamplesSplitField");

                            int dtMaxDepth = Integer.parseInt(maxDepthField.getText());
                            int dtMinSamplesSplit = Integer.parseInt(minSamplesSplitField.getText());

                            splitResult = dataset.trainTestSplit(trainRatio, false);
                            List<Instance<Double, Integer>> trainSetDT = splitResult.getTrainSet();
                            List<Instance<Double, Integer>> validationSetDT = splitResult.getValidationSet();
                            List<Instance<Double, Integer>> testSetDT = splitResult.getTestSet();

                            model = new DecisionTree<>(dtMaxDepth, dtMinSamplesSplit);
                            model.train(trainSetDT, validationSetDT);

                            metrics = model.test(testSetDT);
                            break;

                        default:
                            throw new IllegalArgumentException("Unsupported classifier selected.");
                    }

                    double finalAccuracy = metrics.getAccuracy() * 100.0;
                    double finalPrecision = metrics.getPrecision() * 100.0;
                    double finalRecall = metrics.getRecall() * 100.0;
                    double finalF1Score = metrics.getF1Score() * 100.0;

                    Platform.runLater(() -> {
                        accuracyLabel.setText(String.format("%.2f%%", finalAccuracy));
                        precisionLabel.setText(String.format("%.2f%%", finalPrecision));
                        recallLabel.setText(String.format("%.2f%%", finalRecall));
                        f1ScoreLabel.setText(String.format("%.2f%%", finalF1Score));

                        setUIEnabled(true);
                        showAlert(Alert.AlertType.INFORMATION, "Success", "Training and Evaluation Completed.");
                    });

                } catch (Exception e) {
                    Platform.runLater(() -> {
                        setUIEnabled(true);
                        showAlert(Alert.AlertType.ERROR, "Error", e.getMessage());
                    });
                }
                return null;
            }
        };

        new Thread(trainTask).start();
    }

    private void setUIEnabled(boolean enabled) {
        filePathField.setDisable(!enabled);
        classifierComboBox.setDisable(!enabled);
        hyperparametersBox.setDisable(!enabled);
        trainSplitField.setDisable(!enabled);
        testSplitField.setDisable(!enabled);
    }

    private void showAlert(Alert.AlertType type, String title, String message) {
        Alert alert = new Alert(type);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}
