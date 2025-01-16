package com.example.ml.ui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

import java.util.logging.Level;
import java.util.logging.Logger;

public class HelloApplication extends Application {
    private static final Logger LOGGER = Logger.getLogger(HelloApplication.class.getName());

    @Override
    public void start(Stage stage) throws Exception {
        try {
            LOGGER.log(Level.INFO, "Loading FXML...");
            FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/com/example/ml/ui/main.fxml"));
            Scene scene = new Scene(fxmlLoader.load(), 800, 600);
            stage.setTitle("Machine Learning Framework");
            stage.setScene(scene);
            stage.show();
            LOGGER.log(Level.INFO, "FXML loaded successfully.");
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error loading FXML", e);
            throw e;
        }
    }

    public static void main(String[] args) {
        launch();
    }
}
