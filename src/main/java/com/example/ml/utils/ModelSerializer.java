package com.example.ml.utils;

import com.example.ml.model.Model;

import java.io.*;

public class ModelSerializer {

    public static void saveModel(Model<Double, Integer> model, String filePath) throws IOException {
        try (FileOutputStream fileOut = new FileOutputStream(filePath);
             ObjectOutputStream out = new ObjectOutputStream(fileOut)) {
            out.writeObject(model);
        }
    }

    @SuppressWarnings("unchecked")
    public static Model<Double, Integer> loadModel(String filePath) throws IOException, ClassNotFoundException {
        try (FileInputStream fileIn = new FileInputStream(filePath);
             ObjectInputStream in = new ObjectInputStream(fileIn)) {
            return (Model<Double, Integer>) in.readObject();
        }
    }
}
