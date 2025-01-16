module com.example.ml {
    requires javafx.controls;
    requires javafx.fxml;
    requires org.knowm.xchart;
    requires java.sql;
    requires commons.logging;
    requires java.desktop;


    opens com.example.ml to javafx.fxml;
    exports com.example.ml.ui;
    opens com.example.ml.ui to javafx.fxml;
}