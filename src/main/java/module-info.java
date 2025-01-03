module com.example.ml {
    requires javafx.controls;
    requires javafx.fxml;


    opens com.example.ml to javafx.fxml;
    exports com.example.ml;
}