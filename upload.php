<?php
session_start();
require 'config.php'; // Database configuration

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_FILES["file"])) {
    $file = $_FILES["file"];
    $file_name = basename($file["name"]);
    $file_tmp = $file["tmp_name"];
    $file_type = strtolower(pathinfo($file_name, PATHINFO_EXTENSION));
    $upload_dir = "uploads/";
    $file_path = $upload_dir . uniqid() . '.' . $file_type;

    // Check for upload errors
    if ($file["error"] !== UPLOAD_ERR_OK) {
        echo "Error uploading file: " . $file["error"];
        exit();
    }

    // Check if file is uploaded
    if (move_uploaded_file($file_tmp, $file_path)) {
        // Save file path to session
        $_SESSION['file_path'] = $file_path;
        header('Location: data_ingestion.php');
        exit();
    } else {
        echo "File upload failed.";
    }
} else {
    echo "No file uploaded.";
}
?>
