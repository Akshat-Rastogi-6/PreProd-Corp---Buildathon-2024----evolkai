<?php
session_start();
require 'config.php'; // Database configuration

$file_path = $_SESSION['file_path'] ?? '';

if ($file_path) {
    $python_script = 'D:/Projects/PreProd-Corp---Buildathon-2024----evolkai/python_scripts/app.py'; // Full path to Python script
    $python_executable = 'C:/Path/To/Python/python.exe'; // Full path to Python executable
    $command = escapeshellcmd("$python_executable \"$python_script\" \"$file_path\"");
    
    // Log the command
    file_put_contents('command.log', "Executing command: $command" . PHP_EOL, FILE_APPEND);
    
    // Execute Python script and capture output
    $output = shell_exec($command);
    
    // Log the output
    file_put_contents('python_output.log', "Python script output: $output" . PHP_EOL, FILE_APPEND);
    
    // Process the output from Python script
    if ($output !== null) {
        $output = trim($output); // Trim any extra whitespace
        if (strpos($output, '|') !== false) {
            list($status, $rows, $columns) = explode('|', $output);
            
            if ($status === 'success') {
                // Save file metadata to the database
                $user_id = $_SESSION['user_id'];
                $file_name = basename($file_path);
                $file_type = strtolower(pathinfo($file_name, PATHINFO_EXTENSION));

                $stmt = $pdo->prepare("INSERT INTO uploaded_files (user_id, file_name, file_path, file_type, rows, columns) VALUES (?, ?, ?, ?, ?, ?)");
                $stmt->execute([$user_id, $file_name, $file_path, $file_type, $rows, $columns]);

                // Log user action
                $action_type = "File Upload";
                $action_details = "Uploaded file: $file_name with dimensions: $rows rows, $columns columns";
                $stmt = $pdo->prepare("INSERT INTO user_actions (user_id, action_type, action_details) VALUES (?, ?, ?)");
                $stmt->execute([$user_id, $action_type, $action_details]);

                $message = "File uploaded successfully. File Name: $file_name | Rows: $rows | Columns: $columns";
            } else {
                $message = "Error: $status";
            }
        } else {
            $message = "Invalid output format from Python script.";
        }
    } else {
        $message = "No output from Python script.";
    }
} else {
    $message = "No file path found.";
}

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Ingestion</title>
    <link rel="stylesheet" href="css/dashboard.css">
</head>
<body>
    <header>
        <h1>Data Ingestion</h1>
        <a href="dashboard.php">Back to Dashboard</a>
    </header>
    <div class="container">
        <div class="section">
            <h2>Data Ingestion Results</h2>
            <p><?php echo htmlspecialchars($message); ?></p>
        </div>
    </div>
</body>
</html>
