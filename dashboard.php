<?php
session_start();
// Check if the user is logged in
if (!isset($_SESSION['user_id'])) {
    header("Location: signin.php");
    exit();
}

// Retrieve user name from session
$user_name = htmlspecialchars($_SESSION['name']);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <link rel="stylesheet" href="css/styles_ml.css">
    <style>
        .dashboard-container {
            display: flex;
            height: 100vh;
        }
        .sidebar {
            width: 20%;
            background-color: #f4f4f4;
            padding: 20px;
            box-shadow: 2px 0 5px rgba(0, 0, 0, 0.1);
        }
        .sidebar-button {
            display: block;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 15px;
            text-align: left;
            text-decoration: none;
            font-size: 16px;
            margin: 5px 0;
            cursor: pointer;
            width: 100%;
            box-sizing: border-box;
        }
        .sidebar-button:hover {
            background-color: #45a049;
        }
        .content {
            width: 80%;
            padding: 20px;
            overflow-y: auto;
        }
        .content iframe {
            width: 100%;
            height: 100%;
            border: none;
        }
    </style>
    <script>
        function loadContent(page) {
            document.getElementById('content-frame').src = page;
        }
    </script>
</head>
<body>
        <div class="welcome">
            <h1>Welcome, <?php echo $user_name; ?></h1>
        </div>
    <div class="dashboard-container">
        <div class="sidebar">
            <button class="sidebar-button" onclick="loadContent('data_ingestion.php')">Data Ingestion</button>
            <button class="sidebar-button" onclick="loadContent('data_transformation.php')">Data Transformation</button>
            <button class="sidebar-button" onclick="loadContent('data_training.php')">Data Training</button>
            <button class="sidebar-button" onclick="loadContent('freeze_learning.php')">Freeze Learning</button>
        </div>
        <div class="content">
            <iframe id="content-frame" src="data_ingestion.php"></iframe>
        </div>
    </div>
</body>
</html>
