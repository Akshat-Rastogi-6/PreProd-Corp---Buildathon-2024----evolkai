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
        <button class= "label"><a href="logout.php">Log Out</a></button>
    </header>
    <div class="container">
        <nav class="sidebar">
            <button onclick="showSection('data-ingestion')">Data Ingestion</button>
            <button onclick="showSection('data-transformation')">Data Transformation</button>
            <button onclick="showSection('data-training')">Data Training</button>
            <button onclick="showSection('result-analysis')">Result Analysis</button>
            <button onclick="showSection('freeze-learning')">Freeze Learning</button>
        </nav>
        <main class="content">
        <form id="upload-form" method="post" enctype="multipart/form-data">
        <input type="file" id="file" name="file" accept=".csv, .xlsx, .xls" required>
        <input type="submit" value="Ingest">
    </form>

    <?php
    if (isset($_FILES['file'])) {
        $target_dir = "uploads/";
        $target_file = $target_dir . basename($_FILES['file']['name']);
        $uploadOk = 1;
        $fileType = strtolower(pathinfo($target_file, PATHINFO_EXTENSION));

        // Check if file already exists
        if (file_exists($target_file)) {
            echo "Sorry, file already exists.";
            $uploadOk = 0;
        }

        // Check file size
        if ($_FILES['file']['size'] > 50000000000) {
            echo "Sorry, your file is too large.";
            $uploadOk = 0;
        }

        // Allow only CSV, XLS, and XLSX files
        if ($fileType != "csv" && $fileType != "xls" && $fileType != "xlsx") {
            echo "Sorry, only CSV, XLS, and XLSX files are allowed.";
            $uploadOk = 0;
        }

        // Upload the file
        if ($uploadOk == 0) {
            echo "Sorry, your file was not uploaded.";
        } else {
            if (move_uploaded_file($_FILES['file']['tmp_name'], $target_file)) {
                echo "The file ". htmlspecialchars(basename($_FILES['file']['name'])). " has been uploaded.";
            } else {
                echo "Sorry, there was an error uploading your file.";
            }
        }
    }
    ?>
        </main>
    </div>
</body>
</html>
