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
    <link rel="stylesheet" href="css/dashboard.css">
</head>
<body>
    <header>
        <div>
            <img src="assets/evolkai.png" alt="Evolkai Logo" class="logo">
        </div>
        <div class = "welcome">
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
    <!-- <script src="js/dashboard.js"></script> -->
</body>
</html>



<!-- <div id="data-ingestion" class="section">
                <h2>Data Ingestion</h2>
                <form id="upload-form" action="upload.php" method="post" enctype="multipart/form-data">
                    <input type="file" id="file" name="file" accept=".csv, .xlsx, .xls" required>
                    <button type="submit">Upload</button>
                </form>
            </div>
            <div id="data-transformation" class="section" style="display:none;">
                <h2>Data Transformation</h2>
                <a href="data_transformation.php">Start Transformation</a>
            </div>
            <div id="data-training" class="section" style="display:none;">
                <h2>Data Training</h2>
                <a href="data_training.php">Start Training</a>
            </div>
            <div id="result-analysis" class="section" style="display:none;">
                <h2>Result Analysis</h2>
            </div>
            <div id="freeze-learning" class="section" style="display:none;">
                <h2>Freeze Learning</h2>
            </div> -->