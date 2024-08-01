<?php
session_start();

// Check if the user is logged in
if (!isset($_SESSION['user_id'])) {
    header("Location: signin.php");
    exit();
}

// Retrieve user name from session
$user_name = $_SESSION['name'];
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
        <h1>Welcome, <?php echo htmlspecialchars($user_name); ?></h1>
        <a href="logout.php">Log Out</a>
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
            <div id="data-ingestion" class="section">
                <h2>Data Ingestion</h2>
                <form id="upload-form" action="upload.php" method="post" enctype="multipart/form-data">
                    <input type="file" id="file" name="file" accept=".csv, .xlsx, .xls" required>
                    <button type="submit">Upload</button>
                </form>
            </div>
            <div id="data-transformation" class="section" style="display:none;">
                <h2>Data Transformation</h2>
                <!-- Content for Data Transformation -->
                <a href="data_transformation.php">Start Transformation</a>
            </div>
            <div id="data-training" class="section" style="display:none;">
                <h2>Data Training</h2>
                <!-- Content for Data Training -->
                <a href="data_training.php">Start Training</a>
            </div>
            <div id="result-analysis" class="section" style="display:none;">
                <h2>Result Analysis</h2>
                <!-- Content for Result Analysis -->
            </div>
            <div id="freeze-learning" class="section" style="display:none;">
                <h2>Freeze Learning</h2>
                <!-- Content for Freeze Learning -->
            </div>
        </main>
    </div>
    <script src="js/dashboard.js"></script>
</body>
</html>
