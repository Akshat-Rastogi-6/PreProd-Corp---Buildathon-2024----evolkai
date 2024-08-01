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
</head>
<body>
    <header>
        <h1>Welcome, <?php echo htmlspecialchars($user_name); ?></h1>
        <a href="logout.php">Log Out</a>
    </header>
    <main>
        <!-- Dashboard components: Data ingestion, transformation, training, analysis, etc. -->
    </main>
</body>
</html>
