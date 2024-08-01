<?php
session_start();

if (!isset($_SESSION['email'])) {
    header("Location: signin.php");
    exit();
}
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
        <h1>Welcome, <?php echo htmlspecialchars($_SESSION['email']); ?></h1>
        <a href="logout.php">Log Out</a>
    </header>
    <main>
        <!-- Dashboard components: Data ingestion, transformation, training, analysis, etc. -->
    </main>
</body>
</html>
