<?php
session_start();
require 'config.php';

// Enable error reporting
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $user_id = $_POST['user_id'];
    $password = $_POST['password'];

    if (empty($user_id) || empty($password)) {
        echo "User ID and Password are required.";
    } else {
        try {
            // Prepare and execute SQL query
            $stmt = $pdo->prepare("SELECT name, password_hash FROM users WHERE user_id = ?");
            $stmt->execute([$user_id]);
            $user = $stmt->fetch();

            if ($user && password_verify($password, $user['password_hash'])) {
                $_SESSION['user_id'] = $user_id;
                $_SESSION['name'] = $user['name']; // Store name in session
                header("Location: dashboard.php");
                exit();
            } else {
                echo "Invalid User ID or Password.";
            }
        } catch (PDOException $e) {
            echo "Database error: " . $e->getMessage();
        }
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign In</title>
    <link rel="stylesheet" href="styles.css"> <!-- Add your stylesheet -->
</head>
<body>
    <h1>Sign In</h1>
    <form method="post" action="">
        <label for="user_id">User ID:</label>
        <input type="text" id="user_id" name="user_id" required>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required>
        <button type="submit">Sign In</button>
    </form>
    <p>Don't have an account? <a href="signup.php">Sign Up</a></p>
</body>
</html>
