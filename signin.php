<?php
session_start();
require 'config.php';

// Enable error reporting
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

$error_message = "";

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $user_id = $_POST['user_id'];
    $password = $_POST['password'];

    if (empty($user_id) || empty($password)) {
        $error_message = "User ID and Password are required.";
    } else {
        try {
            // Prepare and execute SQL query
            $stmt = $pdo->prepare("SELECT name, password_hash FROM users WHERE user_id = ?");
            $stmt->execute([$user_id]);
            $user = $stmt->fetch();

            if ($user && password_verify($password, $user['password_hash'])) {
                session_regenerate_id(true); // Regenerate session ID to prevent session fixation
                $_SESSION['user_id'] = $user_id;
                $_SESSION['name'] = $user['name']; // Store name in session
                header("Location: http://localhost:8503/?user=" . urlencode($user['name']));
                exit();
            } else {
                $error_message = "Invalid User ID or Password.";
            }
        } catch (PDOException $e) {
            $error_message = "Database error: " . $e->getMessage();
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
    <link rel="stylesheet" href="css/signin.css"> <!-- Add your stylesheet -->
</head>
<body>

<header>
    <nav>
        <div>
            <img src="assets/evolkai.png" alt="Logo" class="logo">
        </div>
    </nav>
</header>

<section class="second_screen" id="second_screen">
    <div class="card">
        <div>
            <img src="assets/signin.png" alt="Sign In" class="image">
        </div>
        <div class="container">
            <form method="post" action="">
                <input type="text" id="user_id" name="user_id" placeholder="User ID" required>
                <input type="password" id="password" name="password" placeholder="Password" required>
                <div class="center">
                    <button type="submit">Sign In</button>
                </div>
            </form>
            <div class="paragraph2">
                <p>Don't have an account? <a href="index.php">Sign Up</a></p>
            </div>
            <div class="error_message">
                <?php
                if (!empty($error_message)) {
                    echo "<p>$error_message</p>";
                }
                ?>
            </div>
        </div>
    </div>
</section>

</body>
</html>
