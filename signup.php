<?php
require 'config.php'; // Include database configuration

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $user_id = $_POST['user_id'];
    $name = $_POST['name']; // New field
    $email = $_POST['email'];
    $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];

    // Validate inputs
    $errors = [];
    if (empty($user_id) || empty($name) || empty($email) || empty($password) || empty($confirm_password)) {
        $errors[] = "All fields are required.";
    }
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $errors[] = "Invalid email format.";
    }
    if ($password !== $confirm_password) {
        $errors[] = "Passwords do not match.";
    }
    
    // Check if user ID or email already exists
    if (empty($errors)) {
        $stmt = $pdo->prepare("SELECT user_id, email FROM users WHERE user_id = ? OR email = ?");
        $stmt->execute([$user_id, $email]);
        $existingUser = $stmt->fetch();
        
        if ($existingUser) {
            if ($existingUser['user_id'] === $user_id) {
                $errors[] = "User ID already exists.";
            }
            if ($existingUser['email'] === $email) {
                $errors[] = "Email is already in use.";
            }
        }
    }

    // If no errors, hash the password and store user info
    if (empty($errors)) {
        $password_hash = password_hash($password, PASSWORD_BCRYPT);

        $sql = "INSERT INTO users (user_id, name, email, password_hash) VALUES (?, ?, ?, ?)";
        $stmt = $pdo->prepare($sql);

        if ($stmt->execute([$user_id, $name, $email, $password_hash])) {
            echo "Registration successful. You can now <a href='signin.php'>sign in</a>.";
        } else {
            echo "Registration failed. Please try again.";
        }
    } else {
        foreach ($errors as $error) {
            echo "<p>$error</p>";
        }
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up</title>
    <link rel="stylesheet" href="styles.css"> <!-- Add your stylesheet -->
</head>
<body>
    <h1>Sign Up</h1>
    <form method="post" action="">
        <label for="user_id">User ID:</label>
        <input type="text" id="user_id" name="user_id" required>
        <label for="name">Name:</label>
        <input type="text" id="name" name="name" required> <!-- New field -->
        <label for="email">Email:</label>
        <input type="email" id="email" name="email" required>
        <label for="password">Password:</label>
        <input type="password" id="password" name="password" required>
        <label for="confirm_password">Confirm Password:</label>
        <input type="password" id="confirm_password" name="confirm_password" required>
        <button type="submit">Sign Up</button>
    </form>
    <p>Already have an account? <a href="signin.php">Sign In</a></p>
</body>
</html>
