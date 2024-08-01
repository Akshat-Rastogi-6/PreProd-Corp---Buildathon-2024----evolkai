<?php
require 'config.php'; // Include database configuration

$errors = [];

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $user_id = $_POST['user_id'];
    $name = $_POST['name']; // New field
    $email = $_POST['email'];
    $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];

    // Validate inputs
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
            $success_message = "Registration successful. You can now <a href='signin.php'>sign in</a>.";
        } else {
            $errors[] = "Registration failed. Please try again.";
        }
    }
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="css/style.css">
    <title>Evolkai - Accurate</title>
</head>
<body>
    <header>
        <nav>
            <div>
                <img src="assets/evolkai.png" alt="Evolkai Logo" class="logo">
            </div>
            <div>
                <img src="assets/accrate.png" alt="">
                <a href = "signin.php"><button class="label" style="margin-right: 5px;">Sign In</button></a>
            </div>
        </nav>
    </header>
    <section class="main">
        <h1 class="heading1">ACCURATE</h1>  
        <div class="paragraph">
            <p>
                Discover the ultimate data analysis tool that transforms any dataset into insightful metrics, delivering accuracy, precision, recall scores, and comprehensive survey results instantly. Unveil the full potential of your data today!
            </p>
        </div>

        <a href="#second_screen" class="link"><button class="label">Sign Up</button></a>
    </section>
    
    <section class="second_screen" id="second_screen">
        <div class="card">
            <div>
                <img src="assets/signup.png" alt="Sign Up" class="image">
            </div>
            <div class="container">
                <form method="post" action="">
                    <input type="text" id="user_id" name="user_id" placeholder="User ID" required>
                    <input type="text" id="name" name="name" placeholder="Name" required> <!-- New field -->
                    <input type="email" id="email" name="email" placeholder="Email" required>
                    <input type="password" id="password" name="password" placeholder="Password" required>
                    <input type="password" id="confirm_password" name="confirm_password" placeholder="Confirm Password" required>
                    <div class="center">
                        <button type="submit">Sign Up</button>
                    </div>
                </form>
                <div class="paragraph2">
                    <p>Already have an account? <a href="signin.php">Sign In</a></p>
                    <div class="error_message">
                        <?php
                        if (!empty($errors)) {
                            foreach ($errors as $error) {
                                echo "<p>$error</p>";
                            }
                        }
                        if (!empty($success_message)) {
                            echo "<p>$success_message</p>";
                        }
                        ?>
                    </div>
                </div>
            </div>
        </div>
    </section>
</body>
</html>
