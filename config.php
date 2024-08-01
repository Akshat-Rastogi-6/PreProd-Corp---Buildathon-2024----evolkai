<?php
$host = 'localhost'; // database host
$db = 'accurate'; // database name
$user = 'root'; // database username
$pass = '123456'; // database password

try {
    $pdo = new PDO("mysql:host=$host;dbname=$db", $user, $pass);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    echo "Connection failed: " . $e->getMessage();
}
?>
