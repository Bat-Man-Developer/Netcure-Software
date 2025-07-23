<?php
// register.php
session_start();

$error_message = '';
$success_message = '';

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);

    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error_message = "Invalid email format";
    } else {
        // Generate verification code
        $verification_code = sprintf("%06d", mt_rand(1, 999999));
        
        // In a real application, you would save this code to the database
        $_SESSION['verification_code'] = $verification_code;
        $_SESSION['registration_email'] = $email;

        // Send verification email (you'd need to implement this function)
        if (send_verification_email($email, $verification_code)) {
            $success_message = "Verification code sent to your email. Please check your inbox.";
        } else {
            $error_message = "Failed to send verification email. Please try again.";
        }
    }
}

function send_verification_email($email, $code) {
    // Implement your email sending logic here
    // For this example, we'll just return true
    return true;
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Registration</title>
    <link rel="stylesheet" href="style3.css">
</head>
<body>
    <div class="container">
        <h1>Secure Registration</h1>
        <?php if ($error_message): ?>
            <div class="error"><?php echo $error_message; ?></div>
        <?php endif; ?>
        <?php if ($success_message): ?>
            <div class="success"><?php echo $success_message; ?></div>
        <?php endif; ?>
        <form action="register.php" method="post">
            <input type="email" name="email" placeholder="Email" required>
            <button type="submit">Register</button>
        </form>
        <div class="links">
            <a href="login.php">Login</a>
        </div>
    </div>
</body>
</html>