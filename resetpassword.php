<?php
// reset_password.php
session_start();

$error_message = '';
$success_message = '';

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);

    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error_message = "Invalid email format";
    } else {
        // Generate reset token
        $reset_token = bin2hex(random_bytes(32));
        
        // In a real application, you would save this token to the database
        $_SESSION['reset_token'] = $reset_token;
        $_SESSION['reset_email'] = $email;

        // Send reset email (you'd need to implement this function)
        if (send_reset_email($email, $reset_token)) {
            $success_message = "Password reset instructions sent to your email. Please check your inbox.";
        } else {
            $error_message = "Failed to send reset email. Please try again.";
        }
    }
}

function send_reset_email($email, $token) {
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
    <title>Reset Password</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <h1>Reset Password</h1>
        <?php if ($error_message): ?>
            <div class="error"><?php echo $error_message; ?></div>
        <?php endif; ?>
        <?php if ($success_message): ?>
            <div class="success"><?php echo $success_message; ?></div>
        <?php endif; ?>
        <form action="reset_password.php" method="post">
            <input type="email" name="email" placeholder="Email" required>
            <button type="submit">Reset Password</button>
        </form>
        <div class="links">
            <a href="login.php">Login</a>
        </div>
    </div>
</body>
</html>