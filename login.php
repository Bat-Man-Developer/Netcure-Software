<?php
// login.php
session_start();

$error_message = '';

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $email = filter_var($_POST['email'], FILTER_SANITIZE_EMAIL);
    $password = $_POST['password'];

    // Check if email is valid
    if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $error_message = "Invalid email format";
    } else {
        // Check if email is in scam list (you'd need to implement this function)
        if (is_email_in_scam_list($email)) {
            $error_message = "This email is not allowed";
        } else {
            // Check password strength
            if (strlen($password) < 8 || !preg_match('/[A-Z]/', $password) || 
                !preg_match('/[a-z]/', $password) || !preg_match('/[0-9]/', $password) || 
                !preg_match('/[^A-Za-z0-9]/', $password)) {
                $error_message = "Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character";
            } else {
                // Here you would typically check the credentials against a database
                // For this example, we'll just set a session variable
                $_SESSION['user_email'] = $email;
                header("Location: dashboard.php");
                exit;
            }
        }
    }
}

function is_email_in_scam_list($email) {
    // Implement your scam list check here
    // For this example, we'll just check for a known scam domain
    $scam_domains = ['scam.com', 'fraud.com'];
    $domain = substr(strrchr($email, "@"), 1);
    return in_array($domain, $scam_domains);
}
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure Login</title>
    <link rel="stylesheet" href="css/style3.css">
</head>
<body>
    <div class="container">
        <h1>Secure Login</h1>
        <?php if ($error_message): ?>
            <div class="error"><?php echo $error_message; ?></div>
        <?php endif; ?>
        <form action="login.php" method="post">
            <input type="email" name="email" placeholder="Email" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit">Login</button>
        </form>
        <div class="links">
            <a href="register.php">Register</a>
            <a href="reset_password.php">Reset Password</a>
        </div>
    </div>
</body>
</html>