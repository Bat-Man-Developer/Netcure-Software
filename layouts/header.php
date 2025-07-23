<!DOCTYPE html>
	<html lang="en">
		<head>
			<meta charset="utf-8">
			<meta name="viewport" content="width=device-width, initial-scale=1.0">
			<title>NET-CURE PROTECTION</title>
			<link rel="stylesheet" type="text/css" href="../css/style.css">
			<link rel="stylesheet" type="text/css" href="../css/style2.css">
			<link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,200;0,300;0,400;0,500;1,200;1,300&display=swap" rel="stylesheet">
			<link rel="stylesheet" type="text/css" href="https://stackpath.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css">
			<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
			<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
			<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
			<div>
				<nav class="topnavbar">
					<ul>
						<a class="signout" href="index.php?adminlogout=1" name="adminlogout">Signout</a>
					</ul>
					<ul>
						<a href="#"><img class="notificationpic" src="../images/notification_pic.png" alt="Snow">
						<?php if(isset($_SESSION['totalnotificationquantity']) && $_SESSION['totalnotificationquantity'] != 0) { ?>
								<span class="notificationquantity">
									<?php echo $_SESSION['totalnotificationquantity']; ?>
								</span>
						<?php }?>
						</a>
					</ul>
					<ul>
						<a id="adminmenuicon" onclick="adminmenutoggle()" alt="Snow">Menu</a>
					</ul>
				</nav>
			</div>

			<div id="sidenav" class="sidenav">
				<div class="logo">
					<a href="index.php"><img src="../images/net-cure_logo.png" alt="Snow"></a>
				</div>
				<a href="account.php">Account</a>
				<a href="test_novel_model_performance.php">Test Model</a>
				<a href="block_ip.php">Block IP Address</a>
				<a href="zero_trust_mode.php">Activate Zero Trust Mode</a>
				<a href="alerts.php">Alerts</a>
				<a href="settings.php">Settings</a>
				<a href="help.php">Help?</a><br><br><br>
			</div>
		</head>

		<!----------js for toggle menu----------->
		<script>
			function adminmenutoggle(){ 
				var sidenav = document.getElementById("sidenav");
				sidenav.style.marginLeft == "-200px";
				if(sidenav.style.marginLeft == "0px")
				{
					sidenav.style.marginLeft = "-200px";
				}
				else
				{
					sidenav.style.marginLeft = "0px";
				}
			}
		</script>
