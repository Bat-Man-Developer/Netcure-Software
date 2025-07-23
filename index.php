<?php
include('layouts/header.php');
?>
  <body>
    <!--------- Dashboard-Page ------------>
    <section class="dashboard">
      <div class="dashboardcontainer" id="dashboardcontainer">
        <div class="text-center mt-3 pt-5 col-lg-6 col-md-12 col-sm-12">
          <p class="text-center" style="color: green"><?php if(isset($_GET['registermessage'])){ echo $_GET['registermessage']; }?></p>
          <p class="text-center" style="color: green"><?php if(isset($_GET['loginmessage'])){ echo $_GET['loginmessage']; }?></p>
          <p class="text-center" style="color: red"><?php if(isset($_GET['errormessage'])){ echo $_GET['errormessage']; }?></p>
          <h3>NET-CURE HOME</h3>
          <hr class="mx-auto">
          <div class="dashboardadmininfo" id="dashboardadmininfo">
          </div>
          <div class="dashboardinfo" id="dashboardinfo">
            <div class="admindashboardcontainer">
              <div class="adminidscontainer">
                <div class="adminidstable">
                  <h1 class="adminidstitle"></h1>
                  <div class="adminIdsBtnNav">
                    <a class="btn" href="index.php">Home</a>
                    <a class="btn" href="dashboard.php">Dashboard</a>
                    <a class="btn" href="ibmgranitemodels.php">IBM Granite Models</a>
                    <a class="btn" href="monitor_network.php">Live Traffic</a>
                    <a class="btn" href="model_performance.php">Model Performance</a>
                    <a class="btn" href="index.php">Refresh</a>
                  </div>
                </div><br><br>
                <h1 style="color: blue;">AI in Cybersecurity.</h1><br><br><br><br>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section><br><br><br><br><br><br><br><br><br>
  </body>
</html>