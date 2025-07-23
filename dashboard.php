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
          <h3>NET-CURE DASHBOARD</h3>
          <hr class="mx-auto">
          <div class="dashboardadmininfo" id="dashboardadmininfo">
          </div>
          <div class="dashboardinfo" id="dashboardinfo">
            <div class="admindashboardcontainer">
              <div class="adminidscontainer">
                <div class="adminidstable">
                  <h1 class="adminidstitle"></h1>
                  <h2 class="adminidstitle"><p id="time"></p></h2>
                  <h3 style="color: blue; font-size: small"><p id="networkSuccessMessage">Loading Network Data...</p></h3>
                  <div class="adminIdsBtnNav">
                    <a class="btn" href="index.php">Home</a>
                    <a class="btn" href="dashboard.php">Dashboard</a>
                    <a class="btn" href="ibmgranitemodels.php">IBM Granite Models</a>
                    <a class="btn" href="monitor_network.php">Live Traffic</a>
                    <a class="btn" href="model_performance.php">Model Performance</a>
                    <a class="btn" href="dashboard.php">Refresh</a>
                  </div>
                  <div class="chartcontainer">
                      <canvas id="feature1Chart"></canvas>
                  </div>
                  <div class="chartcontainer">
                      <canvas id="feature2Chart"></canvas>
                  </div>
                  <div class="chartcontainer">
                      <canvas id="feature3Chart"></canvas>
                  </div>
                  <div class="chartcontainer">
                      <canvas id="feature4Chart"></canvas>
                  </div>
                  <div class="chartcontainer">
                      <canvas id="feature5Chart"></canvas>
                  </div>
                  <div class="chartcontainer">
                      <canvas id="feature6Chart"></canvas>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section><br><br><br><br><br><br><br><br><br>
    <script src="js/network_data/dashboard.js"></script>
  </body>
</html>