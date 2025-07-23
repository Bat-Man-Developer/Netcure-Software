<?php
include('layouts/header.php');
?>
  <body>
    <!--------- Live-Network-Page ------------>
    <section class="dashboard">
      <div class="dashboardcontainer" id="dashboardcontainer">
        <div class="text-center mt-3 pt-5 col-lg-6 col-md-12 col-sm-12">
          <p class="text-center" style="color: green"><?php if(isset($_GET['registermessage'])){ echo $_GET['registermessage']; }?></p>
          <p class="text-center" style="color: green"><?php if(isset($_GET['loginmessage'])){ echo $_GET['loginmessage']; }?></p>
          <p class="text-center" style="color: red"><?php if(isset($_GET['errormessage'])){ echo $_GET['errormessage']; }?></p>
          <h3>NET-CURE LIVE NETWORK</h3>
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
                    <a class="btn" href="monitor_network.php">Refresh</a>
                  </div>
                  <table class="adminlocalnetworktraffictable">
                    <thead>
                      <tr>
                        <th scope="col">Source IP</th>
                        <th scope="col">Destination IP</th>
                        <th scope="col">Protocol</th>
                        <th scope="col">Source Port</th>
                        <th scope="col">Destination Port</th>
                        <th scope="col">Length</th>
                        <th scope="col">Timestamp</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td id="sourceIPs"></td>
                        <td id="destinationIPs"></td>
                        <td id="protocols"></td>
                        <td id="sourcePorts"></td>
                        <td id="destinationPorts"></td>
                        <td id="lengths"></td>
                        <td id="timestamps"></td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section><br><br><br><br><br><br><br><br><br>	
    <script src="js/network_data/monitor_network.js"></script>
  </body>
</html>