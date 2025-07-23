<?php
  include('layouts/header.php');
?>
  <body>
    <!--------- Model Performance-Page ------------>
    <section class="dashboard">
      <div class="dashboardcontainer" id="dashboardcontainer">
        <div class="text-center mt-3 pt-5 col-lg-6 col-md-12 col-sm-12">
          <p class="text-center" style="color: green"><?php if(isset($_GET['registermessage'])){ echo $_GET['registermessage']; }?></p>
          <p class="text-center" style="color: green"><?php if(isset($_GET['loginmessage'])){ echo $_GET['loginmessage']; }?></p>
          <p class="text-center" style="color: red"><?php if(isset($_GET['errormessage'])){ echo $_GET['errormessage']; }?></p>
          <h3>NET-CURE PERFORMANCE</h3>
          <hr class="mx-auto">
          <div class="dashboardadmininfo" id="dashboardadmininfo">
          </div>
          <div class="dashboardinfo" id="dashboardinfo">
            <div class="admindashboardcontainer">
              <div class="adminidscontainer">
                <div class="adminidstable">
                  <h1 class="adminidstitle"></h1>
                  <h2 class="adminidstitle"><p id="localnetworktraffictime"></p></h2>
                  <h3 style="color: blue; font-size: small"><p id="responseMessage">Loading Model Performance...</p></h3>
                  <div class="adminIdsBtnNav">
                    <a class="btn" href="index.php">Home</a>
                    <a class="btn" href="dashboard.php">Dashboard</a>
                    <a class="btn" href="ibmgranitemodels.php">IBM Granite Models</a>
                    <a class="btn" href="monitor_network.php">Live Traffic</a>
                    <a class="btn" href="model_performance.php">Model Performance</a>
                    <a style="background-color: gold;" class="btn" href="run_model.php">Run Model</a>
                    <a class="btn" href="model_performance.php">Refresh</a>
                  </div>
                  <main>
                    <h1>Model Performance</h1>
                    <div class="container">
                      <div class="plot-container">
                        <div class="plot">
                          <h2>Confusion Matrix</h2>
                          <img id="confusion-matrix" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Individual vs Advocated Model Performance Metrics</h2>
                          <img id="individual-vs-advocated-model-performance-metrics" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Model Weights</h2>
                          <img id="model-weights" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Prediction Uncertainty</h2>
                          <img id="prediction-uncertainty" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Feature Importance</h2>
                          <img id="feature-importance" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Temporal Analysis</h2>
                          <img id="temporal-analysis" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Feature Correlation Heatmap</h2>
                          <img id="correlation-heatmap" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Flow Duration Analysis</h2>
                          <img id="flow-duration-analysis" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Attack Clustering</h2>
                          <img id="attack-clustering" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Protocol Distribution</h2>
                          <img id="protocol-distribution" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Distribution of Source Port By Label</h2>
                          <img id="distribution-source-port" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Distribution of Destination Port By Label</h2>
                          <img id="distribution-destination-port" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>Distribution of Protocol By Label</h2>
                          <img id="distribution-protocol" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 0</h2>
                          <img id="roc-curves-class-0" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 1</h2>
                          <img id="roc-curves-class-1" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 2</h2>
                          <img id="roc-curves-class-2" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 3</h2>
                          <img id="roc-curves-class-3" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 4</h2>
                          <img id="roc-curves-class-4" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 5</h2>
                          <img id="roc-curves-class-5" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 6</h2>
                          <img id="roc-curves-class-6" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 7</h2>
                          <img id="roc-curves-class-7" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 8</h2>
                          <img id="roc-curves-class-8" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 9</h2>
                          <img id="roc-curves-class-9" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 10</h2>
                          <img id="roc-curves-class-10" alt="Loading Image...">
                        </div>
                        <div class="plot">
                          <h2>ROC Curves Class 11</h2>
                          <img id="roc-curves-class-11" alt="Loading Image...">
                        </div>
                      </div>
                    </div>
                  </main>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section><br><br><br><br><br><br><br><br><br>
    <script src="js/model_performance/model_performance.js"></script>
  </body>
</html>