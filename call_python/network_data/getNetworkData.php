<?php
  class NetworkData
  {
    private $pythonExePath = "c:/Users/user/anaconda3/python.exe";
    private $scriptPaths = "c:/Xampp/htdocs/net-cure-website/python/network_data/NetworkData.py";
    private $command = "-c"; 

    private function executeCommand($scriptPath)
    {
        $escapedPythonScript = escapeshellarg($scriptPath);
        $fullCommand = $this->pythonExePath . " " . $escapedPythonScript . " " . $this->command;
        return shell_exec($fullCommand);
    }

    public function getNetworkData()
    {
        $scriptPath = $this->scriptPaths;
        return $this->executeCommand($scriptPath);
    }
  }

  $networkData = new NetworkData();
  echo $networkData->getNetworkData();