<?php
class Performance
{
    private $pythonExePath = "c:/Users/user/anaconda3/python.exe";
    private $scriptPaths = "c:/Xampp/htdocs/net-cure-website/python/model_performance/model_performance.py";
    private $command = "-c"; 

    private function executeCommand($scriptPath)
    {
        $escapedPythonScript = escapeshellarg($scriptPath);
        $fullCommand = $this->pythonExePath . " " . $escapedPythonScript . " " . $this->command;
        return shell_exec($fullCommand);
    }

    public function getPerformance()
    {
        $scriptPath = $this->scriptPaths;
        return $this->executeCommand($scriptPath);
    }
}

$performance = new Performance();
echo $performance->getPerformance();