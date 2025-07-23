class NetworkData {
    constructor() {
        this.networkSuccessMessage = null;
        this.time = null;
        this.sourceIPs = [];
        this.destinationIPs = [];
        this.protocols = [];
        this.sourcePorts = [];
        this.destinationPorts = [];
        this.lengths = [];
        this.timestamps = [];
    }

    executeAjaxGetRequest(endpoint, callback) {
        $.ajax({
            url: endpoint,
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
                'X-XSS-Protection': '1; mode=block',
                'X-Content-Type-Options': 'nosniff',
                'X-Frame-Options': 'DENY',
                'Referrer-Policy': 'no-referrer'
            },
            success: (response) => {
                callback(response);
            },
            error: (xhr, status, error) => {
                console.error('Error getting network packets:', error);
                callback(null);
            }
        });
    }

    executeGetNetworkData() {
        this.executeAjaxGetRequest('../../call_python/network_data/getNetworkData.php', (response) => {
            if (response) {
                const data = response.split('|');

                this.networkSuccessMessage = data[0];
                $('#networkSuccessMessage').html(this.networkSuccessMessage);

                this.time = data[1];
                $('#time').html(this.time);

                this.sourceIPs = data[2].replace(/'/g, "").replace(/ /g, "").split(",");
                $('#sourceIPs').html(this.sourceIPs.join(', '));

                this.destinationIPs = data[3].replace(/'/g, "").replace(/ /g, "").split(",");
                $('#destinationIPs').html(this.destinationIPs.join(', '));

                this.protocols = data[4].replace(/'/g, "").replace(/ /g, "").split(",");
                $('#protocols').html(this.protocols.join(', '));

                this.sourcePorts = data[5].replace(/'/g, "").replace(/ /g, "").split(",");
                $('#sourcePorts').html(this.sourcePorts.join(', '));

                this.destinationPorts = data[6].replace(/'/g, "").replace(/ /g, "").split(",");
                $('#destinationPorts').html(this.destinationPorts.join(', '));

                this.lengths = data[7].replace(/'/g, "").replace(/ /g, "").split(",").map(Number);
                $('#lengths').html(this.lengths.join(', '));

                this.timestamps = data[8].replace(/'/g, "").replace(/ /g, "").split(",");
                $('#timestamps').html(this.timestamps.join(', '));

                this.updateCharts();
            }
        });
    }

    generateLineChart(elementId, labels, data, label) {
        var ctx = document.getElementById(elementId).getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    updateCharts() {
        this.generateLineChart("feature1Chart", this.sourceIPs, this.lengths, "Source IPs vs Packet Lengths");
        this.generateLineChart("feature2Chart", this.destinationIPs, this.lengths, "Destination IPs vs Packet Lengths");
        this.generateLineChart("feature3Chart", this.protocols, this.lengths, "Protocols vs Packet Lengths");
        this.generateLineChart("feature4Chart", this.sourcePorts, this.lengths, "Source Ports vs Packet Lengths");
        this.generateLineChart("feature5Chart", this.destinationPorts, this.lengths, "Destination Ports vs Packet Lengths");
        this.generateLineChart("feature6Chart", this.timestamps, this.lengths, "Timestamps vs Packet Lengths");
    }

    initializeEventListeners() {
        $(document).ready(() => {
            // Initial execution
            this.executeGetNetworkData();

            // Execute PHP code every 60 seconds
            setInterval(() => {
                this.executeGetNetworkData();
            }, 60000);
        });
    }
}

// Usage
const networkData = new NetworkData();
networkData.initializeEventListeners();