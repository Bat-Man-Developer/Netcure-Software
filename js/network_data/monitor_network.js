class NetworkData {
    constructor() {
        this.networkSuccessMessage = null;
        this.time = null;
        this.sourceIPs = null;
        this.destinationIPs = null;
        this.protocols = null;
        this.sourcePorts = null;
        this.destinationPorts = null;
        this.lengths = null;
        this.timestamps = null;
    }

    executeGetNetworkData() {
        $.ajax({
            url: '../../call_python/network_data/getNetworkData.php',
            method: 'GET',
            success: (response) => {
                const data = response.split('|');

                this.networkSuccessMessage = data[0];
                $('#networkSuccessMessage').html(this.networkSuccessMessage);

                this.time = data[1];
                $('#time').html(this.time);

                this.sourceIPs = data[2];
                $('#sourceIPs').html(this.sourceIPs);

                this.destinationIPs = data[3];
                $('#destinationIPs').html(this.destinationIPs);

                this.protocols = data[4];
                $('#protocols').html(this.protocols);

                this.sourcePorts = data[5];
                $('#sourcePorts').html(this.sourcePorts);

                this.destinationPorts = data[6];
                $('#destinationPorts').html(this.destinationPorts);

                this.lengths = data[7];
                $('#lengths').html(this.lengths);

                this.timestamps = data[8];
                $('#timestamps').html(this.timestamps);
            },
            error: (xhr, status, error) => {
                console.error('Error getting network packet data:', error);
            }
        });
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