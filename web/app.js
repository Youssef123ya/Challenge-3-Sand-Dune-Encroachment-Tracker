// Sand Dune Encroachment Tracker Application
class SandDuneTracker {
    constructor() {
        this.data = {
            "monitoring_sites": [
                {
                    "id": "site_001",
                    "name": "Al-Ahsa Governorate",
                    "location": {"lat": 25.4, "lon": 49.6},
                    "current_position": {"x": 1250, "y": 800},
                    "movement_rate": 13.2,
                    "risk_level": "HIGH",
                    "last_updated": "2025-09-27T00:00:00Z",
                    "satellite_coverage": "Sentinel-1/2, Landsat-8"
                },
                {
                    "id": "site_002", 
                    "name": "Rigboland Sand Sea",
                    "location": {"lat": 33.8, "lon": 51.9},
                    "current_position": {"x": 980, "y": 1200},
                    "movement_rate": 8.7,
                    "risk_level": "MEDIUM",
                    "last_updated": "2025-09-27T00:00:00Z",
                    "satellite_coverage": "Sentinel-1/2"
                },
                {
                    "id": "site_003",
                    "name": "Middle Draa Region",
                    "location": {"lat": 30.2, "lon": -5.8},
                    "current_position": {"x": 1450, "y": 950},
                    "movement_rate": 6.3,
                    "risk_level": "LOW",
                    "last_updated": "2025-09-27T00:00:00Z",
                    "satellite_coverage": "Landsat-8, SPOT-6"
                }
            ],
            "infrastructure": [
                {"id": "infra_001", "name": "Highway 95", "type": "road", "position": {"x": 1300, "y": 850}, "risk_distance": 45},
                {"id": "infra_002", "name": "Agricultural Zone A", "type": "agriculture", "position": {"x": 1180, "y": 780}, "risk_distance": 78},
                {"id": "infra_003", "name": "Water Treatment Plant", "type": "facility", "position": {"x": 1400, "y": 920}, "risk_distance": 120},
                {"id": "infra_004", "name": "Village Settlement", "type": "residential", "position": {"x": 980, "y": 1150}, "risk_distance": 32}
            ],
            "predictions": [
                {
                    "model": "CNN",
                    "accuracy": 0.89,
                    "prediction": {"x_movement": 4.2, "y_movement": -2.1},
                    "confidence": 0.85
                },
                {
                    "model": "LSTM",
                    "accuracy": 0.84,
                    "prediction": {"x_movement": 3.8, "y_movement": -1.9},
                    "confidence": 0.78
                },
                {
                    "model": "Hybrid CNN-LSTM",
                    "accuracy": 0.93,
                    "prediction": {"x_movement": 4.0, "y_movement": -2.0},
                    "confidence": 0.91
                },
                {
                    "model": "Ensemble",
                    "accuracy": 0.95,
                    "prediction": {"x_movement": 4.1, "y_movement": -2.0},
                    "confidence": 0.92,
                    "uncertainty": {"x_std": 0.3, "y_std": 0.2}
                }
            ],
            "alerts": [
                {
                    "id": "alert_001",
                    "type": "INFRASTRUCTURE_RISK",
                    "severity": "HIGH",
                    "message": "Village Settlement at critical risk in 12 days",
                    "timestamp": "2025-09-27T01:15:00Z",
                    "site_id": "site_002"
                },
                {
                    "id": "alert_002", 
                    "type": "MOVEMENT_ALERT",
                    "severity": "WARNING",
                    "message": "Accelerated movement detected: 13.2 m/month",
                    "timestamp": "2025-09-27T00:45:00Z",
                    "site_id": "site_001"
                }
            ],
            "historical_data": [
                {"date": "2025-09-20", "movement": 3.2, "accuracy": 0.87},
                {"date": "2025-09-21", "movement": 2.8, "accuracy": 0.89},
                {"date": "2025-09-22", "movement": 4.1, "accuracy": 0.91},
                {"date": "2025-09-23", "movement": 3.7, "accuracy": 0.88},
                {"date": "2025-09-24", "movement": 4.5, "accuracy": 0.93},
                {"date": "2025-09-25", "movement": 3.9, "accuracy": 0.90},
                {"date": "2025-09-26", "movement": 4.2, "accuracy": 0.92}
            ],
            "config": {
                "movement_threshold": 5.0,
                "risk_zones": {"high": 50, "medium": 100, "low": 200},
                "prediction_horizon": 365,
                "confidence_intervals": [0.68, 0.95],
                "update_frequency": "6 hours"
            }
        };

        this.selectedSite = null;
        this.charts = {};
        this.accessibilityMode = false;

        this.init();
    }

    init() {
        this.setupTabNavigation();
        this.setupAccessibility();
        this.renderDashboard();
        this.setupEventListeners();
        this.populateDropdowns();
    }

    populateDropdowns() {
        // Ensure model selection dropdown is properly populated
        const modelSelect = document.getElementById('modelSelect');
        if (modelSelect) {
            modelSelect.innerHTML = '';
            this.data.predictions.forEach((prediction, index) => {
                const option = document.createElement('option');
                option.value = prediction.model;
                option.textContent = prediction.model;
                if (prediction.model === 'Ensemble') {
                    option.selected = true;
                }
                modelSelect.appendChild(option);
            });
        }
    }

    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.nav-tab');
        const tabContents = document.querySelectorAll('.tab-content');

        tabButtons.forEach(button => {
            button.addEventListener('click', () => {
                const targetTab = button.dataset.tab;
                
                // Update active states
                tabButtons.forEach(btn => btn.classList.remove('active'));
                tabContents.forEach(content => content.classList.remove('active'));
                
                button.classList.add('active');
                document.getElementById(targetTab).classList.add('active');

                // Initialize tab-specific content
                this.initializeTab(targetTab);
            });
        });
    }

    setupAccessibility() {
        const accessibilityToggle = document.getElementById('accessibilityToggle');
        accessibilityToggle.addEventListener('click', () => {
            this.accessibilityMode = !this.accessibilityMode;
            document.body.classList.toggle('accessibility-high-contrast', this.accessibilityMode);
            document.body.classList.toggle('accessibility-large-text', this.accessibilityMode);
            
            accessibilityToggle.textContent = this.accessibilityMode ? 
                '🔍 Normal View' : '🔍 Accessibility';
        });
    }

    setupEventListeners() {
        // Prediction controls
        document.getElementById('runPrediction').addEventListener('click', () => {
            this.runPrediction();
        });

        // Configuration controls
        document.getElementById('saveConfig').addEventListener('click', () => {
            this.saveConfiguration();
        });

        document.getElementById('exportReport').addEventListener('click', () => {
            this.exportReport();
        });
    }

    initializeTab(tabName) {
        switch(tabName) {
            case 'dashboard':
                this.renderDashboard();
                break;
            case 'predictions':
                this.renderPredictions();
                break;
            case 'risk-assessment':
                this.renderRiskAssessment();
                break;
            case 'analysis':
                this.renderAnalysis();
                break;
            case 'configuration':
                this.renderConfiguration();
                break;
        }
    }

    renderDashboard() {
        this.updateKPIs();
        this.renderMap();
        this.renderSiteList();
        this.renderAlertsList();
    }

    updateKPIs() {
        const alertCount = this.data.alerts.length;
        const avgMovement = (this.data.monitoring_sites.reduce((sum, site) => sum + site.movement_rate, 0) / this.data.monitoring_sites.length).toFixed(1);
        const modelAccuracy = Math.max(...this.data.predictions.map(p => p.accuracy * 100));
        const activeSites = this.data.monitoring_sites.length;

        document.getElementById('activeAlerts').textContent = alertCount;
        document.getElementById('avgMovement').textContent = avgMovement;
        document.getElementById('modelAccuracy').textContent = `${modelAccuracy.toFixed(0)}%`;
        document.getElementById('activeSites').textContent = activeSites;
    }

    renderMap() {
        const mapContainer = document.getElementById('mapContainer');
        
        // Clear existing content except legend
        const existingContent = mapContainer.querySelectorAll(':not(.map-legend)');
        existingContent.forEach(el => el.remove());

        // Create SVG for the map
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.setAttribute('viewBox', '0 0 1600 1400');
        svg.style.position = 'absolute';
        svg.style.top = '0';
        svg.style.left = '0';

        // Render monitoring sites
        this.data.monitoring_sites.forEach(site => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', site.current_position.x);
            circle.setAttribute('cy', site.current_position.y);
            circle.setAttribute('r', '20');
            
            const color = site.risk_level === 'HIGH' ? '#ff5459' : 
                         site.risk_level === 'MEDIUM' ? '#e68161' : '#32b8cd';
            circle.setAttribute('fill', color);
            circle.setAttribute('stroke', '#fff');
            circle.setAttribute('stroke-width', '2');
            circle.style.cursor = 'pointer';
            
            // Add click event
            circle.addEventListener('click', () => {
                this.selectSite(site);
            });

            // Add movement vector
            const vector = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            vector.setAttribute('x1', site.current_position.x);
            vector.setAttribute('y1', site.current_position.y);
            vector.setAttribute('x2', site.current_position.x + (site.movement_rate * 3));
            vector.setAttribute('y2', site.current_position.y - (site.movement_rate * 2));
            vector.setAttribute('stroke', color);
            vector.setAttribute('stroke-width', '3');
            vector.setAttribute('marker-end', 'url(#arrowhead)');

            svg.appendChild(circle);
            svg.appendChild(vector);

            // Add label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', site.current_position.x);
            text.setAttribute('y', site.current_position.y - 30);
            text.setAttribute('text-anchor', 'middle');
            text.setAttribute('fill', '#fff');
            text.setAttribute('font-size', '14');
            text.setAttribute('font-weight', 'bold');
            text.textContent = site.name;
            svg.appendChild(text);
        });

        // Render infrastructure
        this.data.infrastructure.forEach(infra => {
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', infra.position.x - 8);
            rect.setAttribute('y', infra.position.y - 8);
            rect.setAttribute('width', '16');
            rect.setAttribute('height', '16');
            rect.setAttribute('fill', '#a7a9a9');
            rect.setAttribute('stroke', '#fff');
            rect.setAttribute('stroke-width', '1');
            svg.appendChild(rect);
        });

        // Add arrowhead marker
        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '7');
        marker.setAttribute('refX', '9');
        marker.setAttribute('refY', '3.5');
        marker.setAttribute('orient', 'auto');
        
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3.5, 0 7');
        polygon.setAttribute('fill', '#32b8cd');
        
        marker.appendChild(polygon);
        defs.appendChild(marker);
        svg.appendChild(defs);

        mapContainer.appendChild(svg);
    }

    renderSiteList() {
        const siteList = document.getElementById('siteList');
        siteList.innerHTML = '';

        this.data.monitoring_sites.forEach(site => {
            const siteItem = document.createElement('div');
            siteItem.className = 'site-item';
            if (this.selectedSite && this.selectedSite.id === site.id) {
                siteItem.classList.add('selected');
            }
            
            siteItem.innerHTML = `
                <div class="site-header">
                    <div class="site-name">${site.name}</div>
                    <span class="status status--${site.risk_level.toLowerCase()}">${site.risk_level}</span>
                </div>
                <div class="site-info">
                    <div class="site-info-item">
                        <div class="site-info-label">Movement Rate</div>
                        <div class="site-info-value">${site.movement_rate} m/month</div>
                    </div>
                    <div class="site-info-item">
                        <div class="site-info-label">Location</div>
                        <div class="site-info-value">${site.location.lat}°, ${site.location.lon}°</div>
                    </div>
                    <div class="site-info-item">
                        <div class="site-info-label">Satellite Coverage</div>
                        <div class="site-info-value">${site.satellite_coverage}</div>
                    </div>
                </div>
            `;

            siteItem.addEventListener('click', () => {
                this.selectSite(site);
            });

            siteList.appendChild(siteItem);
        });
    }

    renderAlertsList() {
        const alertsList = document.getElementById('alertsList');
        alertsList.innerHTML = '';

        this.data.alerts.forEach(alert => {
            const alertItem = document.createElement('div');
            alertItem.className = `alert-item ${alert.severity.toLowerCase()}`;
            
            const icon = alert.severity === 'HIGH' ? '🚨' : '⚠️';
            const timestamp = new Date(alert.timestamp).toLocaleString();
            
            alertItem.innerHTML = `
                <div class="alert-icon">${icon}</div>
                <div class="alert-content">
                    <div class="alert-message">${alert.message}</div>
                    <div class="alert-timestamp">${timestamp}</div>
                </div>
            `;

            alertsList.appendChild(alertItem);
        });
    }

    renderPredictions() {
        this.renderPredictionCards();
        this.renderPredictionChart();
    }

    renderPredictionCards() {
        const predictionCards = document.getElementById('predictionCards');
        predictionCards.innerHTML = '';

        this.data.predictions.forEach(prediction => {
            const card = document.createElement('div');
            card.className = 'prediction-card';
            
            const movement = Math.sqrt(
                prediction.prediction.x_movement ** 2 + 
                prediction.prediction.y_movement ** 2
            ).toFixed(1);

            card.innerHTML = `
                <div class="prediction-model">${prediction.model}</div>
                <div class="prediction-value">${movement}m</div>
                <div class="prediction-accuracy">Accuracy: ${(prediction.accuracy * 100).toFixed(1)}%</div>
                <div class="prediction-confidence">Confidence: ${(prediction.confidence * 100).toFixed(1)}%</div>
            `;

            predictionCards.appendChild(card);
        });
    }

    renderPredictionChart() {
        const ctx = document.getElementById('predictionChart');
        if (!ctx) return;

        if (this.charts.prediction) {
            this.charts.prediction.destroy();
        }

        this.charts.prediction = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: ['X Movement', 'Y Movement', 'Accuracy', 'Confidence'],
                datasets: this.data.predictions.map((prediction, index) => ({
                    label: prediction.model,
                    data: [
                        Math.abs(prediction.prediction.x_movement),
                        Math.abs(prediction.prediction.y_movement),
                        prediction.accuracy * 10,
                        prediction.confidence * 10
                    ],
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'][index] + '40',
                    borderColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5'][index],
                    borderWidth: 2
                }))
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    r: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' },
                        angleLines: { color: '#626c7c' }
                    }
                }
            }
        });
    }

    renderRiskAssessment() {
        this.renderInfrastructureList();
        this.renderRiskTimelineChart();
    }

    renderInfrastructureList() {
        const infrastructureList = document.getElementById('infrastructureList');
        infrastructureList.innerHTML = '';

        this.data.infrastructure.forEach(infra => {
            const item = document.createElement('div');
            item.className = 'infrastructure-item';
            
            const riskLevel = infra.risk_distance < 50 ? 'HIGH' : 
                             infra.risk_distance < 100 ? 'MEDIUM' : 'LOW';
            
            item.innerHTML = `
                <div class="infrastructure-info">
                    <div class="infrastructure-name">${infra.name}</div>
                    <div class="infrastructure-type">${infra.type.toUpperCase()}</div>
                </div>
                <div class="risk-distance">
                    <div class="risk-distance-value">${infra.risk_distance}m</div>
                    <div class="risk-distance-label">Distance</div>
                </div>
                <span class="status status--${riskLevel.toLowerCase()}">${riskLevel}</span>
            `;

            infrastructureList.appendChild(item);
        });
    }

    renderRiskTimelineChart() {
        const ctx = document.getElementById('riskTimelineChart');
        if (!ctx) return;

        if (this.charts.riskTimeline) {
            this.charts.riskTimeline.destroy();
        }

        const dates = [];
        const riskLevels = [];
        
        for (let i = 0; i < 30; i++) {
            const date = new Date();
            date.setDate(date.getDate() + i);
            dates.push(date.toLocaleDateString());
            
            // Simulate increasing risk over time
            riskLevels.push(Math.min(100, 20 + (i * 2) + Math.random() * 10));
        }

        this.charts.riskTimeline = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Risk Level',
                    data: riskLevels,
                    borderColor: '#ff5459',
                    backgroundColor: '#ff545940',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' }
                    },
                    y: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    renderAnalysis() {
        this.renderHistoricalChart();
        this.renderPerformanceChart();
        this.renderTrendStats();
    }

    renderHistoricalChart() {
        const ctx = document.getElementById('historicalChart');
        if (!ctx) return;

        if (this.charts.historical) {
            this.charts.historical.destroy();
        }

        this.charts.historical = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.data.historical_data.map(d => d.date),
                datasets: [
                    {
                        label: 'Movement (m)',
                        data: this.data.historical_data.map(d => d.movement),
                        borderColor: '#1FB8CD',
                        backgroundColor: '#1FB8CD40',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Accuracy',
                        data: this.data.historical_data.map(d => d.accuracy * 10),
                        borderColor: '#FFC185',
                        backgroundColor: '#FFC18540',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        ticks: { color: '#f5f5f5' },
                        grid: { drawOnChartArea: false }
                    }
                }
            }
        });
    }

    renderPerformanceChart() {
        const ctx = document.getElementById('performanceChart');
        if (!ctx) return;

        if (this.charts.performance) {
            this.charts.performance.destroy();
        }

        this.charts.performance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: this.data.predictions.map(p => p.model),
                datasets: [{
                    label: 'Accuracy (%)',
                    data: this.data.predictions.map(p => p.accuracy * 100),
                    backgroundColor: ['#1FB8CD', '#FFC185', '#B4413C', '#ECEBD5']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#f5f5f5' }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' }
                    },
                    y: {
                        ticks: { color: '#f5f5f5' },
                        grid: { color: '#626c7c' },
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }

    renderTrendStats() {
        const trendStats = document.getElementById('trendStats');
        trendStats.innerHTML = '';

        const avgMovement = this.data.historical_data.reduce((sum, d) => sum + d.movement, 0) / this.data.historical_data.length;
        const avgAccuracy = this.data.historical_data.reduce((sum, d) => sum + d.accuracy, 0) / this.data.historical_data.length;
        const trend = this.data.historical_data[this.data.historical_data.length - 1].movement > this.data.historical_data[0].movement ? 'Increasing' : 'Decreasing';

        const stats = [
            { label: 'Average Movement', value: `${avgMovement.toFixed(1)}m` },
            { label: 'Average Accuracy', value: `${(avgAccuracy * 100).toFixed(1)}%` },
            { label: 'Movement Trend', value: trend },
            { label: 'Data Points', value: this.data.historical_data.length }
        ];

        stats.forEach(stat => {
            const statDiv = document.createElement('div');
            statDiv.className = 'trend-stat';
            statDiv.innerHTML = `
                <div class="trend-stat-value">${stat.value}</div>
                <div class="trend-stat-label">${stat.label}</div>
            `;
            trendStats.appendChild(statDiv);
        });
    }

    renderConfiguration() {
        // Load current configuration values
        document.getElementById('movementThreshold').value = this.data.config.movement_threshold;
        document.getElementById('highRiskDistance').value = this.data.config.risk_zones.high;
        document.getElementById('mediumRiskDistance').value = this.data.config.risk_zones.medium;
        document.getElementById('lowRiskDistance').value = this.data.config.risk_zones.low;
    }

    selectSite(site) {
        this.selectedSite = site;
        
        // Re-render site list to show selection
        this.renderSiteList();
        
        // Show detailed site information notification
        this.showNotification(`Selected site: ${site.name} - Movement: ${site.movement_rate}m/month, Risk: ${site.risk_level}`, 'info');
    }

    runPrediction() {
        const modelSelect = document.getElementById('modelSelect');
        const timeHorizon = document.getElementById('timeHorizon');
        
        const selectedModel = modelSelect.value;
        const horizon = parseInt(timeHorizon.value);
        
        // Simulate prediction run with loading state
        const button = document.getElementById('runPrediction');
        button.textContent = 'Running...';
        button.disabled = true;
        
        setTimeout(() => {
            // Reset button
            button.textContent = 'Run Prediction';
            button.disabled = false;
            
            // Update prediction results
            this.renderPredictionCards();
            this.renderPredictionChart();
            
            // Show success message
            this.showNotification(`Prediction completed using ${selectedModel} model for ${horizon} days`, 'success');
        }, 2000);
    }

    saveConfiguration() {
        // Get updated values
        this.data.config.movement_threshold = parseFloat(document.getElementById('movementThreshold').value);
        this.data.config.risk_zones.high = parseInt(document.getElementById('highRiskDistance').value);
        this.data.config.risk_zones.medium = parseInt(document.getElementById('mediumRiskDistance').value);
        this.data.config.risk_zones.low = parseInt(document.getElementById('lowRiskDistance').value);
        
        this.showNotification('Configuration saved successfully', 'success');
    }

    exportReport() {
        // Simulate report export
        this.showNotification('Report export started. Download will begin shortly.', 'info');
        
        setTimeout(() => {
            // Create and download a simple report file
            const reportData = {
                timestamp: new Date().toISOString(),
                sites: this.data.monitoring_sites.length,
                alerts: this.data.alerts.length,
                summary: 'Sand Dune Monitoring Report Generated'
            };
            
            const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'sand-dune-report.json';
            a.click();
            URL.revokeObjectURL(url);
        }, 1000);
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification--${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--color-surface);
            border: 1px solid var(--color-border);
            border-radius: var(--radius-base);
            padding: var(--space-16);
            z-index: 1000;
            max-width: 300px;
            box-shadow: var(--shadow-lg);
            color: var(--color-text);
        `;
        
        document.body.appendChild(notification);
        
        // Remove notification after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new SandDuneTracker();
});