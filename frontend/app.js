/**
 * RivIntel Matrix - Modern JavaScript SPA
 * Advanced Environmental Intelligence Dashboard
 * Author: Nikitha Kunapareddy
 */

class RivIntelApp {
    constructor() {
        this.apiBase = '/api/v1';
        this.currentView = 'dashboard';
        this.realTimeData = {};
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDashboard();
        this.startRealTimeUpdates();
        this.initializeCharts();
        this.setupWebSocket();
    }

    setupEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.renderNavigation();
            this.renderMainContent();
        });

        // Navigation event handlers
        window.addEventListener('hashchange', () => {
            this.handleRouteChange();
        });

        // Real-time data handlers
        window.addEventListener('resize', () => {
            this.resizeCharts();
        });
    }

    renderNavigation() {
        const nav = document.createElement('nav');
        nav.className = 'navbar';
        nav.innerHTML = `
            <div class="nav-container">
                <div class="logo">
                    <h2>🌊 RivIntel Matrix</h2>
                </div>
                <ul class="nav-menu">
                    <li><a href="#dashboard" class="nav-link">Dashboard</a></li>
                    <li><a href="#monitoring" class="nav-link">Real-time Monitor</a></li>
                    <li><a href="#analytics" class="nav-link">AI Analytics</a></li>
                    <li><a href="#predictions" class="nav-link">Predictions</a></li>
                    <li><a href="#alerts" class="nav-link">Alerts</a></li>
                    <li><a href="#reports" class="nav-link">Reports</a></li>
                </ul>
                <div class="status-indicator">
                    <span class="status-dot active"></span>
                    <span>System Online</span>
                </div>
            </div>
        `;
        document.body.appendChild(nav);
    }

    renderMainContent() {
        const main = document.createElement('main');
        main.id = 'app-container';
        main.className = 'app-container';
        document.body.appendChild(main);
        this.loadDashboard();
    }

    async loadDashboard() {
        const container = document.getElementById('app-container');
        container.innerHTML = `
            <div class="dashboard">
                <header class="dashboard-header">
                    <h1>Environmental Intelligence Dashboard</h1>
                    <div class="header-stats">
                        <div class="stat-card">
                            <h3 id="water-quality-score">--</h3>
                            <p>Water Quality Score</p>
                        </div>
                        <div class="stat-card">
                            <h3 id="active-sensors">--</h3>
                            <p>Active Sensors</p>
                        </div>
                        <div class="stat-card">
                            <h3 id="predictions-made">--</h3>
                            <p>AI Predictions Today</p>
                        </div>
                    </div>
                </header>

                <div class="dashboard-grid">
                    <div class="chart-container">
                        <h3>Real-time Water Quality</h3>
                        <canvas id="water-quality-chart"></canvas>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Environmental Predictions</h3>
                        <canvas id="predictions-chart"></canvas>
                    </div>
                    
                    <div class="chart-container">
                        <h3>Ecosystem Health</h3>
                        <canvas id="ecosystem-chart"></canvas>
                    </div>
                    
                    <div class="alert-panel">
                        <h3>Active Alerts</h3>
                        <div id="alerts-list"></div>
                    </div>
                    
                    <div class="ml-insights">
                        <h3>AI Insights</h3>
                        <div id="ml-insights-content"></div>
                    </div>
                    
                    <div class="real-time-data">
                        <h3>Live Sensor Data</h3>
                        <div id="sensor-data-grid"></div>
                    </div>
                </div>
            </div>
        `;
        
        await this.loadDashboardData();
    }

    async loadDashboardData() {
        try {
            // Fetch real-time data from Python backend
            const [waterData, predictions, ecosystem, alerts] = await Promise.all([
                this.fetchAPI('/water-quality/real-time'),
                this.fetchAPI('/predictions/current'),
                this.fetchAPI('/ecosystem/health'),
                this.fetchAPI('/alerts/active')
            ]);

            this.updateDashboardStats(waterData, predictions, ecosystem);
            this.renderCharts(waterData, predictions, ecosystem);
            this.renderAlerts(alerts);
            this.renderMLInsights(predictions);
            this.renderSensorData(waterData);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showErrorMessage('Failed to load dashboard data');
        }
    }

    async fetchAPI(endpoint) {
        const response = await fetch(`${this.apiBase}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API call failed: ${response.statusText}`);
        }
        return await response.json();
    }

    updateDashboardStats(waterData, predictions, ecosystem) {
        document.getElementById('water-quality-score').textContent = 
            waterData.overall_score?.toFixed(1) || '--';
        document.getElementById('active-sensors').textContent = 
            waterData.active_sensors || '--';
        document.getElementById('predictions-made').textContent = 
            predictions.daily_count || '--';
    }

    renderCharts(waterData, predictions, ecosystem) {
        this.renderWaterQualityChart(waterData);
        this.renderPredictionsChart(predictions);
        this.renderEcosystemChart(ecosystem);
    }

    renderWaterQualityChart(data) {
        const ctx = document.getElementById('water-quality-chart');
        if (this.charts.waterQuality) {
            this.charts.waterQuality.destroy();
        }
        
        this.charts.waterQuality = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.timestamps || [],
                datasets: [{
                    label: 'pH Level',
                    data: data.ph_levels || [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true
                }, {
                    label: 'Dissolved Oxygen',
                    data: data.dissolved_oxygen || [],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    fill: true
                }, {
                    label: 'Turbidity',
                    data: data.turbidity || [],
                    borderColor: '#e74c3c',
                    backgroundColor: 'rgba(231, 76, 60, 0.1)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    renderPredictionsChart(data) {
        const ctx = document.getElementById('predictions-chart');
        if (this.charts.predictions) {
            this.charts.predictions.destroy();
        }
        
        this.charts.predictions = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.prediction_types || [],
                datasets: [{
                    label: 'Prediction Accuracy (%)',
                    data: data.accuracy_scores || [],
                    backgroundColor: [
                        '#3498db', '#2ecc71', '#f39c12', 
                        '#e74c3c', '#9b59b6', '#1abc9c'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    renderEcosystemChart(data) {
        const ctx = document.getElementById('ecosystem-chart');
        if (this.charts.ecosystem) {
            this.charts.ecosystem.destroy();
        }
        
        this.charts.ecosystem = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: data.health_metrics || [],
                datasets: [{
                    label: 'Current Health',
                    data: data.current_scores || [],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.2)',
                    pointBackgroundColor: '#2ecc71'
                }, {
                    label: 'Predicted Health',
                    data: data.predicted_scores || [],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.2)',
                    pointBackgroundColor: '#3498db'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: true,
                        max: 100
                    }
                }
            }
        });
    }

    renderAlerts(alerts) {
        const alertsList = document.getElementById('alerts-list');
        if (!alerts || alerts.length === 0) {
            alertsList.innerHTML = '<p>No active alerts</p>';
            return;
        }

        alertsList.innerHTML = alerts.map(alert => `
            <div class="alert-item ${alert.severity}">
                <div class="alert-icon">${this.getAlertIcon(alert.severity)}</div>
                <div class="alert-content">
                    <h4>${alert.title}</h4>
                    <p>${alert.description}</p>
                    <small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
                <button class="alert-dismiss" onclick="app.dismissAlert('${alert.id}')">×</button>
            </div>
        `).join('');
    }

    renderMLInsights(predictions) {
        const insights = document.getElementById('ml-insights-content');
        insights.innerHTML = `
            <div class="insight-grid">
                <div class="insight-card">
                    <h4>🧠 Neural Network Status</h4>
                    <p>LSTM model accuracy: ${predictions.lstm_accuracy || 'N/A'}%</p>
                </div>
                <div class="insight-card">
                    <h4>🔮 Next Prediction</h4>
                    <p>${predictions.next_prediction || 'Calculating...'}</p>
                </div>
                <div class="insight-card">
                    <h4>⚡ Model Performance</h4>
                    <p>Processing ${predictions.data_points_per_second || 0} points/sec</p>
                </div>
            </div>
        `;
    }

    renderSensorData(data) {
        const sensorGrid = document.getElementById('sensor-data-grid');
        const sensors = data.sensors || [];
        
        sensorGrid.innerHTML = sensors.map(sensor => `
            <div class="sensor-card ${sensor.status}">
                <h4>${sensor.name}</h4>
                <div class="sensor-value">${sensor.value} ${sensor.unit}</div>
                <div class="sensor-status">${sensor.status}</div>
                <div class="sensor-time">${new Date(sensor.last_update).toLocaleTimeString()}</div>
            </div>
        `).join('');
    }

    setupWebSocket() {
        this.ws = new WebSocket('ws://localhost:8000/ws');
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleRealTimeUpdate(data);
        };

        this.ws.onclose = () => {
            console.log('WebSocket connection closed. Attempting to reconnect...');
            setTimeout(() => this.setupWebSocket(), 5000);
        };
    }

    handleRealTimeUpdate(data) {
        if (data.type === 'sensor_update') {
            this.updateSensorData(data.payload);
        } else if (data.type === 'alert') {
            this.addNewAlert(data.payload);
        } else if (data.type === 'prediction') {
            this.updatePredictions(data.payload);
        }
    }

    updateSensorData(sensorData) {
        // Update charts with new data
        if (this.charts.waterQuality && sensorData.water_quality) {
            const chart = this.charts.waterQuality;
            chart.data.labels.push(new Date().toLocaleTimeString());
            chart.data.datasets.forEach((dataset, index) => {
                dataset.data.push(sensorData.water_quality[index]);
            });
            
            // Keep only last 20 data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
            
            chart.update('none');
        }
    }

    addNewAlert(alert) {
        const alertsList = document.getElementById('alerts-list');
        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${alert.severity} new-alert`;
        alertElement.innerHTML = `
            <div class="alert-icon">${this.getAlertIcon(alert.severity)}</div>
            <div class="alert-content">
                <h4>${alert.title}</h4>
                <p>${alert.description}</p>
                <small>${new Date().toLocaleString()}</small>
            </div>
            <button class="alert-dismiss" onclick="app.dismissAlert('${alert.id}')">×</button>
        `;
        
        alertsList.insertBefore(alertElement, alertsList.firstChild);
        
        // Remove the 'new-alert' class after animation
        setTimeout(() => {
            alertElement.classList.remove('new-alert');
        }, 1000);
    }

    getAlertIcon(severity) {
        const icons = {
            'low': '💡',
            'medium': '⚠️', 
            'high': '🚨',
            'critical': '🔴'
        };
        return icons[severity] || '📢';
    }

    async dismissAlert(alertId) {
        try {
            await fetch(`${this.apiBase}/alerts/${alertId}/dismiss`, {
                method: 'POST'
            });
            
            const alertElement = document.querySelector(`[onclick="app.dismissAlert('${alertId}')"]`).parentElement;
            alertElement.style.opacity = '0';
            setTimeout(() => alertElement.remove(), 300);
        } catch (error) {
            console.error('Error dismissing alert:', error);
        }
    }

    startRealTimeUpdates() {
        setInterval(() => {
            if (this.currentView === 'dashboard') {
                this.loadDashboardData();
            }
        }, 30000); // Update every 30 seconds
    }

    initializeCharts() {
        // Initialize Chart.js defaults
        Chart.defaults.font.family = "'Poppins', sans-serif";
        Chart.defaults.color = '#666';
        Chart.defaults.borderColor = 'rgba(0,0,0,0.1)';
    }

    resizeCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.resize();
        });
    }

    handleRouteChange() {
        const hash = window.location.hash.substring(1) || 'dashboard';
        this.currentView = hash;
        
        switch (hash) {
            case 'dashboard':
                this.loadDashboard();
                break;
            case 'monitoring':
                this.loadMonitoring();
                break;
            case 'analytics':
                this.loadAnalytics();
                break;
            case 'predictions':
                this.loadPredictions();
                break;
            case 'alerts':
                this.loadAlerts();
                break;
            case 'reports':
                this.loadReports();
                break;
            default:
                this.loadDashboard();
        }
    }

    showErrorMessage(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = `
            <div class="error-content">
                <h3>⚠️ Error</h3>
                <p>${message}</p>
                <button onclick="this.parentElement.parentElement.remove()">Dismiss</button>
            </div>
        `;
        document.body.appendChild(errorDiv);
        
        setTimeout(() => {
            if (errorDiv.parentElement) {
                errorDiv.remove();
            }
        }, 5000);
    }
}

// Initialize the application
const app = new RivIntelApp();

// Export for global access
window.app = app;
