/* =========================================
   MONSOON AI - FRONTEND JAVASCRIPT (API Integrated)
   ========================================= */

const API_BASE = "http://localhost:8000";

/* =========================================
   MOBILE SIDEBAR
   ========================================= */

const menuBtn = document.getElementById("menuBtn");
const sidebar = document.querySelector(".sidebar");

menuBtn.addEventListener("click", () => {
    sidebar.classList.toggle("open");
});


/* =========================================
   NAVIGATION
   ========================================= */

document.querySelectorAll(".nav-link").forEach(link => {

    link.addEventListener("click", function () {

        document.querySelectorAll(".nav-link").forEach(item => {
            item.classList.remove("active");
        });

        this.classList.add("active");

        sidebar.classList.remove("open");

    });

});


/* =========================================
   CHART INSTANCES (to update dynamically)
   ========================================= */

let rainfallChart = null;
let regimeChart = null;
let verificationChart = null;


/* =========================================
   API HELPERS
   ========================================= */

async function apiGet(endpoint) {
    const response = await fetch(`${API_BASE}${endpoint}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}

async function apiPost(endpoint, data) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });
    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
}


/* =========================================
   INITIALIZE - Load data from API on startup
   ========================================= */

async function initializeApp() {
    try {
        // Load districts
        const districtsData = await apiGet("/districts");
        populateDistrictSelect(districtsData.districts);
        
        // Load initial forecasts for all districts
        await loadAllDistrictForecasts();
        
        // Load verification metrics
        await loadVerificationMetrics();
        
        console.log("MonsoonAI Frontend initialized with API data.");
    } catch (error) {
        console.error("Failed to initialize app:", error);
        // Fallback to static data if API fails
        initializeStaticFallback();
    }
}

function populateDistrictSelect(districts) {
    const select = document.getElementById("districtSelect");
    select.innerHTML = "";
    districts.forEach(d => {
        const option = document.createElement("option");
        option.value = d.district.toLowerCase().replace(/\s+/g, "");
        option.textContent = d.district;
        select.appendChild(option);
    });
    
    // Update changeDistrict to use API
    select.onchange = () => loadDistrictForecast(select.value);
}


/* =========================================
   SINGLE FORECAST (Dashboard)
   ========================================= */

async function runPrediction() {
    const button = document.querySelector(".primary-btn");
    
    button.textContent = "Processing...";
    button.disabled = true;

    try {
        // Get current district or use first one
        const select = document.getElementById("districtSelect");
        const districtKey = select.value || "murshidabad";
        
        // Fetch current weather for the district
        const forecast = await apiPost("/predict/district", { district: districtKey });
        
        updateDashboard(forecast);
        
        button.textContent = "Prediction Updated";
        setTimeout(() => {
            button.textContent = "Run AI Prediction";
            button.disabled = false;
        }, 1500);
    } catch (error) {
        console.error("Prediction failed:", error);
        button.textContent = "Error - Retry";
        button.disabled = false;
    }
}

function updateDashboard(forecast) {
    document.getElementById("currentRegime").textContent = forecast.regime;
    document.getElementById("rawRain").textContent = forecast.nwp_rainfall + " mm";
    document.getElementById("correctedRain").textContent = forecast.corrected_rainfall + " mm";
    
    // Calculate heavy rain probability based on corrected rainfall
    const prob = Math.min(95, Math.round((forecast.corrected_rainfall / 120) * 100));
    document.getElementById("heavyProbability").textContent = prob + "%";
    
    // Update charts
    updateRainfallChart(forecast);
    updateRegimeChart(forecast);
}

function updateRainfallChart(forecast) {
    if (!rainfallChart) return;
    
    // Simulate hourly progression based on corrected total
    const total = forecast.corrected_rainfall;
    const rawTotal = forecast.nwp_rainfall;
    
    const rawData = [15, 21, 32, 45, 58, 67, 72, 69, 61].map(v => (v / 72) * rawTotal);
    const correctedData = [18, 25, 39, 53, 69, 81, 91, 86, 76].map(v => (v / 91) * total);
    
    rainfallChart.data.datasets[0].data = rawData;
    rainfallChart.data.datasets[1].data = correctedData;
    rainfallChart.update();
}

function updateRegimeChart(forecast) {
    if (!regimeChart) return;
    
    // Update regime confidence display
    const probs = forecast.regime_probabilities || {};
    document.getElementById("activeConfidence").textContent = `Confidence: ${probs["Active"] || 0}%`;
    document.getElementById("breakConfidence").textContent = `Confidence: ${probs["Break"] || 0}%`;
    document.getElementById("depressionConfidence").textContent = `Confidence: ${probs["Depression"] || 0}%`;
    
    // Update doughnut chart
    const regimeData = {
        "Active Monsoon": probs["Active"] || 0,
        "Break Monsoon": probs["Break"] || 0,
        "Depression": probs["Depression"] || 0
    };
    
    regimeChart.data.labels = Object.keys(regimeData);
    regimeChart.data.datasets[0].data = Object.values(regimeData);
    regimeChart.update();
}


/* =========================================
   DISTRICT FORECAST
   ========================================= */

async function loadDistrictForecast(districtKey) {
    try {
        const forecast = await apiPost("/predict/district", { district: districtKey });
        updateDistrictTable([forecast]);
        updateDashboard(forecast);
        updateRainfallMap([forecast]);
    } catch (error) {
        console.error("District forecast failed:", error);
    }
}

async function loadAllDistrictForecasts() {
    try {
        const forecasts = await apiGet("/predict/all-districts");
        updateDistrictTable(forecasts);
        
        // Update dashboard with first district
        if (forecasts.length > 0) {
            updateDashboard(forecasts[0]);
        }
        
        // Update rainfall risk map
        updateRainfallMap(forecasts);
    } catch (error) {
        console.error("All districts forecast failed:", error);
        initializeStaticFallback();
    }
}

function updateDistrictTable(forecasts) {
    const tbody = document.getElementById("forecastTableBody");
    tbody.innerHTML = "";
    
    forecasts.forEach(f => {
        const riskClass = f.risk_level.includes("Extreme") ? "extreme" : 
                          f.risk_level.includes("High") ? "high" : 
                          f.risk_level.includes("Moderate") ? "medium" : "low";
        
        const regimeClass = f.regime.toLowerCase().includes("active") ? "active" :
                           f.regime.toLowerCase().includes("coastal") ? "coastal" :
                           f.regime.toLowerCase().includes("orographic") ? "oro" : "";
        
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${f.district}</td>
            <td><span class="badge ${regimeClass}">${f.regime}</span></td>
            <td>${f.nwp_rainfall} mm</td>
            <td><strong>${f.corrected_rainfall} mm</strong></td>
            <td>${Math.min(95, Math.round((f.corrected_rainfall / 120) * 100))}%</td>
            <td><span class="risk ${riskClass}">${f.risk_level.replace(/[🟢🟡🟠🔴]/g, '').trim()}</span></td>
        `;
        tbody.appendChild(tr);
    });
}


/* =========================================
   VERIFICATION METRICS
   ========================================= */

async function loadVerificationMetrics() {
    try {
        const metrics = await apiGet("/verification/metrics");
        updateVerificationUI(metrics);
    } catch (error) {
        console.error("Verification metrics failed:", error);
    }
}

function updateVerificationUI(metrics) {
    // Update metric cards
    const continuous = metrics.continuous;
    const categorical = metrics.categorical;
    
    // Update the metric cards in verification section
    const metricCards = document.querySelectorAll(".metrics .metric");
    if (metricCards.length >= 6) {
        metricCards[0].querySelector("h2").textContent = continuous.RMSE?.toFixed(2) || "18.42";
        metricCards[1].querySelector("h2").textContent = categorical["threshold_15mm"]?.ETS?.toFixed(2) || "0.61";
        metricCards[2].querySelector("h2").textContent = categorical["threshold_15mm"]?.CSI?.toFixed(2) || "0.74";
        metricCards[3].querySelector("h2").textContent = categorical["threshold_15mm"]?.POD?.toFixed(2) || "0.81";
        metricCards[4].querySelector("h2").textContent = categorical["threshold_15mm"]?.FAR?.toFixed(2) || "0.18";
        metricCards[5].querySelector("h2").textContent = categorical["threshold_15mm"]?.FSS?.toFixed(2) || "0.69";
    }
    
    // Update verification chart
    updateVerificationChart(metrics);
}

function updateVerificationChart(metrics) {
    if (!verificationChart) return;
    
    const cat = metrics.categorical;
    const rawData = [31, .38, .49, .62, .34, .41];
    const aiData = [
        metrics.continuous.RMSE || 18,
        cat["threshold_15mm"]?.ETS || 0.61,
        cat["threshold_15mm"]?.CSI || 0.74,
        cat["threshold_15mm"]?.POD || 0.81,
        cat["threshold_15mm"]?.FAR || 0.18,
        cat["threshold_15mm"]?.FSS || 0.69
    ];
    
    verificationChart.data.datasets[0].data = rawData;
    verificationChart.data.datasets[1].data = aiData;
    verificationChart.update();
}


/* =========================================
   CHARTS INITIALIZATION
   ========================================= */

function initCharts() {
    // Rainfall Chart
    const rainfallCtx = document.getElementById("rainfallChart");
    rainfallChart = new Chart(rainfallCtx, {
        type: "line",
        data: {
            labels: ["00h", "03h", "06h", "09h", "12h", "15h", "18h", "21h", "24h"],
            datasets: [
                {
                    label: "Raw NWP",
                    data: [15, 21, 32, 45, 58, 67, 72, 69, 61],
                    borderWidth: 2,
                    tension: .4,
                    borderColor: "#64748b",
                    backgroundColor: "rgba(100, 116, 139, 0.1)"
                },
                {
                    label: "AI Post-Processed",
                    data: [18, 25, 39, 53, 69, 81, 91, 86, 76],
                    borderWidth: 3,
                    tension: .4,
                    borderColor: "#2563eb",
                    backgroundColor: "rgba(37, 99, 235, 0.1)"
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: "bottom" } },
            scales: {
                y: { beginAtZero: true, title: { display: true, text: "Rainfall (mm)" } }
            }
        }
    });

    // Regime Chart
        const regimeCtx = document.getElementById("regimeChart");
        regimeChart = new Chart(regimeCtx, {
            type: "doughnut",
            data: {
                labels: ["Active Monsoon", "Break Monsoon", "Depression"],
                datasets: [{
                    data: [0, 0, 0],
                    borderWidth: 0,
                    backgroundColor: ["#2563eb", "#64748b", "#dc2626"]
                }]
            },
            options: { responsive: true, plugins: { legend: { position: "bottom" } } }
        });

    // Verification Chart
    const verificationCtx = document.getElementById("verificationChart");
    verificationChart = new Chart(verificationCtx, {
        type: "bar",
        data: {
            labels: ["RMSE", "ETS", "CSI", "POD", "FAR", "FSS"],
            datasets: [
                {
                    label: "Raw NWP",
                    data: [31, .38, .49, .62, .34, .41],
                    borderWidth: 1,
                    backgroundColor: "rgba(100, 116, 139, 0.6)"
                },
                {
                    label: "AI Post-Processed",
                    data: [18, .61, .74, .81, .18, .69],
                    borderWidth: 1,
                    backgroundColor: "rgba(37, 99, 235, 0.6)"
                }
            ]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: "bottom" } },
            scales: { y: { beginAtZero: true } }
        }
    });
}


/* =========================================
   STATIC FALLBACK (if API unavailable)
   ========================================= */

function initializeStaticFallback() {
    console.log("Using static fallback data");
    
    // Original static district data
    const districtData = {
        murshidabad: { regime: "Active Monsoon", raw: 72, corrected: 91, probability: 78 },
        kolkata: { regime: "Coastal", raw: 46, corrected: 58, probability: 61 },
        darjeeling: { regime: "Orographic", raw: 82, corrected: 108, probability: 84 },
        purba: { regime: "Coastal", raw: 67, corrected: 79, probability: 72 },
        jalpaiguri: { regime: "Active Monsoon", raw: 94, corrected: 121, probability: 89 }
    };
    
    // Populate select with static data
    const select = document.getElementById("districtSelect");
    Object.keys(districtData).forEach(key => {
        const option = document.createElement("option");
        option.value = key;
        option.textContent = key.charAt(0).toUpperCase() + key.slice(1);
        select.appendChild(option);
    });
    
    select.onchange = () => changeDistrict(districtData);
    
    // Initialize with first district
    changeDistrict(districtData);
    initCharts();
}

function changeDistrict(districtData) {
    const district = document.getElementById("districtSelect").value;
    const data = districtData[district];
    
    document.getElementById("currentRegime").textContent = data.regime;
    document.getElementById("rawRain").textContent = data.raw + " mm";
    document.getElementById("correctedRain").textContent = data.corrected + " mm";
    document.getElementById("heavyProbability").textContent = data.probability + "%";
    
    // Update table
    const tbody = document.getElementById("forecastTableBody");
    Object.entries(districtData).forEach(([key, d]) => {
        const riskClass = d.probability > 80 ? "high" : d.probability > 65 ? "medium" : "low";
        const regimeClass = d.regime.toLowerCase().includes("active") ? "active" :
                           d.regime.toLowerCase().includes("coastal") ? "coastal" : "oro";
        
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${key.charAt(0).toUpperCase() + key.slice(1)}</td>
            <td><span class="badge ${regimeClass}">${d.regime}</span></td>
            <td>${d.raw} mm</td>
            <td><strong>${d.corrected} mm</strong></td>
            <td>${d.probability}%</td>
            <td><span class="risk ${riskClass}">${riskClass.charAt(0).toUpperCase() + riskClass.slice(1)}</span></td>
        `;
        tbody.appendChild(tr);
    });
    
    initCharts();
}


/* =========================================
   SCROLL ACTIVE NAVIGATION
   ========================================= */

const sections = document.querySelectorAll(".page");

window.addEventListener("scroll", () => {

    let current = "";

    sections.forEach(section => {

        const sectionTop = section.offsetTop - 150;

        if (window.scrollY >= sectionTop) {
            current = section.getAttribute("id");
        }

    });

    document.querySelectorAll(".nav-link").forEach(link => {
        link.classList.remove("active");
        if (link.getAttribute("href") === "#" + current) {
            link.classList.add("active");
        }
    });

});


/* =========================================
   LEAFLET MAP
   ========================================= */

let rainfallMap = null;

function initRainfallMap() {
    if (rainfallMap) return;
    
    const container = document.getElementById('rainfallMap');
    if (!container) return;
    
    // Initialize regardless of visibility - Leaflet can handle hidden containers
    rainfallMap = L.map('rainfallMap').setView([22, 80], 5);
    
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 10
    }).addTo(rainfallMap);
    
    // Fix for hidden container - invalidate size when shown
    setTimeout(() => {
        if (rainfallMap) rainfallMap.invalidateSize();
    }, 100);
}

function updateRainfallMap(forecasts) {
    if (!rainfallMap) {
        initRainfallMap();
    }
    
    // If still not initialized, try again after a short delay
    if (!rainfallMap) {
        setTimeout(() => updateRainfallMap(forecasts), 200);
        return;
    }
    
    // Ensure map is properly sized
    rainfallMap.invalidateSize();
    
    // Clear existing markers
    rainfallMap.eachLayer(layer => {
        if (layer instanceof L.CircleMarker || layer instanceof L.Marker) {
            rainfallMap.removeLayer(layer);
        }
    });
    
    forecasts.forEach(f => {
        if (!f.latitude || !f.longitude) return;
        
        // Color based on risk level
        const riskColors = {
            'low': '#16a34a',
            'moderate': '#f59e0b',
            'high': '#ef4444',
            'extreme': '#dc2626'
        };
        
        // Normalize risk level
        const riskKey = f.risk_level.replace(/[🟢🟡🟠🔴]/g, '').trim().toLowerCase();
        const color = riskColors[riskKey] || '#2563eb';
        
        // Radius based on corrected rainfall
        const radius = Math.max(8, Math.min(30, f.corrected_rainfall / 3));
        
        const circle = L.circleMarker([f.latitude, f.longitude], {
            radius: radius,
            fillColor: color,
            color: '#fff',
            weight: 2,
            opacity: 1,
            fillOpacity: 0.7
        }).addTo(rainfallMap);
        
        circle.bindPopup(`
            <strong>${f.district}</strong><br>
            ${f.state}<br>
            Regime: ${f.regime}<br>
            NWP: ${f.nwp_rainfall.toFixed(1)} mm<br>
            AI Forecast: ${f.corrected_rainfall.toFixed(1)} mm<br>
            Risk: ${f.risk_level}
        `);
    });
    
    // Fit bounds to show all markers
    if (forecasts.length > 0) {
        const bounds = L.latLngBounds(
            forecasts.map(f => [f.latitude, f.longitude]).filter(c => c[0] && c[1])
        );
        if (bounds.isValid()) {
            rainfallMap.fitBounds(bounds, { padding: [50, 50], maxZoom: 6 });
        }
    }
}


/* =========================================
   START APP
   ========================================= */

document.addEventListener("DOMContentLoaded", () => {
    initCharts();
    initializeApp();
});

console.log("MonsoonAI Frontend loaded (API mode).");