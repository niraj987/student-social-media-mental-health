// --- GLOBAL STATE ---
let rawDataset = [];
let filteredDataset = [];
let edaData = null;
let modelData = null;
let currentPage = 1;
const rowsPerPage = 10;

// Colors for stress levels
const STRESS_COLORS = {
    "Low": "hsl(142, 72%, 45%)",
    "Medium": "hsl(38, 92%, 50%)",
    "High": "hsl(0, 84%, 60%)",
    "Very High": "hsl(340, 85%, 50%)"
};

// --- INITIALIZATION ---
document.addEventListener("DOMContentLoaded", async () => {
    // 1. Fetch EDA stats & Model params
    try {
        const [edaRes, modelRes] = await Promise.all([
            fetch("data_export.json"),
            fetch("trained_model.json")
        ]);
        
        edaData = await edaRes.json();
        modelData = await modelRes.json();
        
        // Populate KPI cards
        updateKPICards(edaData);
        
        // Populate Form Select Elements dynamically
        populateSelectElements(modelData.categorical_mappings);
        
        // Render Charts using Chart.js
        renderCharts(edaData);
        
        // Render Correlation Grid
        renderCorrelationGrid(edaData.correlation_matrix);
        
        // Bind prediction form events (Real-time Live Inference)
        bindSandboxEvents();
        
        // Run initial prediction
        runLiveInference();
        
    } catch (err) {
        console.error("Error loading JSON metadata:", err);
    }
    
    // 2. Fetch and parse raw CSV for Dataset Explorer
    try {
        const csvRes = await fetch("Student Social Media And Mental Health Impact.csv");
        const csvText = await csvRes.text();
        rawDataset = parseCSV(csvText);
        filteredDataset = [...rawDataset];
        
        // Populate filters and data table
        populateExplorerFilters(rawDataset);
        renderTablePage(1);
        
    } catch (err) {
        console.error("Error loading raw CSV:", err);
    }
    
    // 3. Bind navigation/action buttons
    bindActionButtons();
});

// --- HELPER FUNCTIONS ---

// CSV Parser Helper
function parseCSV(text) {
    const lines = text.split("\n");
    if (lines.length === 0) return [];
    
    // Parse headers
    const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, ""));
    const records = [];
    
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line) continue;
        
        // Custom simple CSV splitter (handles standard columns without nested commas)
        const cols = line.split(",").map(c => c.trim().replace(/^"|"$/g, ""));
        
        if (cols.length === headers.length) {
            const rowObj = {};
            headers.forEach((h, idx) => {
                rowObj[h] = cols[idx];
            });
            records.push(rowObj);
        }
    }
    return records;
}

// Populate Dropdowns dynamically based on model training mapping values
function populateSelectElements(mappings) {
    const genderSelect = document.getElementById("input-gender");
    const academicSelect = document.getElementById("input-academic");
    const platformSelect = document.getElementById("input-platform");
    const purposeSelect = document.getElementById("input-purpose");
    
    // Gender Select
    Object.keys(mappings.Gender).forEach(val => {
        genderSelect.add(new Option(val, mappings.Gender[val]));
    });
    
    // Academic Level Select
    Object.keys(mappings.Academic_Level).forEach(val => {
        academicSelect.add(new Option(val, mappings.Academic_Level[val]));
    });
    
    // Platform Select
    Object.keys(mappings.Most_Used_Platform).forEach(val => {
        platformSelect.add(new Option(val, mappings.Most_Used_Platform[val]));
    });
    
    // Purpose Select
    Object.keys(mappings.Purpose_Of_Use).forEach(val => {
        purposeSelect.add(new Option(val, mappings.Purpose_Of_Use[val]));
    });
}

// Update top dashboard stats (from analysis)
function updateKPICards(eda) {
    document.getElementById("stat-total-records").innerText = eda.total_records.toLocaleString();
    document.getElementById("stat-avg-screentime").innerText = eda.averages.screen_time.toFixed(1) + "h";
    document.getElementById("stat-avg-sleep").innerText = eda.averages.sleep.toFixed(1) + "h";
    document.getElementById("stat-risk-ratio").innerText = (eda.high_risk_ratio * 100).toFixed(1) + "%";
}

// --- RENDER VISUALIZATIONS ---
function renderCharts(eda) {
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.borderColor = "rgba(255, 255, 255, 0.05)";
    
    // 1. Platform Distribution Bar Chart
    const platformCtx = document.getElementById("platformChart").getContext("2d");
    const platformLabels = Object.keys(eda.platform_distribution);
    const platformValues = Object.values(eda.platform_distribution);
    
    new Chart(platformCtx, {
        type: "bar",
        data: {
            labels: platformLabels,
            datasets: [{
                label: "Students",
                data: platformValues,
                backgroundColor: "rgba(139, 92, 246, 0.6)",
                borderColor: "rgb(139, 92, 246)",
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: "rgba(255, 255, 255, 0.05)" } },
                x: { grid: { display: false } }
            }
        }
    });

    // 2. Stress screen time relationship chart
    const stressScreenCtx = document.getElementById("stressScreenChart").getContext("2d");
    const stressLabels = ["Low", "Medium", "High", "Very High"];
    const stressValues = stressLabels.map(lvl => eda.stress_screentime_avg[lvl] || 0);
    
    new Chart(stressScreenCtx, {
        type: "bar",
        data: {
            labels: stressLabels,
            datasets: [{
                label: "Avg Screen Hours/Day",
                data: stressValues,
                backgroundColor: [
                    "rgba(34, 197, 94, 0.5)",
                    "rgba(245, 158, 11, 0.5)",
                    "rgba(239, 68, 68, 0.5)",
                    "rgba(244, 63, 94, 0.6)"
                ],
                borderColor: [
                    "rgb(34, 197, 94)",
                    "rgb(245, 158, 11)",
                    "rgb(239, 68, 68)",
                    "rgb(244, 63, 94)"
                ],
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: "rgba(255, 255, 255, 0.05)" } },
                x: { grid: { display: false } }
            }
        }
    });

    // 3. Screen Time vs. Mental Health Score Scatter
    const scatterCtx = document.getElementById("scatterChart").getContext("2d");
    const scatterPoints = eda.screentime_vs_mentalhealth;
    
    // Sort scatter points into datasets by stress level for colored labels
    const scatterDatasets = ["Low", "Medium", "High", "Very High"].map(lvl => {
        return {
            label: lvl + " Stress",
            data: scatterPoints.filter(p => p.stress === lvl).map(p => ({ x: p.x, y: p.y })),
            backgroundColor: STRESS_COLORS[lvl],
            pointRadius: 4,
            pointHoverRadius: 6
        };
    });

    new Chart(scatterCtx, {
        type: "scatter",
        data: { datasets: scatterDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: "top", labels: { boxWidth: 10 } }
            },
            scales: {
                x: { 
                    title: { display: true, text: "Daily Social Media Hours" },
                    grid: { color: "rgba(255, 255, 255, 0.03)" }
                },
                y: { 
                    title: { display: true, text: "Mental Health Score" },
                    grid: { color: "rgba(255, 255, 255, 0.03)" }
                }
            }
        }
    });
}

// Render the interactive correlation matrix grid
function renderCorrelationGrid(matrix) {
    const gridContainer = document.getElementById("correlation-matrix-grid");
    gridContainer.innerHTML = ""; // Clear
    
    const labels = Object.keys(matrix);
    
    // 1. Create header label row
    const headerRow = document.createElement("div");
    headerRow.className = "corr-header-row";
    labels.forEach(lbl => {
        const header = document.createElement("div");
        header.className = "corr-header-label";
        // Shorten labels for display
        header.innerText = lbl.replace(" Hours", "h").replace(" Hours/Night", "h").replace(" Score", "");
        headerRow.appendChild(header);
    });
    gridContainer.appendChild(headerRow);
    
    // 2. Create row elements
    labels.forEach(lblY => {
        const row = document.createElement("div");
        row.className = "corr-row";
        
        // Row title
        const rowLabel = document.createElement("div");
        rowLabel.className = "corr-row-label";
        rowLabel.innerText = lblY;
        row.appendChild(rowLabel);
        
        // Cells container
        const cellsContainer = document.createElement("div");
        cellsContainer.className = "corr-cells";
        
        labels.forEach(lblX => {
            const coef = matrix[lblY][lblX];
            const cell = document.createElement("div");
            cell.className = "corr-cell";
            cell.innerText = coef.toFixed(2);
            
            // Set HSL color background based on positive (blue/cyan) or negative (red/orange) correlation
            let bgColor = "rgba(0,0,0,0)";
            if (coef > 0) {
                bgColor = `rgba(6, 182, 212, ${coef * 0.7})`; // Cyan scale
            } else if (coef < 0) {
                bgColor = `rgba(239, 68, 68, ${Math.abs(coef) * 0.7})`; // Red scale
            }
            
            cell.style.backgroundColor = bgColor;
            cell.setAttribute("data-tooltip", `${lblY} vs ${lblX}: ${coef.toFixed(4)}`);
            cellsContainer.appendChild(cell);
        });
        
        row.appendChild(cellsContainer);
        gridContainer.appendChild(row);
    });
}

// --- MACHINE LEARNING LIVE INFERENCE ENGINES ---

// Standard OLS Prediction
function predictLinearRegression(features, model) {
    const weights = model.coefficients;
    const intercept = model.intercept;
    const means = model.means;
    const stds = model.stds;
    
    // Standardize input features
    const scaledFeatures = [];
    for (let i = 0; i < features.length; i++) {
        const scaled = (features[i] - means[i]) / stds[i];
        scaledFeatures.push(scaled);
    }
    
    // Dot product
    let score = intercept;
    for (let i = 0; i < scaledFeatures.length; i++) {
        score += scaledFeatures[i] * weights[i];
    }
    
    // Clip score within valid 0 to 10 boundaries
    return Math.max(0.0, Math.min(10.0, score));
}

// Recursive Decision Tree Traversal
function predictDecisionTree(features, node) {
    // Base case: Leaf node
    if (node.type === "leaf") {
        return { value: node.value, probability: node.probability };
    }
    
    // Split node traversal
    const val = features[node.feature];
    if (val <= node.threshold) {
        return predictDecisionTree(features, node.left);
    } else {
        return predictDecisionTree(features, node.right);
    }
}

// Gather UI form values and run predictions
function runLiveInference() {
    if (!modelData) return;
    
    // Gather form inputs in target vector order:
    // [Age, Usage, Unlocks, Study, Phys, Sleep, Gender, Academic, Platform, Purpose]
    const features = [
        parseFloat(document.getElementById("input-age").value) || 21.0,
        parseFloat(document.getElementById("input-screentime").value) || 4.5,
        parseFloat(document.getElementById("input-unlocks").value) || 120.0,
        parseFloat(document.getElementById("input-study").value) || 4.0,
        parseFloat(document.getElementById("input-physical").value) || 1.5,
        parseFloat(document.getElementById("input-sleep").value) || 7.0,
        parseInt(document.getElementById("input-gender").value) || 0,
        parseInt(document.getElementById("input-academic").value) || 0,
        parseInt(document.getElementById("input-platform").value) || 0,
        parseInt(document.getElementById("input-purpose").value) || 0
    ];
    
    // 1. Predict Mental Health Score (Regression)
    const mhScore = predictLinearRegression(features, modelData.linear_regression);
    document.getElementById("pred-mh-score").innerText = mhScore.toFixed(1);
    
    // Fill rating bar
    const barFill = document.getElementById("pred-mh-bar");
    barFill.style.width = (mhScore * 10).toFixed(0) + "%";
    
    // Set comment based on rating scale
    let comment = "Severe risk to wellness. Action advised.";
    if (mhScore >= 7.5) {
        comment = "Excellent baseline. Keep up your habits!";
    } else if (mhScore >= 6.0) {
        comment = "Moderate baseline. Small changes can improve focus.";
    } else if (mhScore >= 4.5) {
        comment = "Fair baseline. Showing digital fatigue symptoms.";
    }
    document.getElementById("pred-mh-comment").innerText = comment;
    
    // 2. Predict Stress Level (Decision Tree Classifier)
    const dtResult = predictDecisionTree(features, modelData.decision_tree);
    const stressIdx = dtResult.value;
    const stressName = modelData.stress_level_inverse_mapping[stressIdx.toString()];
    
    const badge = document.getElementById("pred-stress-badge");
    badge.innerText = stressName;
    badge.style.backgroundColor = STRESS_COLORS[stressName];
    
    // Align needle gauge
    const needle = document.getElementById("risk-needle");
    const needlePercent = [12.5, 37.5, 62.5, 87.5][stressIdx];
    needle.style.left = needlePercent + "%";
}

// Bind live update listeners to form inputs
function bindSandboxEvents() {
    const inputs = document.querySelectorAll("#risk-form input, #risk-form select");
    inputs.forEach(input => {
        input.addEventListener("input", runLiveInference);
        input.addEventListener("change", runLiveInference);
    });
}

// --- RISK MANAGEMENT REPORT ENGINE ---
function generateRiskReport() {
    if (!modelData) return;
    
    // Gather inputs
    const age = parseFloat(document.getElementById("input-age").value) || 21;
    const screentime = parseFloat(document.getElementById("input-screentime").value) || 4.5;
    const unlocks = parseFloat(document.getElementById("input-unlocks").value) || 120;
    const study = parseFloat(document.getElementById("input-study").value) || 4.0;
    const physical = parseFloat(document.getElementById("input-physical").value) || 1.5;
    const sleep = parseFloat(document.getElementById("input-sleep").value) || 7.0;
    
    const genderSel = document.getElementById("input-gender");
    const genderStr = genderSel.options[genderSel.selectedIndex].text;
    
    const platformSel = document.getElementById("input-platform");
    const platformStr = platformSel.options[platformSel.selectedIndex].text;
    
    // Execute live inference to grab current results
    const features = [
        age, screentime, unlocks, study, physical, sleep,
        parseInt(genderSel.value) || 0,
        parseInt(document.getElementById("input-academic").value) || 0,
        parseInt(platformSel.value) || 0,
        parseInt(document.getElementById("input-purpose").value) || 0
    ];
    
    const mhScore = predictLinearRegression(features, modelData.linear_regression);
    const dtResult = predictDecisionTree(features, modelData.decision_tree);
    const stressIdx = dtResult.value;
    const stressName = modelData.stress_level_inverse_mapping[stressIdx.toString()];
    
    // Populate report summary values
    document.getElementById("rep-mh-score").innerText = mhScore.toFixed(1);
    document.getElementById("rep-stress-level").innerText = stressName;
    document.getElementById("rep-stress-level").style.color = STRESS_COLORS[stressName];
    
    // Format date
    const date = new Date();
    document.getElementById("report-date").innerText = date.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
    
    // 1. Identify specific risk factors (Risk Assessment Logic)
    const driversList = document.getElementById("report-risk-drivers");
    driversList.innerHTML = "";
    
    const riskDrivers = [];
    if (screentime > 5.5) {
        riskDrivers.push({ text: `Elevated Screen Time: ${screentime}h/day exceeds the 5.5h high-risk threshold.`, class: "danger" });
    }
    if (unlocks > 150) {
        riskDrivers.push({ text: `Frequent Device Unlocks: ${unlocks} unlocks/day indicates high dopamine loops.`, class: "danger" });
    }
    if (sleep < 6.0) {
        riskDrivers.push({ text: `Sleep Deprivation: ${sleep}h/night is below the physiological baseline.`, class: "danger" });
    }
    if (physical < 1.0) {
        riskDrivers.push({ text: `Sedentary Behaviour: ${physical}h/day physical activity is insufficient.`, class: "warning" });
    }
    
    if (riskDrivers.length === 0) {
        driversList.innerHTML = `<li class="warning" style="border-left-color: var(--color-success)">No critical risk drivers identified. Maintain healthy routines.</li>`;
    } else {
        riskDrivers.forEach(d => {
            const li = document.createElement("li");
            li.className = d.class;
            li.innerText = d.text;
            driversList.appendChild(li);
        });
    }
    
    // Write dynamic summary text based on predictions
    let summaryText = `Based on our from-scratch Gini Decision Tree and OLS regression models, the student is classified as having a `;
    if (stressIdx >= 2 || mhScore < 5.5) {
        summaryText += `<strong style="color:var(--color-danger)">HIGH RISK PROFILE</strong>. `;
        summaryText += `Compounded digital stressors (overuse of ${platformStr}) are severely impacting neurological recovery cycles (sleep and attention spans). immediate behavioural modifications are recommended.`;
    } else if (stressIdx === 1 || mhScore < 7.0) {
        summaryText += `<strong style="color:var(--color-warning)">MODERATE RISK PROFILE</strong>. `;
        summaryText += `The student displays symptoms of digital fatigue. While coping mechanisms are stable, habits such as late-night usage or high unlock rates should be managed to prevent transition to high-risk states.`;
    } else {
        summaryText += `<strong style="color:var(--color-success)">LOW RISK PROFILE</strong>. `;
        summaryText += `The student exhibits strong habits with balanced screen time, consistent sleep, and regular study intervals. Continue following current parameters.`;
    }
    document.getElementById("report-summary-text").innerHTML = summaryText;

    // 2. Populate Mitigation Protocols (Risk Management Strategies)
    const mitigationsGrid = document.getElementById("report-mitigations");
    mitigationsGrid.innerHTML = "";
    
    const detoxTarget = Math.max(2.0, screentime - 2.0).toFixed(1);
    const sleepTarget = Math.max(7.5, sleep + 1.0).toFixed(1);
    
    const protocols = [
        {
            title: "📱 Digital Detox Protocol",
            desc: `Reduce average screen time from ${screentime}h to a target of ${detoxTarget}h. Utilize app blockers for ${platformStr} during primary study blocks.`
        },
        {
            title: "🌙 Sleep Optimization",
            desc: `Increase sleep duration to ${sleepTarget}h. Implement a digital curfew (no screen usage 45 minutes prior to sleep) to restore circadian rhythm.`
        },
        {
            title: "🏋️ Physiological Recovery",
            desc: `Incorporate a minimum of 45 minutes of daily moderate exercise. Physical exertion directly offsets cortisol levels induced by excessive unlocks.`
        }
    ];
    
    protocols.forEach(p => {
        const div = document.createElement("div");
        div.className = "mitigation-card";
        div.innerHTML = `
            <h5>${p.title}</h5>
            <p>${p.desc}</p>
        `;
        mitigationsGrid.appendChild(div);
    });

    // Reveal report section and scroll smoothly to it
    const reportCard = document.getElementById("risk-report-card");
    reportCard.classList.remove("hidden");
    reportCard.scrollIntoView({ behavior: "smooth" });
}

// --- COHORTS DATA EXPLORER TABLE ---
function populateExplorerFilters(dataset) {
    const platformFilter = document.getElementById("filter-platform");
    const genderFilter = document.getElementById("filter-gender");
    
    // Extract unique platforms & genders
    const platforms = [...new Set(dataset.map(r => r.Most_Used_Platform))].sort();
    const genders = [...new Set(dataset.map(r => r.Gender))].sort();
    
    platforms.forEach(p => {
        if(p) platformFilter.add(new Option(p, p));
    });
    genders.forEach(g => {
        if(g) genderFilter.add(new Option(g, g));
    });
    
    // Bind change events
    platformFilter.addEventListener("change", applyTableFilters);
    genderFilter.addEventListener("change", applyTableFilters);
}

function applyTableFilters() {
    const platformVal = document.getElementById("filter-platform").value;
    const genderVal = document.getElementById("filter-gender").value;
    
    filteredDataset = rawDataset.filter(row => {
        const matchesPlatform = !platformVal || row.Most_Used_Platform === platformVal;
        const matchesGender = !genderVal || row.Gender === genderVal;
        return matchesPlatform && matchesGender;
    });
    
    currentPage = 1;
    renderTablePage(1);
}

function renderTablePage(page) {
    const tbody = document.querySelector("#data-sample-table tbody");
    tbody.innerHTML = "";
    
    const startIndex = (page - 1) * rowsPerPage;
    const endIndex = Math.min(startIndex + rowsPerPage, filteredDataset.length);
    
    const pageData = filteredDataset.slice(startIndex, endIndex);
    
    pageData.forEach(row => {
        const tr = document.createElement("tr");
        
        // Style stress levels cell inside table
        const stressBadge = `<span class="badge" style="background:${STRESS_COLORS[row.Stress_Level]}20; color:${STRESS_COLORS[row.Stress_Level]}; border: 1px solid ${STRESS_COLORS[row.Stress_Level]}40">${row.Stress_Level}</span>`;
        
        tr.innerHTML = `
            <td>${row.Age}</td>
            <td>${row.Gender}</td>
            <td>${row.Most_Used_Platform}</td>
            <td>${parseFloat(row.Avg_Daily_Usage_Hours).toFixed(1)}</td>
            <td>${row.Daily_Unlocks}</td>
            <td>${parseFloat(row.Study_Hours).toFixed(1)}</td>
            <td>${parseFloat(row.Sleep_Hours_Per_Night).toFixed(1)}</td>
            <td>${stressBadge}</td>
            <td><strong>${parseFloat(row.Mental_Health_Score).toFixed(1)}</strong></td>
        `;
        tbody.appendChild(tr);
    });
    
    // Update pagination text
    const totalCount = filteredDataset.length;
    const displayStart = totalCount === 0 ? 0 : startIndex + 1;
    document.getElementById("table-pagination-info").innerText = 
        `Showing ${displayStart}-${endIndex} of ${totalCount.toLocaleString()} records`;
        
    // Disable/enable buttons
    document.getElementById("btn-prev-page").disabled = page === 1;
    document.getElementById("btn-next-page").disabled = endIndex >= totalCount;
}

// --- ACTION BUTTON BINDINGS ---
function bindActionButtons() {
    // Generate wellness report
    document.getElementById("btn-generate-report").addEventListener("click", generateRiskReport);
    
    // Close report card
    document.getElementById("btn-close-report").addEventListener("click", () => {
        document.getElementById("risk-report-card").classList.add("hidden");
        document.getElementById("predictive-sandbox").scrollIntoView({ behavior: "smooth" });
    });
    
    // Pagination controls
    document.getElementById("btn-prev-page").addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            renderTablePage(currentPage);
        }
    });
    
    document.getElementById("btn-next-page").addEventListener("click", () => {
        const maxPage = Math.ceil(filteredDataset.length / rowsPerPage);
        if (currentPage < maxPage) {
            currentPage++;
            renderTablePage(currentPage);
        }
    });
}
