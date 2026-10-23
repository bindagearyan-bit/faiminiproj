/**
 * CardioAI - Clinical Machine Learning Prediction Controller
 * Powers real-time inference, slider-to-input bidirectional synchronization,
 * dynamic progress tracks, authentic Colab-style Seaborn/Matplotlib visualization,
 * and seamless live connection to Google Colab Cloud Backend.
 */

(function () {
    "use strict";

    const CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"];

    const DEFAULT_VALUES = {
        age: 55,
        sex: 1,
        cp: 4,
        trestbps: 130,
        chol: 240,
        fbs: 0,
        restecg: 0,
        thalach: 150,
        exang: 0,
        oldpeak: 1.0,
        slope: 2,
        ca: 0,
        thal: 3.0,
    };

    let currentVizOverride = "prob_gauge";
    let isAutoInferEnabled = true;
    let debounceTimer = null;
    let isSubmitting = false;

    // DOM Elements
    const form = document.getElementById("predictionForm");
    const autoInferToggle = document.getElementById("autoInferToggle");
    const btnClassify = document.getElementById("btnClassify");
    const btnRandomize = document.getElementById("btnRandomize");
    const btnResetDefaults = document.getElementById("btnResetDefaults");

    // Output Diagnostic Elements
    const resultCard = document.getElementById("resultCard");
    const statusPill = document.getElementById("statusPill");
    const diagnosisHeadline = document.getElementById("diagnosisHeadline");
    const riskTierVal = document.getElementById("riskTierVal");
    const probDiseaseText = document.getElementById("probDiseaseText");
    const probHealthyText = document.getElementById("probHealthyText");
    const probBarDisease = document.getElementById("probBarDisease");
    const probBarHealthy = document.getElementById("probBarHealthy");
    const factorsList = document.getElementById("factorsList");
    const latencyVal = document.getElementById("latencyVal");

    // Visualization Elements
    const vizCardTitle = document.getElementById("vizCardTitle");
    const vizLibraryBadge = document.getElementById("vizLibraryBadge");
    const vizImage = document.getElementById("vizImage");
    const plotLoader = document.getElementById("plotLoader");
    const vizReasonText = document.getElementById("vizReasonText");
    const switcherPills = document.getElementById("switcherPills");

    /**
     * Updates range slider background fill for an executive tactile appearance.
     */
    function updateSliderTrack(slider) {
        if (!slider) return;
        const min = parseFloat(slider.min) || 0;
        const max = parseFloat(slider.max) || 100;
        const val = parseFloat(slider.value) || min;
        const pct = Math.max(0, Math.min(100, ((val - min) / (max - min)) * 100));
        slider.style.background = `linear-gradient(to right, #06b6d4 0%, #3b82f6 ${pct}%, #1e293b ${pct}%, #1e293b 100%)`;
    }

    function setupContinuousSync() {
        CONTINUOUS_FEATURES.forEach((feat) => {
            const slider = document.getElementById(`${feat}_slider`);
            const numInput = document.getElementById(feat);

            if (!slider || !numInput) return;

            // Initial track fill
            updateSliderTrack(slider);

            slider.addEventListener("input", () => {
                numInput.value = slider.value;
                updateSliderTrack(slider);
                triggerAutoInference();
            });

            numInput.addEventListener("input", () => {
                slider.value = numInput.value;
                updateSliderTrack(slider);
                triggerAutoInference();
            });
        });
    }

    function setupSegmentedControls() {
        const controls = document.querySelectorAll(".segmented-selector, .segmented-control");
        controls.forEach((ctrl) => {
            const name = ctrl.getAttribute("data-name");
            const hiddenInput = document.getElementById(name);
            const buttons = ctrl.querySelectorAll(".segment-pill, .segment-btn");

            buttons.forEach((btn) => {
                btn.addEventListener("click", () => {
                    buttons.forEach((b) => b.classList.remove("active"));
                    btn.classList.add("active");
                    const val = btn.getAttribute("data-val");
                    if (hiddenInput) {
                        hiddenInput.value = val;
                        triggerAutoInference();
                    }
                });
            });
        });

        ["restecg", "slope", "thal"].forEach((id) => {
            const el = document.getElementById(id);
            if (el) {
                el.addEventListener("change", () => triggerAutoInference());
            }
        });
    }

    function getFormData() {
        const data = {};
        const formData = new FormData(form);

        for (const [key, value] of formData.entries()) {
            data[key] = parseFloat(value);
        }

        data.override_viz = currentVizOverride;
        return data;
    }

    function setFormData(data) {
        Object.keys(data).forEach((key) => {
            const val = data[key];

            const numInput = document.getElementById(key);
            const slider = document.getElementById(`${key}_slider`);
            if (numInput) numInput.value = val;
            if (slider) {
                slider.value = val;
                updateSliderTrack(slider);
            }

            const select = document.querySelector(`select#${key}`);
            if (select) {
                select.value = String(val);
                if (key === "thal" && !select.value) {
                    select.value = Number(val).toFixed(1);
                }
            }

            const segCtrl = document.querySelector(`.segmented-selector[data-name="${key}"], .segmented-control[data-name="${key}"]`);
            if (segCtrl) {
                const hiddenInput = document.getElementById(key);
                if (hiddenInput) hiddenInput.value = val;
                const buttons = segCtrl.querySelectorAll(".segment-pill, .segment-btn");
                buttons.forEach((b) => {
                    const btnVal = b.getAttribute("data-val");
                    if (String(btnVal) === String(val) || Number(btnVal) === Number(val)) {
                        b.classList.add("active");
                    } else {
                        b.classList.remove("active");
                    }
                });
            }
        });

        runPrediction();
    }

    function triggerAutoInference() {
        if (!isAutoInferEnabled) return;
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            runPrediction();
        }, 70);
    }

    async function runPrediction() {
        if (isSubmitting) return;
        isSubmitting = true;

        const payload = getFormData();

        if (plotLoader) plotLoader.style.display = "flex";
        if (vizImage) vizImage.style.display = "none";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            renderPredictionResults(data);
        } catch (err) {
            console.error("Prediction execution failed:", err);
            if (plotLoader) plotLoader.style.display = "none";
        } finally {
            isSubmitting = false;
        }
    }

    function renderPredictionResults(res) {
        if (latencyVal) {
            latencyVal.textContent = `${res.inference_time_ms} ms`;
        }

        if (resultCard) {
            resultCard.classList.remove("state-high-risk", "state-moderate-risk", "state-low-risk");
            if (isHigh) {
                resultCard.classList.add("state-high-risk");
            } else if (isModerate) {
                resultCard.classList.add("state-moderate-risk");
            } else {
                resultCard.classList.add("state-low-risk");
            }
        }

        if (statusPill) {
            if (isHigh) {
                statusPill.className = "status-tag red";
                statusPill.textContent = "🔴 Coronary Heart Disease (Class 1)";
            } else if (isModerate) {
                statusPill.className = "status-tag amber";
                statusPill.textContent = "🟡 Borderline / Moderate Risk";
            } else {
                statusPill.className = "status-tag green";
                statusPill.textContent = "🟢 Healthy Patient Profile (Class 0)";
            }
        }

        if (diagnosisHeadline) {
            if (isHigh) {
                diagnosisHeadline.textContent = "Heart Disease Risk Detected";
            } else if (isModerate) {
                diagnosisHeadline.textContent = "Borderline Cardiac Risk Factors";
            } else {
                diagnosisHeadline.textContent = "No Heart Disease Detected (Healthy)";
            }
        }

        if (riskTierVal) {
            if (isHigh) {
                riskTierVal.textContent = "HIGH RISK";
                riskTierVal.className = "box-value red";
            } else if (isModerate) {
                riskTierVal.textContent = "MODERATE RISK";
                riskTierVal.className = "box-value amber";
            } else {
                riskTierVal.textContent = "LOW RISK";
                riskTierVal.className = "box-value green";
            }
        }

        if (probDiseaseText) probDiseaseText.textContent = `CAD Risk: ${probCAD}%`;
        if (probHealthyText) probHealthyText.textContent = `Healthy: ${probHealthy}%`;
        if (probBarDisease) probBarDisease.style.width = `${probCAD}%`;
        if (probBarHealthy) probBarHealthy.style.width = `${probHealthy}%`;

        const probTrack = document.querySelector(".prob-track");
        if (probTrack) probTrack.setAttribute("aria-valuenow", String(probCAD));

        if (factorsList) {
            factorsList.innerHTML = "";
            if (res.key_factors && res.key_factors.length > 0) {
                res.key_factors.forEach((factor) => {
                    const li = document.createElement("li");
                    const isRiskFactor = factor.includes("High") || factor.includes("Silent") || factor.includes("Stress") || factor.includes("Blocked") || factor.includes("Strain");
                    li.className = isRiskFactor ? "factor-item factor-risk" : "factor-item factor-safe";
                    li.innerHTML = `<span class="factor-icon">${isRiskFactor ? "⚠️" : "✅"}</span> <span class="factor-text">${factor}</span>`;
                    factorsList.appendChild(li);
                });
            } else {
                const li = document.createElement("li");
                li.className = "factor-item factor-safe";
                li.innerHTML = `<span class="factor-icon">✅</span> <span class="factor-text">All primary health readings are within the standard normal baseline.</span>`;
                factorsList.appendChild(li);
            }
        }

        const viz = res.visualization;
        if (viz) {
            if (vizCardTitle) vizCardTitle.textContent = viz.title;
            if (vizLibraryBadge) vizLibraryBadge.textContent = viz.library_used;
            if (vizReasonText) vizReasonText.textContent = viz.reason;

            if (viz.image_base64 && vizImage) {
                vizImage.src = `data:image/png;base64,${viz.image_base64}`;
                vizImage.onload = () => {
                    if (plotLoader) plotLoader.style.display = "none";
                    vizImage.style.display = "block";
                };
            }
        }
    }

    function setupPresets() {
        const presetData = {
            "2": {
                age: 67,
                sex: 1,
                cp: 4,
                trestbps: 160,
                chol: 286,
                fbs: 0,
                restecg: 2,
                thalach: 108,
                exang: 1,
                oldpeak: 2.6,
                slope: 2,
                ca: 2.0,
                thal: 7.0,
            },
            "3": {
                age: 37,
                sex: 0,
                cp: 3,
                trestbps: 120,
                chol: 195,
                fbs: 0,
                restecg: 0,
                thalach: 178,
                exang: 0,
                oldpeak: 0.0,
                slope: 1,
                ca: 0.0,
                thal: 3.0,
            },
            "4": {
                age: 54,
                sex: 1,
                cp: 2,
                trestbps: 138,
                chol: 235,
                fbs: 0,
                restecg: 0,
                thalach: 150,
                exang: 0,
                oldpeak: 1.2,
                slope: 2,
                ca: 0.0,
                thal: 3.0,
            },
        };

        const presetButtons = document.querySelectorAll(".btn-case-preset");

        document.querySelectorAll(".btn-case-preset[data-preset]").forEach((btn) => {
            btn.addEventListener("click", () => {
                presetButtons.forEach((b) => b.classList.remove("active-preset"));
                btn.classList.add("active-preset");
                const presetKey = btn.getAttribute("data-preset");
                if (presetData[presetKey]) {
                    setFormData(presetData[presetKey]);
                }
            });
        });

        if (btnRandomize) {
            btnRandomize.addEventListener("click", () => {
                presetButtons.forEach((b) => b.classList.remove("active-preset"));
                btnRandomize.classList.add("active-preset");
                const randomPatient = {
                    age: Math.floor(Math.random() * (75 - 32 + 1)) + 32,
                    sex: Math.random() > 0.35 ? 1 : 0,
                    cp: [1, 2, 3, 4][Math.floor(Math.random() * 4)],
                    trestbps: Math.floor(Math.random() * (185 - 105 + 1)) + 105,
                    chol: Math.floor(Math.random() * (360 - 150 + 1)) + 150,
                    fbs: Math.random() > 0.8 ? 1 : 0,
                    restecg: [0, 1, 2][Math.floor(Math.random() * 3)],
                    thalach: Math.floor(Math.random() * (195 - 95 + 1)) + 95,
                    exang: Math.random() > 0.6 ? 1 : 0,
                    oldpeak: parseFloat((Math.random() * 3.8).toFixed(1)),
                    slope: [1, 2, 3][Math.floor(Math.random() * 3)],
                    ca: [0, 1, 2, 3][Math.floor(Math.random() * 4)],
                    thal: [3.0, 6.0, 7.0][Math.floor(Math.random() * 3)],
                };
                setFormData(randomPatient);
            });
        }

        if (btnResetDefaults) {
            btnResetDefaults.addEventListener("click", () => {
                presetButtons.forEach((b) => b.classList.remove("active-preset"));
                btnResetDefaults.classList.add("active-preset");
                setFormData(DEFAULT_VALUES);
            });
        }
    }

    function setupFacultySwitcher() {
        if (!switcherPills) return;

        const pills = switcherPills.querySelectorAll(".view-pill, .pill-btn");
        pills.forEach((pill) => {
            pill.addEventListener("click", () => {
                pills.forEach((p) => {
                    p.classList.remove("active");
                    p.setAttribute("aria-selected", "false");
                });
                pill.classList.add("active");
                pill.setAttribute("aria-selected", "true");
                currentVizOverride = pill.getAttribute("data-viz");
                runPrediction();
            });
        });
    }

    function init() {
        setupContinuousSync();
        setupSegmentedControls();
        setupPresets();
        setupFacultySwitcher();

        if (autoInferToggle) {
            autoInferToggle.addEventListener("change", (e) => {
                isAutoInferEnabled = e.target.checked;
            });
        }

        if (form) {
            form.addEventListener("submit", (e) => {
                e.preventDefault();
                runPrediction();
            });
        }

        runPrediction();
    }

    document.addEventListener("DOMContentLoaded", init);
})();
