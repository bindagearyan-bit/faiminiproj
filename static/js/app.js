/**
 * CardioStratify - Interactive Diagnostic Application Logic
 * Evidence-based live machine learning inference and visualization controller.
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

    let currentVizOverride = "auto";
    let isAutoInferEnabled = true;
    let debounceTimer = null;
    let isSubmitting = false;

    // DOM Elements
    const form = document.getElementById("predictionForm");
    const autoInferToggle = document.getElementById("autoInferToggle");
    const btnClassify = document.getElementById("btnClassify");
    const btnRandomize = document.getElementById("btnRandomize");
    const btnResetDefaults = document.getElementById("btnResetDefaults");

    // Output Elements
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

    function setupContinuousSync() {
        CONTINUOUS_FEATURES.forEach((feat) => {
            const slider = document.getElementById(`${feat}_slider`);
            const numInput = document.getElementById(feat);

            if (!slider || !numInput) return;

            slider.addEventListener("input", () => {
                numInput.value = slider.value;
                triggerAutoInference();
            });

            numInput.addEventListener("input", () => {
                slider.value = numInput.value;
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
            if (slider) slider.value = val;

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
        }, 180);
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
            if (statusPill) statusPill.textContent = "Evaluation Error";
            if (diagnosisHeadline) diagnosisHeadline.textContent = "Inference Pipeline Error";
        } finally {
            isSubmitting = false;
        }
    }

    function renderPredictionResults(res) {
        const isCAD = res.prediction === 1;
        const probCAD = res.prob_disease;
        const probHealthy = res.prob_no_disease;

        if (latencyVal) latencyVal.textContent = `${res.inference_time_ms} ms`;

        if (resultCard) {
            resultCard.classList.remove("state-disease", "state-healthy");
            resultCard.classList.add(isCAD ? "state-disease" : "state-healthy");
        }

        if (statusPill) {
            statusPill.className = "status-tag " + (isCAD ? "red" : "green");
            statusPill.textContent = isCAD ? "Positive (Class 1)" : "Normal Sinus (Class 0)";
        }

        if (diagnosisHeadline) {
            diagnosisHeadline.textContent = isCAD
                ? "Coronary Artery Disease Detected"
                : "Absence of Significant Coronary Disease";
        }

        if (riskTierVal) {
            riskTierVal.textContent = res.risk_tier.toUpperCase();
            riskTierVal.className = "box-value " + (
                res.risk_tier.includes("High") ? "red" :
                res.risk_tier.includes("Moderate") ? "amber" : "green"
            );
        }

        if (probDiseaseText) probDiseaseText.textContent = `CAD Probability: ${probCAD}%`;
        if (probHealthyText) probHealthyText.textContent = `Normal: ${probHealthy}%`;
        if (probBarDisease) probBarDisease.style.width = `${probCAD}%`;
        if (probBarHealthy) probBarHealthy.style.width = `${probHealthy}%`;

        const probTrack = document.querySelector(".prob-track");
        if (probTrack) probTrack.setAttribute("aria-valuenow", String(probCAD));

        if (factorsList) {
            factorsList.innerHTML = "";
            if (res.key_factors && res.key_factors.length > 0) {
                res.key_factors.forEach((factor) => {
                    const li = document.createElement("li");
                    li.textContent = factor;
                    factorsList.appendChild(li);
                });
            } else {
                const li = document.createElement("li");
                li.textContent = "All primary diagnostic parameters reside within non-critical clinical reference intervals.";
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

        document.querySelectorAll(".btn-case-preset[data-preset]").forEach((btn) => {
            btn.addEventListener("click", () => {
                const presetKey = btn.getAttribute("data-preset");
                if (presetData[presetKey]) {
                    setFormData(presetData[presetKey]);
                }
            });
        });

        if (btnRandomize) {
            btnRandomize.addEventListener("click", () => {
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
