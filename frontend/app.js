// Unified Single-Host: Automatically use the current origin when served over HTTP/HTTPS.
// Falls back to http://localhost:8000 only if opened directly as a file://.
const DEFAULT_API_URL = window.location.protocol.startsWith("http") 
  ? window.location.origin 
  : "http://localhost:8000";

let apiBaseUrl = localStorage.getItem("satquery_api_url") || DEFAULT_API_URL;
let currentMode = "single";

let fileSingle = null;
let fileSlot1 = null;
let fileSlot2 = null;

const QUERY_PRESETS = {
  single: [
    "What kind of land is this?",
    "Highlight the water body referred to in the query.",
    "Describe the land-cover and major objects visible in this image.",
    "Is this an agricultural field or a forested region?",
  ],
  change: [
    "What changed between these two dates, and where did the change occur?",
    "Has the built-up area increased, decreased, or remained unchanged?",
    "Quantify the vegetation loss and urban expansion.",
    "Describe all surface changes visible between T1 and T2.",
  ],
  fusion: [
    "Use the optical and SAR images together to identify built-up and water-covered regions.",
    "Detect structures and land features beneath the optical cloud cover.",
    "How does the radar backscatter complement the optical reflectance here?",
    "Distinguish between surface water and cloud shadows using SAR.",
  ],
};

const backendStatusEl = document.getElementById("backend-status");
const statusTextEl = document.getElementById("status-text");
const settingsToggleBtn = document.getElementById("settings-toggle-btn");
const settingsPanel = document.getElementById("settings-panel");
const apiUrlInput = document.getElementById("api-url-input");
const saveApiUrlBtn = document.getElementById("save-api-url-btn");
const downloadReportBtn = document.getElementById("download-report-btn");

const modeTabs = document.querySelectorAll(".mode-tab");
const tabIndicator = document.getElementById("tab-indicator");
const modeIndicatorBadge = document.getElementById("mode-indicator-badge");
const singleUploadContainer = document.getElementById("single-upload-container");
const dualUploadContainer = document.getElementById("dual-upload-container");
const slot1Label = document.getElementById("slot1-label");
const slot2Label = document.getElementById("slot2-label");

const dropZoneSingle = document.getElementById("drop-zone-single");
const imageInputSingle = document.getElementById("image-input-single");
const dropPromptSingle = document.getElementById("drop-prompt-single");
const previewContainerSingle = document.getElementById("preview-container-single");
const imagePreviewSingle = document.getElementById("image-preview-single");
const previewFilenameSingle = document.getElementById("preview-filename-single");
const previewFilesizeSingle = document.getElementById("preview-filesize-single");
const removeBtnSingle = document.getElementById("remove-btn-single");
const useSampleSingleBtn = document.getElementById("use-sample-single-btn");

const dropZone1 = document.getElementById("drop-zone-1");
const imageInput1 = document.getElementById("image-input-1");
const dropPrompt1 = document.getElementById("drop-prompt-1");
const previewContainer1 = document.getElementById("preview-container-1");
const imagePreview1 = document.getElementById("image-preview-1");
const previewFilename1 = document.getElementById("preview-filename-1");

const dropZone2 = document.getElementById("drop-zone-2");
const imageInput2 = document.getElementById("image-input-2");
const dropPrompt2 = document.getElementById("drop-prompt-2");
const previewContainer2 = document.getElementById("preview-container-2");
const imagePreview2 = document.getElementById("image-preview-2");
const previewFilename2 = document.getElementById("preview-filename-2");
const loadDemoPairBtn = document.getElementById("load-demo-pair-btn");

const questionInput = document.getElementById("question-input");
const quickQuestionsContainer = document.getElementById("quick-questions-container");
const analyzeBtn = document.getElementById("analyze-btn");
const analyzeBtnText = document.getElementById("analyze-btn-text");
const resetBtn = document.getElementById("reset-btn");
const errorMessageEl = document.getElementById("error-message");

const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const loadingStepText = document.getElementById("loading-step-text");
const resultsContent = document.getElementById("results-content");
const taskBadge = document.getElementById("task-badge");
const toolsBadge = document.getElementById("tools-badge");
const geminiAnswerEl = document.getElementById("gemini-answer");

const visualEvidenceBlock = document.getElementById("visual-evidence-block");
const evidenceTitle = document.getElementById("evidence-title");
const visualEvidenceImg = document.getElementById("visual-evidence-img");

const specialistBlockTitle = document.getElementById("specialist-block-title");
const modelTag = document.getElementById("model-tag");
const singleMetricsGrid = document.getElementById("single-metrics-grid");
const changeMetricsGrid = document.getElementById("change-metrics-grid");
const fusionMetricsGrid = document.getElementById("fusion-metrics-grid");
const probabilitiesSection = document.getElementById("probabilities-section");
const probsContainer = document.getElementById("probs-container");

const predictedClassEl = document.getElementById("predicted-class");
const confidenceValEl = document.getElementById("confidence-val");
const confidenceBarEl = document.getElementById("confidence-bar");

const changeDynamicVal = document.getElementById("change-dynamic-val");
const totalChangeVal = document.getElementById("total-change-val");
const vegLossVal = document.getElementById("veg-loss-val");
const builtupGainVal = document.getElementById("builtup-gain-val");

const fusionBuiltupVal = document.getElementById("fusion-builtup-val");
const fusionWaterVal = document.getElementById("fusion-water-val");
const fusionCloudVal = document.getElementById("fusion-cloud-val");
const fusionPenetratedVal = document.getElementById("fusion-penetrated-val");

const auditToggle = document.getElementById("audit-toggle");
const auditArrow = document.getElementById("audit-arrow");
const auditBody = document.getElementById("audit-body");
const auditList = document.getElementById("audit-list");
const auditCount = document.getElementById("audit-count");

const heroLaunchBtn = document.getElementById("hero-launch-btn");
const scrollCueBtn = document.getElementById("scroll-cue");
const heroCopy = document.querySelector(".hero-copy");
const consoleEl = document.getElementById("console");

document.addEventListener("DOMContentLoaded", () => {
  apiUrlInput.value = apiBaseUrl;
  checkBackendHealth();
  setInterval(checkBackendHealth, 10000);
  setupEventListeners();
  updateModeView("single");
  loadAuditTrail();
  requestAnimationFrame(positionTabIndicator);
  window.addEventListener("resize", positionTabIndicator);
  runHeroEntrance();
  setupHeroScroll();
});

function runHeroEntrance() {
  if (!heroCopy) return;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const bgVideo = document.querySelector(".space-globe-video");
  if (bgVideo) {
    const SLOW_RATE = 1.10;
    const pinRate = () => { bgVideo.playbackRate = SLOW_RATE; };
    pinRate();
    bgVideo.addEventListener("loadedmetadata", pinRate);
    bgVideo.addEventListener("play", pinRate);
    bgVideo.addEventListener("ratechange", () => {
      if (Math.abs(bgVideo.playbackRate - SLOW_RATE) > 0.001) pinRate();
    });
    if (reduceMotion) {
      bgVideo.pause();
      bgVideo.removeAttribute("autoplay");
    } else {
      bgVideo.play().catch(() => {});
    }
  }

  if (reduceMotion) {
    heroCopy.classList.add("is-ready");
    return;
  }
  const reveal = () => requestAnimationFrame(() => requestAnimationFrame(() => heroCopy.classList.add("is-ready")));
  if (document.fonts && document.fonts.ready) {
    let done = false;
    document.fonts.ready.then(() => { if (!done) { done = true; reveal(); } });
    setTimeout(() => { if (!done) { done = true; reveal(); } }, 250);
  } else {
    reveal();
  }
}

function setupHeroScroll() {
  const scrollToConsole = () => {
    if (consoleEl) consoleEl.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  if (heroLaunchBtn) heroLaunchBtn.addEventListener("click", scrollToConsole);
  if (scrollCueBtn) scrollCueBtn.addEventListener("click", scrollToConsole);
}

function positionTabIndicator() {
  const activeTab = document.querySelector(".mode-tab.active");
  if (!activeTab || !tabIndicator) return;
  tabIndicator.style.width = `${activeTab.offsetWidth}px`;
  tabIndicator.style.transform = `translateX(${activeTab.offsetLeft - 6}px)`;
}

function animateCountTo(el, target, { prefix = "", suffix = "%", duration = 700 } = {}) {
  if (!el) return;
  const start = 0;
  const startTime = performance.now();
  const ease = (t) => 1 - Math.pow(1 - t, 3);

  function tick(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const value = start + (target - start) * ease(progress);
    el.textContent = `${prefix}${Math.round(value)}${suffix}`;
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function setupEventListeners() {
  modeTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      modeTabs.forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const mode = tab.getAttribute("data-mode");
      updateModeView(mode);
      positionTabIndicator();
    });
  });

  settingsToggleBtn.addEventListener("click", () => settingsPanel.classList.toggle("hidden"));
  saveApiUrlBtn.addEventListener("click", () => {
    const url = apiUrlInput.value.trim().replace(/\/+$/, "");
    if (url) {
      apiBaseUrl = url;
      localStorage.setItem("satquery_api_url", apiBaseUrl);
      settingsPanel.classList.add("hidden");
      checkBackendHealth();
      loadAuditTrail();
    }
  });

  downloadReportBtn.addEventListener("click", handleDownloadReport);

  bindDropZone(dropZoneSingle, imageInputSingle, (file) => {
    fileSingle = file;
    renderFilePreview(file, imagePreviewSingle, previewContainerSingle, dropPromptSingle, previewFilenameSingle, previewFilesizeSingle);
  });
  removeBtnSingle.addEventListener("click", (e) => {
    e.stopPropagation();
    fileSingle = null;
    imageInputSingle.value = "";
    dropPromptSingle.classList.remove("hidden");
    previewContainerSingle.classList.add("hidden");
  });
  useSampleSingleBtn.addEventListener("click", loadSingleSample);

  bindDropZone(dropZone1, imageInput1, (file) => {
    fileSlot1 = file;
    renderSlotPreview(file, imagePreview1, previewContainer1, dropPrompt1, previewFilename1);
  });
  bindDropZone(dropZone2, imageInput2, (file) => {
    fileSlot2 = file;
    renderSlotPreview(file, imagePreview2, previewContainer2, dropPrompt2, previewFilename2);
  });
  loadDemoPairBtn.addEventListener("click", loadDemoPair);

  analyzeBtn.addEventListener("click", handleAnalyze);
  resetBtn.addEventListener("click", handleReset);

  auditToggle.addEventListener("click", (e) => {
    if (e.target.closest(".audit-actions")) return;
    auditBody.classList.toggle("hidden");
    auditArrow.classList.toggle("open");
  });
}

function updateModeView(mode) {
  currentMode = mode;
  hideError();

  if (mode === "single") {
    modeIndicatorBadge.textContent = "Single Image Mode";
    modeIndicatorBadge.className = "badge badge-primary";
    singleUploadContainer.classList.remove("hidden");
    dualUploadContainer.classList.add("hidden");
  } else if (mode === "change") {
    modeIndicatorBadge.textContent = "Bi-Temporal Change Mode";
    modeIndicatorBadge.className = "badge badge-info";
    singleUploadContainer.classList.add("hidden");
    dualUploadContainer.classList.remove("hidden");
    slot1Label.textContent = "Image T1 (Before / Baseline)";
    slot2Label.textContent = "Image T2 (After / Resurvey)";
  } else if (mode === "fusion") {
    modeIndicatorBadge.textContent = "Optical + SAR Cross-Modal Mode";
    modeIndicatorBadge.className = "badge badge-violet";
    singleUploadContainer.classList.add("hidden");
    dualUploadContainer.classList.remove("hidden");
    slot1Label.textContent = "Optical Multispectral Image";
    slot2Label.textContent = "SAR Radar Backscatter Image";
  }

  renderQueryChips(mode);
}

function renderQueryChips(mode) {
  const queries = QUERY_PRESETS[mode] || [];
  quickQuestionsContainer.innerHTML = `<span class="quick-label">Representative Queries:</span>`;

  queries.forEach((q, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "chip-btn";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      questionInput.value = q;
      questionInput.focus();
    });
    quickQuestionsContainer.appendChild(btn);

    if (idx === 0) {
      questionInput.value = q;
    }
  });
}

function bindDropZone(zone, input, onFileSelected) {
  zone.addEventListener("click", () => input.click());
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragover");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onFileSelected(e.dataTransfer.files[0]);
    }
  });
  input.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onFileSelected(e.target.files[0]);
    }
  });
}

/**
 * Browsers cannot natively decode TIFF/GeoTIFF inside an <img> tag, in any
 * browser. So for .tif/.tiff files there is exactly one path to a visible
 * preview: converting it server-side via POST /preview. If that call fails,
 * we must show a clear "preview unavailable" state -- never fall back to
 * pointing <img src> at the raw TIFF bytes, since that silently renders as
 * a broken-image icon with no explanation.
 */
async function renderTiffOrImagePreview(file, imgEl, container, promptEl, nameEl, sizeEl) {
  nameEl.textContent = file.name;
  if (sizeEl) {
    const kb = (file.size / 1024).toFixed(1);
    sizeEl.textContent = kb > 1024 ? `${(kb / 1024).toFixed(2)} MB` : `${kb} KB`;
  }

  const isTiff = file.name.toLowerCase().endsWith(".tif") || file.name.toLowerCase().endsWith(".tiff");

  promptEl.classList.add("hidden");
  container.classList.remove("hidden");
  container.classList.remove("preview-error");

  if (!isTiff) {
    // PNG/JPEG etc. render natively in the browser -- no backend round trip needed.
    const reader = new FileReader();
    reader.onload = (e) => {
      imgEl.src = e.target.result;
    };
    reader.onerror = () => {
      showPreviewError(container, imgEl, "Could not read this file.");
    };
    reader.readAsDataURL(file);
    return;
  }

  // TIFF/GeoTIFF: must be converted server-side to a browser-displayable PNG.
  imgEl.removeAttribute("src");
  container.classList.add("preview-loading");
  nameEl.textContent = `${file.name} — generating preview…`;

  try {
    const formData = new FormData();
    formData.append("image", file, file.name);
    const res = await fetch(`${apiBaseUrl}/preview`, { method: "POST", body: formData });

    if (!res.ok) {
      let detail = `Server returned ${res.status}`;
      try {
        const errBody = await res.json();
        if (errBody && errBody.detail) detail = errBody.detail;
      } catch (_) {}
      throw new Error(detail);
    }

    const data = await res.json();
    if (!data.preview_b64) throw new Error("Backend returned no preview data.");

    imgEl.src = data.preview_b64;
    nameEl.textContent = file.name;
  } catch (err) {
    console.error("TIFF preview failed:", err);
    showPreviewError(container, imgEl, `Preview unavailable: ${err.message}`);
    nameEl.textContent = file.name;
  } finally {
    container.classList.remove("preview-loading");
  }
}

function showPreviewError(container, imgEl, message) {
  imgEl.removeAttribute("src");
  container.classList.add("preview-error");
  let errEl = container.querySelector(".preview-error-message");
  if (!errEl) {
    errEl = document.createElement("div");
    errEl.className = "preview-error-message";
    container.appendChild(errEl);
  }
  errEl.textContent = message;
}

// Kept as thin wrappers so existing call sites (bindDropZone callbacks, etc.) don't change.
async function renderFilePreview(file, imgEl, container, promptEl, nameEl, sizeEl) {
  return renderTiffOrImagePreview(file, imgEl, container, promptEl, nameEl, sizeEl);
}

async function renderSlotPreview(file, imgEl, container, promptEl, nameEl) {
  return renderTiffOrImagePreview(file, imgEl, container, promptEl, nameEl, null);
}

function loadSingleSample() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  ctx.fillStyle = "#1e4620";
  ctx.fillRect(0, 0, 256, 256);
  ctx.strokeStyle = "#1d3557";
  ctx.lineWidth = 20;
  ctx.beginPath();
  ctx.moveTo(0, 40);
  ctx.bezierCurveTo(80, 80, 160, 20, 256, 140);
  ctx.stroke();

  canvas.toBlob((blob) => {
    const file = new File([blob], "sample_sentinel2_patch.png", { type: "image/png" });
    fileSingle = file;
    renderFilePreview(file, imagePreviewSingle, previewContainerSingle, dropPromptSingle, previewFilenameSingle, previewFilesizeSingle);
  }, "image/png");
}

async function loadDemoPair() {
  if (currentMode === "change") {
    fileSlot1 = await createSimulatedFile("bitemporal_t1_before.png", "#226422", "#19376e");
    fileSlot2 = await createSimulatedFile("bitemporal_t2_after.png", "#82878c", "#19376e");
    questionInput.value = "What changed between these two dates, and where did the change occur?";
  } else {
    fileSlot1 = await createSimulatedFile("crossmodal_optical.png", "#286e2d", "#1e3c78");
    fileSlot2 = await createSimulatedFile("crossmodal_sar.png", "#5a5a5a", "#e6e6e6");
    questionInput.value = "Use the optical and SAR images together to identify built-up and water-covered regions.";
  }

  renderSlotPreview(fileSlot1, imagePreview1, previewContainer1, dropPrompt1, previewFilename1);
  renderSlotPreview(fileSlot2, imagePreview2, previewContainer2, dropPrompt2, previewFilename2);
}

function createSimulatedFile(filename, c1, c2) {
  return new Promise((resolve) => {
    const c = document.createElement("canvas");
    c.width = 256;
    c.height = 256;
    const ctx = c.getContext("2d");
    ctx.fillStyle = c1;
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = c2;
    ctx.fillRect(50, 50, 150, 150);
    c.toBlob((blob) => {
      resolve(new File([blob], filename, { type: "image/png" }));
    }, "image/png");
  });
}

async function checkBackendHealth() {
  try {
    const res = await fetch(`${apiBaseUrl}/health`, { method: "GET" });
    if (res.ok) {
      const data = await res.json();
      backendStatusEl.className = "status-badge online";
      statusTextEl.textContent = `Backend Online (${data.service || "SatQuery AI"})`;
      backendStatusEl.title = `Connected to ${apiBaseUrl}`;
      return true;
    }
  } catch (_) {}
  backendStatusEl.className = "status-badge offline";
  statusTextEl.textContent = "Backend Offline";
  backendStatusEl.title = `Could not reach ${apiBaseUrl}.`;
  return false;
}

async function handleAnalyze() {
  hideError();

  const question = questionInput.value.trim();
  if (!question) {
    showError("Please enter a question or select a representative query.");
    return;
  }

  const formData = new FormData();
  formData.append("question", question);

  if (currentMode === "single") {
    if (!fileSingle) {
      showError("Please upload or select a satellite image.");
      return;
    }
    formData.append("image", fileSingle, fileSingle.name);
    if (question.toLowerCase().includes("highlight") || question.toLowerCase().includes("locate")) {
      formData.append("task_hint", "single_image_grounding");
    } else {
      formData.append("task_hint", "single_image_vqa");
    }
  } else {
    if (!fileSlot1 || !fileSlot2) {
      showError(`Please upload both images for ${currentMode === "change" ? "Bi-temporal Change" : "Optical+SAR Fusion"} mode.`);
      return;
    }
    formData.append("image", fileSlot1, fileSlot1.name);
    formData.append("image2", fileSlot2, fileSlot2.name);
    formData.append("task_hint", currentMode === "change" ? "bi_temporal_change" : "cross_modal_sar_optical");
  }

  setLoadingState(true);

  try {
    const res = await fetch(`${apiBaseUrl}/analyze`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      let errDetail = `Server returned HTTP ${res.status}`;
      try {
        const errorJson = await res.json();
        if (errorJson && errorJson.detail) errDetail = errorJson.detail;
      } catch (_) {}
      throw new Error(errDetail);
    }

    const data = await res.json();
    renderAnalysisResults(data);
    loadAuditTrail();
  } catch (err) {
    if (err.message && err.message.includes("Failed to fetch")) {
      showError(`Connection failed: Could not reach backend at ${apiBaseUrl}.`);
    } else {
      showError(`Execution error: ${err.message}`);
    }
  } finally {
    setLoadingState(false);
  }
}

function renderAnalysisResults(data) {
  emptyState.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  taskBadge.textContent = data.task_type || currentMode;
  taskBadge.classList.remove("hidden");

  if (data.tools_executed) {
    toolsBadge.textContent = `Tools: ${data.tools_executed.join(" + ")}`;
    toolsBadge.classList.remove("hidden");
  }

  geminiAnswerEl.innerHTML = marked.parse(data.answer || "No synthesis returned.");

  if (data.visual_evidence_b64) {
    visualEvidenceBlock.classList.remove("hidden");
    visualEvidenceImg.src = data.visual_evidence_b64;

    if (data.task_type === "bi_temporal_change") {
      evidenceTitle.textContent = "Bi-Temporal Spatial Change Heatmap";
    } else if (data.task_type === "cross_modal_sar_optical") {
      evidenceTitle.textContent = "Optical-SAR Cross-Modal Composite Map";
    } else {
      evidenceTitle.textContent = "Text-Guided Spatial Grounding Mask";
    }
  } else {
    visualEvidenceBlock.classList.add("hidden");
  }

  singleMetricsGrid.classList.add("hidden");
  changeMetricsGrid.classList.add("hidden");
  fusionMetricsGrid.classList.add("hidden");
  probabilitiesSection.classList.add("hidden");

  const spec = data.specialist_result || {};

  if (data.task_type === "bi_temporal_change") {
    specialistBlockTitle.textContent = "Bi-Temporal Change Metrics";
    modelTag.textContent = "change_detector_cva";
    changeMetricsGrid.classList.remove("hidden");

    changeDynamicVal.textContent = spec.dominant_trend || "Shift";
    animateCountTo(totalChangeVal, spec.total_change_percentage || 0);
    animateCountTo(vegLossVal, spec.vegetation_loss_percentage || 0, { prefix: "-" });
    animateCountTo(builtupGainVal, spec.built_up_gain_percentage || 0, { prefix: "+" });
  } else if (data.task_type === "cross_modal_sar_optical") {
    specialistBlockTitle.textContent = "Optical-SAR Cross-Modal Metrics";
    modelTag.textContent = "sar_optical_fusion";
    fusionMetricsGrid.classList.remove("hidden");

    animateCountTo(fusionBuiltupVal, spec.built_up_coverage_percentage || 0);
    animateCountTo(fusionWaterVal, spec.water_coverage_percentage || 0);
    animateCountTo(fusionCloudVal, spec.optical_cloud_coverage_percentage || 0);
    animateCountTo(fusionPenetratedVal, spec.radar_cloud_penetration_percentage || 0);
  } else {
    specialistBlockTitle.textContent = "Specialist Land-Cover Verdict";
    modelTag.textContent = "rs-eurosat-classifier";
    singleMetricsGrid.classList.remove("hidden");

    predictedClassEl.textContent = spec.predicted_class || "Unknown";
    const confidence = spec.confidence !== undefined ? Math.round(spec.confidence * 100) : 0;
    animateCountTo(confidenceValEl, confidence, { duration: 800 });
    confidenceBarEl.style.width = `${confidence}%`;

    if (spec.all_probs) {
      probabilitiesSection.classList.remove("hidden");
      probsContainer.innerHTML = "";
      const sorted = Object.entries(spec.all_probs).sort((a, b) => b[1] - a[1]);
      const maxVal = sorted.length > 0 ? sorted[0][1] : 1;

      sorted.forEach(([clsName, prob], idx) => {
        const pct = (prob * 100).toFixed(1);
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
          <span class="prob-name" title="${clsName}">${clsName}</span>
          <div class="prob-bar-container">
            <div class="prob-bar-fill ${idx === 0 ? "top-rank" : ""}" style="width: ${(prob / maxVal) * 100}%"></div>
          </div>
          <span class="prob-pct">${pct}%</span>
        `;
        probsContainer.appendChild(row);
      });
    }
  }
}

async function handleDownloadReport() {
  try {
    const res = await fetch(`${apiBaseUrl}/download-report`);
    if (!res.ok) throw new Error("Could not generate report");
    const text = await res.text();

    const blob = new Blob([text], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `satquery_audit_report_${Date.now()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    showError(`Failed to download audit report: ${err.message}`);
  }
}

async function loadAuditTrail() {
  try {
    const res = await fetch(`${apiBaseUrl}/audit-trail`);
    if (!res.ok) return;
    const data = await res.json();
    const entries = data.entries || [];

    auditCount.textContent = `${entries.length} step${entries.length === 1 ? "" : "s"}`;

    if (entries.length === 0) {
      auditList.innerHTML = `<div class="audit-empty">No recent audit logs found. Execute a query to observe logged agent decisions.</div>`;
      return;
    }

    auditList.innerHTML = "";
    const reversed = [...entries].reverse();

    reversed.forEach((item) => {
      const entryEl = document.createElement("div");
      entryEl.className = "audit-item";
      const timeStr = item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : "";
      const detailsStr = item.details ? JSON.stringify(item.details, null, 2) : "{}";

      entryEl.innerHTML = `
        <div class="audit-item-head">
          <span class="audit-step-tag">${escapeHtml(item.step || "step")}</span>
          <span class="audit-time">${timeStr}</span>
        </div>
        <pre class="audit-details"><code>${escapeHtml(detailsStr)}</code></pre>
      `;
      auditList.appendChild(entryEl);
    });
  } catch (_) {}
}

function setLoadingState(isLoading) {
  if (isLoading) {
    analyzeBtn.disabled = true;
    analyzeBtnText.textContent = "Orchestrating...";
    emptyState.classList.add("hidden");
    resultsContent.classList.add("hidden");
    loadingState.classList.remove("hidden");
  } else {
    analyzeBtn.disabled = false;
    analyzeBtnText.textContent = "Execute Specialist Workflow";
    loadingState.classList.add("hidden");
  }
}

function handleReset() {
  fileSingle = null;
  fileSlot1 = null;
  fileSlot2 = null;

  imageInputSingle.value = "";
  imagePreviewSingle.src = "";
  dropPromptSingle.classList.remove("hidden");
  previewContainerSingle.classList.add("hidden");

  imageInput1.value = "";
  imagePreview1.src = "";
  dropPrompt1.classList.remove("hidden");
  previewContainer1.classList.add("hidden");

  imageInput2.value = "";
  imagePreview2.src = "";
  dropPrompt2.classList.remove("hidden");
  previewContainer2.classList.add("hidden");

  hideError();
  taskBadge.classList.add("hidden");
  toolsBadge.classList.add("hidden");
  resultsContent.classList.add("hidden");
  visualEvidenceBlock.classList.add("hidden");
  emptyState.classList.remove("hidden");
  renderQueryChips(currentMode);
}

function showError(msg) {
  errorMessageEl.textContent = msg;
  errorMessageEl.classList.remove("hidden");
}

function hideError() {
  errorMessageEl.classList.add("hidden");
  errorMessageEl.textContent = "";
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}