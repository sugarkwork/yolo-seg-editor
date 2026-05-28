document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById('seg-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('editor-container');
    const sourceImage = document.getElementById('source-image');

    // Render the canvas backing buffer at this multiple of the image's natural
    // size so CSS-zoom can stretch up to `RESOLUTION_MULTIPLIER`x before pixels
    // are interpolated by the browser. The canvas's CSS size still tracks the
    // image's natural size, so visual layout is unchanged at scale=1.
    const RESOLUTION_MULTIPLIER = 2;

    // State
    let polygons = []; // Array of { classId, points: [{x, y}] }
    let currentPolygon = null;
    let activeClassId = 0;

    // Viewport State
    let scale = 1;
    let offsetX = 0;
    let offsetY = 0;
    let isPanning = false;
    let startPanX = 0;
    let startPanY = 0;

    // Selection state
    let selectedPolygonIndex = -1;
    let hoveredPolygonIndex = -1;
    let hoveredVertexIndex = -1;
    let isDraggingVertex = false;

    // UI Setup
    const classSelectors = document.querySelectorAll('.class-selector');
    classSelectors.forEach(btn => {
        btn.addEventListener('click', () => {
            // Update UI
            classSelectors.forEach(b => {
                b.classList.remove('bg-indigo-500/20', 'border-indigo-500/50', 'text-indigo-300');
                b.classList.add('bg-slate-800/50', 'text-slate-300', 'border-transparent');
            });
            btn.classList.add('bg-indigo-500/20', 'border-indigo-500/50', 'text-indigo-300');
            btn.classList.remove('bg-slate-800/50', 'text-slate-300', 'border-transparent');

            // Set active class
            activeClassId = parseInt(btn.dataset.classId);

            // Change selected polygon's class if one is selected
            if (selectedPolygonIndex >= 0) {
                if (polygons[selectedPolygonIndex].classId !== activeClassId) {
                    polygons[selectedPolygonIndex].classId = activeClassId;
                    draw();
                    saveHistory();
                }
            }
        });
    });

    const addClassBtn = document.getElementById('add-class-btn');
    const newClassInput = document.getElementById('new-class-input');

    // Auto-Segment UI Setup
    const modelSelector = document.getElementById('model-selector');
    const autoSegmentBtn = document.getElementById('auto-segment-btn');
    const aiStatus = document.getElementById('ai-status');
    const aiError = document.getElementById('ai-error');
    const checkDiffBtn = document.getElementById('check-diff-btn');
    const diffResult = document.getElementById('diff-result');

    async function populateModels() {
        if (!modelSelector) return;
        try {
            const resp = await fetch('/api/models');
            if (resp.ok) {
                const data = await resp.json();
                modelSelector.innerHTML = '';
                if (data.models.length === 0) {
                    modelSelector.innerHTML = '<option value="">No models found in models/</option>';
                    autoSegmentBtn.disabled = true;
                } else {
                    data.models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.textContent = m;
                        modelSelector.appendChild(opt);
                    });
                    const lastModel = localStorage.getItem('last_auto_segment_model');
                    if (lastModel && data.models.includes(lastModel)) {
                        modelSelector.value = lastModel;
                    }
                    modelSelector.addEventListener('change', () => {
                        localStorage.setItem('last_auto_segment_model', modelSelector.value);
                    });
                    autoSegmentBtn.disabled = false;
                }
            } else {
                modelSelector.innerHTML = '<option value="">Error fetching models</option>';
            }
        } catch (e) {
            console.error("Failed to fetch models", e);
        }
    }

    populateModels();

    const useDenoiseChk = document.getElementById('use-denoise-chk');
    const denoiseParams = document.getElementById('denoise-params');
    const dnHLum = document.getElementById('dn-h-lum');
    const dnHCol = document.getElementById('dn-h-col');
    const dnTw = document.getElementById('dn-tw');
    const dnSw = document.getElementById('dn-sw');

    function loadDenoiseSettings() {
        if (!useDenoiseChk) return;
        const saved = localStorage.getItem('dataset_editor_denoise_settings');
        if (saved) {
            try {
                const s = JSON.parse(saved);
                useDenoiseChk.checked = s.useDenoise || false;
                if (s.hLum) dnHLum.value = s.hLum;
                if (s.hCol) dnHCol.value = s.hCol;
                if (s.tw) dnTw.value = s.tw;
                if (s.sw) dnSw.value = s.sw;
            } catch (e) {}
        }
        denoiseParams.classList.toggle('hidden', !useDenoiseChk.checked);
    }

    function saveDenoiseSettings() {
        if (!useDenoiseChk) return;
        const s = {
            useDenoise: useDenoiseChk.checked,
            hLum: parseFloat(dnHLum.value),
            hCol: parseFloat(dnHCol.value),
            tw: parseInt(dnTw.value, 10),
            sw: parseInt(dnSw.value, 10)
        };
        localStorage.setItem('dataset_editor_denoise_settings', JSON.stringify(s));
    }

    if (useDenoiseChk) {
        loadDenoiseSettings();
        useDenoiseChk.addEventListener('change', () => {
            denoiseParams.classList.toggle('hidden', !useDenoiseChk.checked);
            saveDenoiseSettings();
        });
        [dnHLum, dnHCol, dnTw, dnSw].forEach(el => {
            if (el) el.addEventListener('change', saveDenoiseSettings);
        });
    }

    const shrinkBtn = document.getElementById('shrink-btn');
    const shrinkPercentInput = document.getElementById('shrink-percent');
    const shrinkError = document.getElementById('shrink-error');

    if (shrinkPercentInput) {
        const savedShrink = localStorage.getItem('dataset_editor_shrink_percent');
        if (savedShrink !== null) shrinkPercentInput.value = savedShrink;
        shrinkPercentInput.addEventListener('change', () => {
            localStorage.setItem('dataset_editor_shrink_percent', shrinkPercentInput.value);
        });
    }

    if (shrinkBtn) {
        shrinkBtn.addEventListener('click', async () => {
            shrinkError.classList.add('hidden');
            const pct = parseFloat(shrinkPercentInput.value);
            if (!(pct > 0)) {
                shrinkError.textContent = 'Enter a positive percent.';
                shrinkError.classList.remove('hidden');
                return;
            }
            if (polygons.length === 0) {
                shrinkError.textContent = 'No polygons to shrink.';
                shrinkError.classList.remove('hidden');
                return;
            }

            const indices = selectedPolygonIndex >= 0 ? [selectedPolygonIndex] : null;

            shrinkBtn.disabled = true;
            try {
                const payload = {
                    polygons: polygons.map(p => ({
                        classId: p.classId,
                        points: p.points.map(pt => ({ x: pt.x, y: pt.y }))
                    })),
                    shrink_percent: pct,
                    indices: indices
                };
                const resp = await fetch('/api/shrink_polygons', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!resp.ok) {
                    const errDetail = await resp.json().catch(() => ({}));
                    shrinkError.textContent = 'Error: ' + (errDetail.detail || resp.statusText);
                    shrinkError.classList.remove('hidden');
                    return;
                }
                const data = await resp.json();
                const newPolys = [];
                let selectedSurvived = false;
                let newSelectedIndex = -1;
                data.polygons.forEach((p, i) => {
                    if (p.points && p.points.length >= 3) {
                        if (i === selectedPolygonIndex) {
                            selectedSurvived = true;
                            newSelectedIndex = newPolys.length;
                        }
                        newPolys.push({ classId: p.classId, points: p.points });
                    }
                });
                polygons = newPolys;
                selectedPolygonIndex = selectedSurvived ? newSelectedIndex : -1;
                saveHistory();
                draw();
            } catch (e) {
                console.error(e);
                shrinkError.textContent = 'Network error.';
                shrinkError.classList.remove('hidden');
            } finally {
                shrinkBtn.disabled = false;
            }
        });
    }

    const snapBtn = document.getElementById('snap-btn');
    const snapIterInput = document.getElementById('snap-iter');
    const snapMarginInput = document.getElementById('snap-margin');
    const snapSmoothInput = document.getElementById('snap-smooth');
    const snapError = document.getElementById('snap-error');

    [['snap-iter', 'dataset_editor_snap_iter'], ['snap-margin', 'dataset_editor_snap_margin'], ['snap-smooth', 'dataset_editor_snap_smooth']].forEach(([id, key]) => {
        const el = document.getElementById(id);
        if (!el) return;
        const saved = localStorage.getItem(key);
        if (saved !== null) el.value = saved;
        el.addEventListener('change', () => localStorage.setItem(key, el.value));
    });

    if (snapBtn) {
        snapBtn.addEventListener('click', async () => {
            snapError.classList.add('hidden');
            if (polygons.length === 0) {
                snapError.textContent = 'No polygons to snap.';
                snapError.classList.remove('hidden');
                return;
            }

            const indices = selectedPolygonIndex >= 0 ? [selectedPolygonIndex] : null;
            const iterations = parseInt(snapIterInput.value, 10);
            const margin = parseInt(snapMarginInput.value, 10);
            const smooth = parseFloat(snapSmoothInput.value);

            snapBtn.disabled = true;
            const origText = snapBtn.textContent;
            snapBtn.textContent = 'Snapping...';
            try {
                const payload = {
                    image_path: window.IMAGE_URL,
                    polygons: polygons.map(p => ({
                        classId: p.classId,
                        points: p.points.map(pt => ({ x: pt.x, y: pt.y }))
                    })),
                    indices: indices,
                    iterations: iterations,
                    margin_px: margin,
                    smooth_px: isNaN(smooth) ? 0.8 : smooth
                };
                const resp = await fetch('/api/snap_polygons', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!resp.ok) {
                    const errDetail = await resp.json().catch(() => ({}));
                    snapError.textContent = 'Error: ' + (errDetail.detail || resp.statusText);
                    snapError.classList.remove('hidden');
                    return;
                }
                const data = await resp.json();
                const newPolys = [];
                let newSelectedIndex = -1;
                data.polygons.forEach((p, i) => {
                    if (p.points && p.points.length >= 3) {
                        if (i === selectedPolygonIndex) newSelectedIndex = newPolys.length;
                        newPolys.push({ classId: p.classId, points: p.points });
                    }
                });
                polygons = newPolys;
                selectedPolygonIndex = newSelectedIndex;
                saveHistory();
                draw();
            } catch (e) {
                console.error(e);
                snapError.textContent = 'Network error.';
                snapError.classList.remove('hidden');
            } finally {
                snapBtn.disabled = false;
                snapBtn.textContent = origText;
            }
        });
    }

    if (autoSegmentBtn) {
        autoSegmentBtn.addEventListener('click', async () => {
            const modelName = modelSelector.value;
            if (!modelName) return;

            autoSegmentBtn.disabled = true;
            aiStatus.classList.remove('hidden');
            aiError.classList.add('hidden');

            try {
                localStorage.setItem('last_auto_segment_model', modelName);
                
                const payload = {
                    dataset_name: window.DATASET_NAME,
                    image_path: window.IMAGE_URL,
                    model_name: modelName
                };
                
                if (useDenoiseChk && useDenoiseChk.checked) {
                    payload.use_denoise = true;
                    payload.h_lum = parseFloat(dnHLum.value);
                    payload.h_col = parseFloat(dnHCol.value);
                    payload.tw = parseInt(dnTw.value, 10);
                    payload.sw = parseInt(dnSw.value, 10);
                }

                const resp = await fetch('/api/auto_segment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (resp.ok) {
                    const data = await resp.json();
                    if (data.polygons && data.polygons.length > 0) {
                        const imgW = sourceImage.naturalWidth;
                        const imgH = sourceImage.naturalHeight;

                        data.polygons.forEach(p => {
                            const absolutePoly = {
                                classId: p.classId,
                                points: p.points.map(pt => ({ x: pt.x * imgW, y: pt.y * imgH }))
                            };
                            polygons.push(absolutePoly);
                        });
                        saveHistory();
                        draw();
                    } else {
                        aiError.textContent = "No objects detected.";
                        aiError.classList.remove('hidden');
                    }
                } else {
                    const errDetail = await resp.json().catch(() => ({}));
                    aiError.textContent = "Error: " + (errDetail.detail || resp.statusText);
                    aiError.classList.remove('hidden');
                }
            } catch (e) {
                console.error(e);
                aiError.textContent = "Network error connecting to inference server.";
                aiError.classList.remove('hidden');
            } finally {
                autoSegmentBtn.disabled = false;
                aiStatus.classList.add('hidden');
            }
        });
    }

    if (checkDiffBtn) {
        checkDiffBtn.addEventListener('click', async () => {
            const modelName = modelSelector.value;
            if (!modelName) {
                alert("Please select a model first.");
                return;
            }

            const origText = checkDiffBtn.textContent;
            checkDiffBtn.textContent = 'Checking...';
            checkDiffBtn.disabled = true;
            aiStatus.classList.remove('hidden');
            diffResult.classList.add('hidden');

            try {
                localStorage.setItem('last_auto_segment_model', modelName);

                const payload = {
                    dataset_name: window.DATASET_NAME,
                    image_path: window.IMAGE_URL,
                    model_name: modelName
                };
                
                if (useDenoiseChk && useDenoiseChk.checked) {
                    payload.use_denoise = true;
                    payload.h_lum = parseFloat(dnHLum.value);
                    payload.h_col = parseFloat(dnHCol.value);
                    payload.tw = parseInt(dnTw.value, 10);
                    payload.sw = parseInt(dnSw.value, 10);
                }

                const resp = await fetch(`/api/dataset/${window.DATASET_NAME}/auto_check_single`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (resp.ok) {
                    const data = await resp.json();
                    const diffScore = data.diff_score;

                    diffResult.querySelector('.label-model').textContent = `Model: ${modelName}`;
                    const scoreEl = diffResult.querySelector('.label-score');
                    scoreEl.textContent = `Diff Score: ${diffScore.toFixed(4)}`;

                    if (diffScore === 0) {
                        scoreEl.className = 'text-emerald-400 font-bold text-lg label-score block mt-1';
                    } else if (diffScore < 0.2) {
                        scoreEl.className = 'text-amber-400 font-bold text-lg label-score block mt-1';
                    } else {
                        scoreEl.className = 'text-rose-400 font-bold text-lg label-score block mt-1';
                    }

                    diffResult.classList.remove('hidden');
                } else {
                    const errDetail = await resp.json().catch(() => ({}));
                    alert("Error checking diff: " + (errDetail.detail || resp.statusText));
                }
            } finally {
                checkDiffBtn.textContent = origText;
                checkDiffBtn.disabled = false;
                aiStatus.classList.add('hidden');
            }
        });
    }

    // Dataset Split UI
    const splitSelector = document.getElementById('split-selector');
    if (splitSelector) {
        // window.IMAGE_URL = /datasets/dogcat/train/images/000.jpg
        const parts = window.IMAGE_URL.split('/');
        if (parts.length >= 5) {
            splitSelector.value = parts[3];
        }

        splitSelector.addEventListener('change', async (e) => {
            const targetSplit = e.target.value;
            try {
                // Auto-save polygons before moving just in case
                if (polygons.length > 0) {
                    const yoloPolygons = polygons.map(poly => {
                        return {
                            classId: poly.classId,
                            points: poly.points.map(pt => ({
                                x: pt.x / sourceImage.naturalWidth,
                                y: pt.y / sourceImage.naturalHeight
                            }))
                        };
                    });

                    await fetch('/api/save_labels', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            dataset_name: window.DATASET_NAME,
                            label_path: window.LABEL_URL,
                            polygons: yoloPolygons
                        })
                    });
                }

                const resp = await fetch(`/api/dataset/${window.DATASET_NAME}/move_image`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_path: window.IMAGE_URL, target_split: targetSplit })
                });

                if (resp.ok) {
                    const data = await resp.json();
                    const newPathParams = new URLSearchParams(window.location.search);
                    newPathParams.set('img', data.new_image_path);
                    newPathParams.set('lbl', data.new_image_path.replace('/images/', '/labels/').replace(/\.[^/.]+$/, ".txt"));
                    window.location.search = newPathParams.toString();
                } else {
                    alert("Failed to move image");
                    window.location.reload();
                }
            } catch (err) {
                console.error("Move error:", err);
                alert("Network error moving image");
                window.location.reload();
            }
        });
    }

    if (addClassBtn && newClassInput) {
        addClassBtn.addEventListener('click', async () => {
            const className = newClassInput.value.trim();
            if (!className) return;

            try {
                const resp = await fetch('/api/class_manage', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_name: window.DATASET_NAME,
                        action: 'add',
                        class_name: className,
                        class_id: -1,
                        target_class_id: -1
                    })
                });

                if (resp.ok) {
                    const data = await resp.json();
                    const newIndex = data.classes.length - 1;

                    const btn = document.createElement('button');
                    btn.className = 'class-selector w-full flex items-center gap-2 p-2 rounded text-left transition-colors bg-slate-800/50 hover:bg-slate-700 text-slate-300 border border-transparent';
                    btn.dataset.classId = newIndex;
                    btn.innerHTML = `<div class="w-3 h-3 rounded-sm class-color-indicator" style="background-color: ${getClassColor(newIndex)}"></div><span class="text-sm truncate font-medium">${className}</span>`;

                    btn.addEventListener('click', () => {
                        document.querySelectorAll('.class-selector').forEach(b => {
                            b.classList.remove('bg-indigo-500/20', 'border-indigo-500/50', 'text-indigo-300');
                            b.classList.add('bg-slate-800/50', 'text-slate-300', 'border-transparent');
                        });
                        btn.classList.add('bg-indigo-500/20', 'border-indigo-500/50', 'text-indigo-300');
                        btn.classList.remove('bg-slate-800/50', 'text-slate-300', 'border-transparent');
                        activeClassId = newIndex;
                        if (selectedPolygonIndex >= 0) {
                            if (polygons[selectedPolygonIndex].classId !== activeClassId) {
                                polygons[selectedPolygonIndex].classId = activeClassId;
                                draw();
                                saveHistory();
                            }
                        }
                    });

                    document.getElementById('class-list-container').appendChild(btn);
                    newClassInput.value = '';
                    btn.click();
                }
            } catch (e) {
                console.error("Failed to add class dynamically", e);
            }
        });
    }

    // History (Undo/Redo)
    const MAX_HISTORY = 10;
    let history = [];
    let historyIndex = -1;

    function saveHistory() {
        if (historyIndex < history.length - 1) {
            history = history.slice(0, historyIndex + 1);
        }
        history.push(JSON.stringify(polygons));
        if (history.length > MAX_HISTORY + 1) {
            history.shift();
        }
        historyIndex = history.length - 1;
    }

    function undo() {
        if (historyIndex > 0) {
            historyIndex--;
            polygons = JSON.parse(history[historyIndex]);
            selectedPolygonIndex = -1;
            draw();
        }
    }

    function redo() {
        if (historyIndex < history.length - 1) {
            historyIndex++;
            polygons = JSON.parse(history[historyIndex]);
            selectedPolygonIndex = -1;
            draw();
        }
    }

    // Helper: color generator based on ID
    function getClassColor(classId, alpha = 1) {
        return `hsla(${(classId * 137.5) % 360}, 70%, 60%, ${alpha})`;
    }

    // Initialization
    function initCanvas() {
        document.getElementById('loading').style.display = 'none';
        canvas.classList.remove('opacity-0');

        // Setup initial canvas dimensions matching the image aspect ratio
        fitImageToContainer();

        // Fetch existing labels
        fetchLabels();
    }

    if (sourceImage.complete) {
        initCanvas();
    } else {
        sourceImage.onload = initCanvas;
    }

    function fitImageToContainer() {
        const containerRect = container.getBoundingClientRect();

        // Reset scale and offsets. Canvas dimensions are managed inside draw().
        scale = 1;

        // Calculate initial zoom to fit container while preserving aspect ratio
        const scaleX = containerRect.width / sourceImage.width;
        const scaleY = containerRect.height / sourceImage.height;
        scale = Math.min(scaleX, scaleY) * 0.95; // 95% to leave some margin

        // Center the image within the container
        offsetX = (containerRect.width - sourceImage.width * scale) / 2;
        offsetY = (containerRect.height - sourceImage.height * scale) / 2;

        draw();
    }

    window.addEventListener('resize', () => {
        draw();
    });

    // Data Load/Save
    async function fetchLabels() {
        if (!window.LABEL_URL) return;

        try {
            const resp = await fetch(`/api/labels?dataset=${window.DATASET_NAME}&label_path=${encodeURIComponent(window.LABEL_URL)}`);
            if (resp.ok) {
                const data = await resp.json();
                polygons = data.polygons.map(p => ({
                    classId: p.classId,
                    // Convert normalized coordinates back to image pixel coordinates
                    points: p.points.map(pt => ({
                        x: pt.x * sourceImage.width,
                        y: pt.y * sourceImage.height
                    }))
                }));
                draw();
                saveHistory(); // Save initial state
            }
        } catch (e) {
            console.error("Failed to load labels", e);
        }
    }

    window.saveLabels = async function (btnId = 'save-btn') {
        const btn = document.getElementById(btnId);
        if (!btn) return false;
        const originalText = btn.innerText;

        // Handle inner elements if any (like svg/span inside button)
        const originalHTML = btn.innerHTML;
        btn.innerText = "Saving...";
        btn.disabled = true;

        // Convert back to normalized coordinates
        const normalizedPolygons = polygons.filter(p => p.points.length >= 3).map(p => ({
            classId: p.classId,
            points: p.points.map(pt => ({
                x: pt.x / sourceImage.width,
                y: pt.y / sourceImage.height
            }))
        }));

        try {
            const resp = await fetch('/api/save_labels', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_name: window.DATASET_NAME,
                    label_path: window.LABEL_URL,
                    polygons: normalizedPolygons
                })
            });

            if (resp.ok) {
                // Saving non-empty labels means the image is no longer a
                // "no detection target" image — clear the negative flag so
                // the data is consistent.
                const negChkEl = document.getElementById('negative-sample-chk');
                if (negChkEl && negChkEl.checked && normalizedPolygons.length > 0) {
                    try {
                        const filename = (window.IMAGE_URL || '').split('?')[0].split('/').pop();
                        if (filename) {
                            await fetch(`/api/dataset/${encodeURIComponent(window.DATASET_NAME)}/toggle_negative`, {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ image_filename: decodeURIComponent(filename), value: false })
                            });
                            negChkEl.checked = false;
                            const ns = document.getElementById('negative-sample-status');
                            if (ns) ns.classList.add('hidden');
                        }
                    } catch (_) { /* non-fatal */ }
                }
                btn.classList.remove('bg-indigo-600', 'bg-emerald-600');
                btn.classList.add('bg-emerald-600');
                btn.innerText = "Saved!";
                setTimeout(() => {
                    btn.innerHTML = originalHTML;
                    btn.disabled = false;
                    btn.classList.remove('bg-emerald-600', 'bg-rose-600');
                    if (btnId === 'save-btn') btn.classList.add('bg-indigo-600');
                    else btn.classList.add('bg-emerald-600');
                }, 2000);
                return true;
            } else {
                throw new Error("Save failed");
            }
        } catch (e) {
            btn.classList.remove('bg-indigo-600', 'bg-emerald-600');
            btn.classList.add('bg-rose-600');
            btn.innerText = "Error";
            console.error(e);
            setTimeout(() => {
                btn.innerHTML = originalHTML;
                btn.disabled = false;
                btn.classList.remove('bg-emerald-600', 'bg-rose-600');
                if (btnId === 'save-btn') btn.classList.add('bg-indigo-600');
                else btn.classList.add('bg-emerald-600');
            }, 2000);
            return false;
        }
    }

    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) saveBtn.addEventListener('click', () => window.saveLabels('save-btn'));

    const saveNextBtn = document.getElementById('save-next-btn');
    if (saveNextBtn) saveNextBtn.addEventListener('click', async () => {
        const success = await window.saveLabels('save-next-btn');
        if (success) {
            try {
                const resp = await fetch(`/api/dataset/${window.DATASET_NAME}/next_unlabeled`);
                const data = await resp.json();
                if (data.status === 'ok') {
                    const params = new URLSearchParams(window.location.search);
                    params.set('img', data.next_image);
                    params.set('lbl', data.next_label);
                    window.location.search = params.toString();
                } else {
                    alert('No more unlabeled images found in the entire dataset!');
                }
            } catch (e) {
                console.error(e);
                alert("Error connecting to server to find next image.");
            }
        }
    });

    const prevImgBtn = document.getElementById('prev-img-btn');
    if (prevImgBtn) prevImgBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch(`/api/dataset/${window.DATASET_NAME}/prev_image?current_image=${encodeURIComponent(window.IMAGE_URL)}`);
            const data = await resp.json();
            if (data.status === 'ok') {
                const params = new URLSearchParams(window.location.search);
                params.set('img', data.prev.image_url);
                params.set('lbl', data.prev.label_url);
                window.location.search = params.toString();
            } else {
                alert('No previous image found.');
            }
        } catch (e) {
            console.error(e);
            alert("Error connecting to server to find previous image.");
        }
    });

    const nextImgBtn = document.getElementById('next-img-btn');
    if (nextImgBtn) nextImgBtn.addEventListener('click', async () => {
        try {
            const resp = await fetch(`/api/dataset/${window.DATASET_NAME}/next_image?current_image=${encodeURIComponent(window.IMAGE_URL)}`);
            const data = await resp.json();
            if (data.status === 'ok') {
                const params = new URLSearchParams(window.location.search);
                params.set('img', data.next.image_url);
                params.set('lbl', data.next.label_url);
                window.location.search = params.toString();
            } else {
                alert('No next image found.');
            }
        } catch (e) {
            console.error(e);
            alert("Error connecting to server to find next image.");
        }
    });

    const deleteImgBtn = document.getElementById('delete-image-btn');
    if (deleteImgBtn) deleteImgBtn.addEventListener('click', async () => {
        if (!confirm("Are you sure you want to permanently delete this image and its labels from the dataset?")) {
            return;
        }

        const originalText = deleteImgBtn.innerHTML;
        deleteImgBtn.innerHTML = "Deleting...";
        deleteImgBtn.disabled = true;

        try {
            // Find next image URL before deleting
            const nextResp = await fetch(`/api/dataset/${window.DATASET_NAME}/next_image?current_image=${encodeURIComponent(window.IMAGE_URL)}`);
            const nextData = await nextResp.json();
            let nextUrl = null;
            if (nextResp.ok && nextData.status === 'ok' && nextData.next.image_url !== window.IMAGE_URL) {
                const params = new URLSearchParams(window.location.search);
                params.set('img', nextData.next.image_url);
                params.set('lbl', nextData.next.label_url);
                nextUrl = "?" + params.toString();
            }

            // Perform delete
            const delResp = await fetch(`/api/dataset/${window.DATASET_NAME}/delete_image`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_path: window.IMAGE_URL })
            });

            if (delResp.ok) {
                // Navigate away
                if (nextUrl) {
                    window.location.search = nextUrl;
                } else {
                    // No other images left
                    window.location.href = `/dataset/${window.DATASET_NAME}`;
                }
            } else {
                const err = await delResp.json().catch(() => ({}));
                alert(err.detail || "Failed to delete image");
                deleteImgBtn.innerHTML = originalText;
                deleteImgBtn.disabled = false;
            }
        } catch (e) {
            console.error(e);
            alert("Network error while deleting image");
            deleteImgBtn.innerHTML = originalText;
            deleteImgBtn.disabled = false;
        }
    });

    // --- Negative Sample Toggle ---
    // Per-image flag stored at datasets/<name>/negative_samples.json (outside
    // train/valid/test). Marking an image as a negative sample says "I am
    // deliberately leaving this empty", which lets the gallery's "Unlabeled
    // (needs work)" filter exclude it, and Save&Next skip it.
    const negChk = document.getElementById('negative-sample-chk');
    const negStatus = document.getElementById('negative-sample-status');

    function imageFilenameFromUrl(url) {
        try {
            const path = url.split('?')[0];
            const idx = path.lastIndexOf('/');
            return idx >= 0 ? decodeURIComponent(path.slice(idx + 1)) : decodeURIComponent(path);
        } catch (e) {
            return null;
        }
    }
    const currentImageFilename = imageFilenameFromUrl(window.IMAGE_URL);

    function setNegativeUI(isNegative) {
        if (!negChk) return;
        negChk.checked = !!isNegative;
        if (negStatus) negStatus.classList.toggle('hidden', !isNegative);
    }

    async function fetchNegativeState() {
        if (!negChk || !currentImageFilename) return;
        try {
            const resp = await fetch(`/api/dataset/${encodeURIComponent(window.DATASET_NAME)}/negative_samples`);
            if (!resp.ok) return;
            const data = await resp.json();
            setNegativeUI((data.items || []).includes(currentImageFilename));
        } catch (e) {
            console.error('Failed to fetch negative-sample state', e);
        }
    }

    async function postNegativeToggle(value) {
        const resp = await fetch(`/api/dataset/${encodeURIComponent(window.DATASET_NAME)}/toggle_negative`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_filename: currentImageFilename, value })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'Toggle failed');
        }
    }

    if (negChk && currentImageFilename) {
        negChk.addEventListener('change', async () => {
            const wantOn = negChk.checked;
            const hadPolygons = polygons.length > 0;
            if (wantOn && hadPolygons) {
                const ok = confirm("Mark as negative sample?\n\nThis will clear all existing polygons on this image. Proceed?");
                if (!ok) {
                    negChk.checked = false;
                    return;
                }
                polygons = [];
                draw();
                saveHistory();
                // Persist the empty label so the file system reflects "no labels"
                // immediately; otherwise the gallery would still show the old
                // labels until the user hits Save.
                try {
                    await fetch('/api/save_labels', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            dataset_name: window.DATASET_NAME,
                            label_path: window.LABEL_URL,
                            polygons: []
                        })
                    });
                } catch (_) { /* non-fatal; user can hit Save later */ }
            }
            try {
                await postNegativeToggle(wantOn);
                setNegativeUI(wantOn);
            } catch (e) {
                console.error(e);
                alert('Failed to update negative-sample flag.');
                negChk.checked = !wantOn;
            }
        });
        fetchNegativeState();
    }

    // Coordinate Conversion (returns coordinates in image-pixel space, which is
    // also the polygon coordinate space; independent of RESOLUTION_MULTIPLIER).
    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return { x: 0, y: 0 };
        const x = (evt.clientX - rect.left) / rect.width * sourceImage.width;
        const y = (evt.clientY - rect.top) / rect.height * sourceImage.height;
        return { x, y };
    }

    // Interaction Handlers
    container.addEventListener('wheel', (e) => {
        e.preventDefault();

        const cvsRect = canvas.getBoundingClientRect();
        if (cvsRect.width === 0) return;

        const ptX = (e.clientX - cvsRect.left) / cvsRect.width * sourceImage.width;
        const ptY = (e.clientY - cvsRect.top) / cvsRect.height * sourceImage.height;

        const zoomIntensity = 0.1;
        const wheel = e.deltaY < 0 ? 1 : -1;
        const zoomFactor = Math.exp(wheel * zoomIntensity);
        scale *= zoomFactor;
        scale = Math.max(0.1, Math.min(scale, 15));

        const contRect = container.getBoundingClientRect();
        offsetX = (e.clientX - contRect.left) - ptX * scale;
        offsetY = (e.clientY - contRect.top) - ptY * scale;

        draw();
    });

    container.addEventListener('mousedown', (e) => {
        if (e.button === 1 || e.button === 2) { // Middle or Right click
            if (e.button === 1) { // Middle click: Pan
                isPanning = true;
                const contRect = container.getBoundingClientRect();
                startPanX = (e.clientX - contRect.left) - offsetX;
                startPanY = (e.clientY - contRect.top) - offsetY;
                container.style.cursor = 'grabbing';
            } else if (e.button === 2) { // Right click: Close polygon or delete point
                if (currentPolygon) {
                    if (currentPolygon.points.length > 2) {
                        polygons.push(currentPolygon);
                        saveHistory();
                    }
                    currentPolygon = null;
                    draw();
                } else if (hoveredVertexIndex !== -1 && selectedPolygonIndex !== -1) {
                    polygons[selectedPolygonIndex].points.splice(hoveredVertexIndex, 1);
                    if (polygons[selectedPolygonIndex].points.length < 3) {
                        polygons.splice(selectedPolygonIndex, 1);
                        selectedPolygonIndex = -1;
                    }
                    saveHistory();
                    hoveredVertexIndex = -1;
                    draw();
                }
            }
            e.preventDefault();
            return;
        }

        // Left click
        if (e.button === 0) {
            const pos = getMousePos(e);

            if (!currentPolygon) {
                if (hoveredVertexIndex !== -1 && selectedPolygonIndex !== -1) {
                    isDraggingVertex = true;
                    return;
                }

                // Check if we clicked on an existing polygon to select it
                const clickedIdx = findPolygonAtPos(pos);
                if (clickedIdx !== -1) {
                    selectedPolygonIndex = clickedIdx;

                    const clickedClass = polygons[clickedIdx].classId;
                    const btn = document.querySelector(`.class-selector[data-class-id="${clickedClass}"]`);
                    if (btn) btn.click();

                    draw();
                    return;
                }

                selectedPolygonIndex = -1;
                // Start new polygon
                currentPolygon = { classId: activeClassId, points: [pos] };
            } else {
                // Add point
                currentPolygon.points.push(pos);
            }
            draw();
        }
    });

    container.addEventListener('mousemove', (e) => {
        if (isPanning) {
            const contRect = container.getBoundingClientRect();
            offsetX = (e.clientX - contRect.left) - startPanX;
            offsetY = (e.clientY - contRect.top) - startPanY;
            draw();
            return;
        }

        const pos = getMousePos(e);

        if (isDraggingVertex && selectedPolygonIndex !== -1 && hoveredVertexIndex !== -1) {
            polygons[selectedPolygonIndex].points[hoveredVertexIndex] = pos;
            draw();
            return;
        }

        if (!currentPolygon) {
            hoveredVertexIndex = -1;
            if (selectedPolygonIndex !== -1) {
                const poly = polygons[selectedPolygonIndex];
                const threshold = 8 / scale;
                for (let i = 0; i < poly.points.length; i++) {
                    const dx = poly.points[i].x - pos.x;
                    const dy = poly.points[i].y - pos.y;
                    if (Math.hypot(dx, dy) < threshold) {
                        hoveredVertexIndex = i;
                        break;
                    }
                }
            }

            if (hoveredVertexIndex !== -1) {
                container.style.cursor = 'crosshair';
                hoveredPolygonIndex = -1;
            } else {
                const hIdx = findPolygonAtPos(pos);
                if (hIdx !== hoveredPolygonIndex) {
                    hoveredPolygonIndex = hIdx;
                }
                container.style.cursor = hIdx !== -1 ? 'pointer' : 'crosshair';
            }
            draw();
        } else {
            draw(pos);
        }
    });

    container.addEventListener('mouseup', (e) => {
        if (e.button === 1) {
            isPanning = false;
            container.style.cursor = currentPolygon ? 'crosshair' : (hoveredPolygonIndex !== -1 ? 'pointer' : 'crosshair');
        } else if (e.button === 0) {
            if (isDraggingVertex) {
                isDraggingVertex = false;
                saveHistory();
            }
        }
    });

    container.addEventListener('contextmenu', e => e.preventDefault());

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            currentPolygon = null;
            selectedPolygonIndex = -1;
            draw();
        } else if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedPolygonIndex >= 0) {
                if (hoveredVertexIndex !== -1) {
                    polygons[selectedPolygonIndex].points.splice(hoveredVertexIndex, 1);
                    if (polygons[selectedPolygonIndex].points.length < 3) {
                        polygons.splice(selectedPolygonIndex, 1);
                        selectedPolygonIndex = -1;
                    }
                    hoveredVertexIndex = -1;
                } else {
                    polygons.splice(selectedPolygonIndex, 1);
                    selectedPolygonIndex = -1;
                    hoveredPolygonIndex = -1;
                }
                saveHistory();
                draw();
            }
        } else if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
            if (e.shiftKey) {
                redo();
            } else {
                undo();
            }
            e.preventDefault();
        } else if (e.key === 'y' && (e.ctrlKey || e.metaKey)) {
            redo();
            e.preventDefault();
        }
    });

    // Point-in-polygon check for selection
    function findPolygonAtPos(pos) {
        // Iterate backwards to select top-most polygon
        for (let i = polygons.length - 1; i >= 0; i--) {
            if (isPointInPolygon(pos, polygons[i].points)) {
                return i;
            }
        }
        return -1;
    }

    function isPointInPolygon(point, vs) {
        // Ray casting algorithm
        let x = point.x, y = point.y;
        let inside = false;
        for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
            let xi = vs[i].x, yi = vs[i].y;
            let xj = vs[j].x, yj = vs[j].y;

            let intersect = ((yi > y) != (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    // Drawing
    function draw(mousePos = null) {
        const imgW = sourceImage.width;
        const imgH = sourceImage.height;
        const internalW = imgW * RESOLUTION_MULTIPLIER;
        const internalH = imgH * RESOLUTION_MULTIPLIER;

        // Backing buffer is RESOLUTION_MULTIPLIER times the image's natural size;
        // CSS size still matches natural size so layout/transform math is unchanged.
        if (canvas.width !== internalW || canvas.height !== internalH) {
            canvas.width = internalW;
            canvas.height = internalH;
        }

        canvas.style.width = imgW + 'px';
        canvas.style.height = imgH + 'px';
        canvas.style.position = 'absolute';
        canvas.style.left = '0';
        canvas.style.top = '0';
        canvas.style.transformOrigin = '0 0';
        canvas.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

        // Map drawing coordinates back to image-pixel space so polygon math and
        // line widths stay identical to the non-supersampled implementation.
        ctx.setTransform(RESOLUTION_MULTIPLIER, 0, 0, RESOLUTION_MULTIPLIER, 0, 0);
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = 'high';

        ctx.clearRect(0, 0, imgW, imgH);

        // Draw source image
        ctx.globalAlpha = 1.0;
        ctx.drawImage(sourceImage, 0, 0);

        // Draw saved polygons
        polygons.forEach((poly, idx) => {
            const isSelected = idx === selectedPolygonIndex;
            const isHovered = idx === hoveredPolygonIndex;

            ctx.beginPath();
            poly.points.forEach((p, i) => {
                if (i === 0) ctx.moveTo(p.x, p.y);
                else ctx.lineTo(p.x, p.y);
            });
            ctx.closePath();

            // Fill
            ctx.globalAlpha = isSelected ? 0.6 : (isHovered ? 0.4 : 0.25);
            ctx.fillStyle = getClassColor(poly.classId);
            ctx.fill();

            // Stroke
            ctx.globalAlpha = isSelected ? 1.0 : 0.8;
            ctx.lineWidth = isSelected ? 3.0 / scale : 1.5 / scale;
            ctx.strokeStyle = isSelected ? '#ffffff' : getClassColor(poly.classId);
            ctx.stroke();

            // Draw points if selected
            if (isSelected) {
                poly.points.forEach((p, i) => {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, (i === hoveredVertexIndex ? 6 : 4) / scale, 0, Math.PI * 2);
                    ctx.fillStyle = (i === hoveredVertexIndex ? '#ef4444' : '#ffffff');
                    ctx.fill();
                    if (i === hoveredVertexIndex) {
                        ctx.lineWidth = 2 / scale;
                        ctx.strokeStyle = '#ffffff';
                        ctx.stroke();
                    }
                });
            }
        });

        // Draw current polygon in progress
        if (currentPolygon && currentPolygon.points.length > 0) {
            ctx.beginPath();
            currentPolygon.points.forEach((p, i) => {
                if (i === 0) ctx.moveTo(p.x, p.y);
                else ctx.lineTo(p.x, p.y);
            });

            if (mousePos) {
                ctx.lineTo(mousePos.x, mousePos.y);
            }

            ctx.globalAlpha = 0.4;
            ctx.fillStyle = getClassColor(activeClassId);
            ctx.fill();

            ctx.globalAlpha = 1.0;
            ctx.lineWidth = 1.5 / scale;
            ctx.strokeStyle = getClassColor(activeClassId);
            ctx.stroke();

            // Draw points
            ctx.fillStyle = '#ffffff';
            currentPolygon.points.forEach(p => {
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3 / scale, 0, Math.PI * 2);
                ctx.fill();
            });

            if (mousePos) {
                ctx.beginPath();
                ctx.arc(mousePos.x, mousePos.y, 3 / scale, 0, Math.PI * 2);
                ctx.fill();
            }
        }
    }
});
