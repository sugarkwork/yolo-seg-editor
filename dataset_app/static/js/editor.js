document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById('seg-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('editor-container');
    const sourceImage = document.getElementById('source-image');

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

    if (autoSegmentBtn) {
        autoSegmentBtn.addEventListener('click', async () => {
            const modelName = modelSelector.value;
            if (!modelName) return;

            autoSegmentBtn.disabled = true;
            aiStatus.classList.remove('hidden');
            aiError.classList.add('hidden');

            try {
                const payload = {
                    dataset_name: window.DATASET_NAME,
                    image_path: window.IMAGE_URL,
                    model_name: modelName
                };

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
                    btn.className = 'class-selector w-full text-left px-3 py-2 rounded text-sm transition-colors flex items-center gap-3 bg-slate-800/50 text-slate-300 border border-transparent';
                    btn.dataset.classId = newIndex;
                    btn.innerHTML = `<span class="w-2.5 h-2.5 rounded-sm shadow-sm" style="background-color: ${getClassColor(newIndex)}"></span>${className}`;

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

        // Reset scale and offsets
        scale = 1;

        canvas.width = sourceImage.width;
        canvas.height = sourceImage.height;

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

    document.getElementById('save-btn').addEventListener('click', async () => {
        const btn = document.getElementById('save-btn');
        const originalText = btn.innerText;
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
                btn.classList.replace('bg-indigo-600', 'bg-emerald-600');
                btn.innerText = "Saved!";
            } else {
                throw new Error("Save failed");
            }
        } catch (e) {
            btn.classList.replace('bg-indigo-600', 'bg-rose-600');
            btn.innerText = "Error";
            console.error(e);
        }

        setTimeout(() => {
            btn.innerText = originalText;
            btn.disabled = false;
            btn.classList.remove('bg-emerald-600', 'bg-rose-600');
            btn.classList.add('bg-indigo-600');
        }, 2000);
    });

    // Coordinate Conversion
    function getMousePos(evt) {
        const rect = canvas.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return { x: 0, y: 0 };
        const x = (evt.clientX - rect.left) / rect.width * canvas.width;
        const y = (evt.clientY - rect.top) / rect.height * canvas.height;
        return { x, y };
    }

    // Interaction Handlers
    container.addEventListener('wheel', (e) => {
        e.preventDefault();

        const cvsRect = canvas.getBoundingClientRect();
        if (cvsRect.width === 0) return;

        const ptX = (e.clientX - cvsRect.left) / cvsRect.width * canvas.width;
        const ptY = (e.clientY - cvsRect.top) / cvsRect.height * canvas.height;

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
        // Only adjust canvas width/height if it changed
        if (canvas.width !== sourceImage.width || canvas.height !== sourceImage.height) {
            canvas.width = sourceImage.width;
            canvas.height = sourceImage.height;
        }

        // Instead of forcing 100% width/height (which breaks aspect ratio),
        // we set explicit pixel sizes matching the logical width/height to CSS width/height
        // and then let the CSS transform scale and position it.
        canvas.style.width = sourceImage.width + 'px';
        canvas.style.height = sourceImage.height + 'px';
        canvas.style.position = 'absolute';
        canvas.style.left = '0';
        canvas.style.top = '0';

        // Actual internal resolution matches the image
        // Then we use CSS transforms to position it
        canvas.style.transformOrigin = '0 0';
        canvas.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

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
