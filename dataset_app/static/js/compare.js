// Compare two datasets: fetch /api/compare and render summary + diff lists
// plus a side-by-side image+polygon viewer for label-different images.

(function () {
    const selectA = document.getElementById('dataset-a');
    const selectB = document.getElementById('dataset-b');
    const compareBtn = document.getElementById('compare-btn');
    const compareBtnText = document.getElementById('compare-btn-text');
    const loading = document.getElementById('loading');
    const summary = document.getElementById('summary');
    const emptyResult = document.getElementById('empty-result');
    const classDiffPanel = document.getElementById('class-diff-panel');
    const classDiffEl = document.getElementById('class-diff');
    const diffPanel = document.getElementById('diff-panel');
    const diffList = document.getElementById('diff-list');
    const diffCount = document.getElementById('diff-count');
    const diffFilter = document.getElementById('diff-filter');
    const onlyAPanel = document.getElementById('only-a-panel');
    const onlyBPanel = document.getElementById('only-b-panel');
    const onlyAList = document.getElementById('only-a-list');
    const onlyBList = document.getElementById('only-b-list');
    const onlyACount = document.getElementById('only-a-count');
    const onlyBCount = document.getElementById('only-b-count');
    const onlyATitle = document.getElementById('only-a-title');
    const onlyBTitle = document.getElementById('only-b-title');

    const viewer = document.getElementById('viewer');
    const viewerStem = document.getElementById('viewer-stem');
    const viewerAName = document.getElementById('viewer-a-name');
    const viewerBName = document.getElementById('viewer-b-name');
    const viewerAEdit = document.getElementById('viewer-a-edit');
    const viewerBEdit = document.getElementById('viewer-b-edit');
    const canvasA = document.getElementById('canvas-a');
    const canvasB = document.getElementById('canvas-b');
    const legendA = document.getElementById('legend-a');
    const legendB = document.getElementById('legend-b');
    const overlayBoth = document.getElementById('overlay-both');
    const viewerClose = document.getElementById('viewer-close');

    let lastResult = null;
    let currentDiffEntry = null;

    // Match editor.js color scheme so visual identity stays consistent.
    function getClassColor(classId, alpha = 1) {
        return `hsla(${(classId * 137.5) % 360}, 70%, 60%, ${alpha})`;
    }

    function parseYoloLabel(text) {
        const polys = [];
        if (!text) return polys;
        for (const raw of text.split(/\r?\n/)) {
            const parts = raw.trim().split(/\s+/);
            if (parts.length < 7 || parts.length % 2 !== 1) continue;
            const classId = parseInt(parts[0], 10);
            if (Number.isNaN(classId)) continue;
            const coords = parts.slice(1).map(Number);
            const points = [];
            for (let i = 0; i < coords.length; i += 2) {
                points.push({ x: coords[i], y: coords[i + 1] });
            }
            polys.push({ classId, points });
        }
        return polys;
    }

    async function fetchLabel(labelUrl) {
        try {
            const resp = await fetch(labelUrl);
            if (!resp.ok) return [];
            return parseYoloLabel(await resp.text());
        } catch (e) {
            return [];
        }
    }

    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = url;
        });
    }

    function drawScene(canvas, image, polygons, classes, opts) {
        // Cap canvas backing buffer at 1600px on the long edge so 4K source
        // images don't blow up GPU memory; CSS still stretches to its box.
        const MAX_EDGE = 1600;
        const longEdge = Math.max(image.naturalWidth, image.naturalHeight);
        const scale = longEdge > MAX_EDGE ? MAX_EDGE / longEdge : 1;
        const W = Math.round(image.naturalWidth * scale);
        const H = Math.round(image.naturalHeight * scale);
        canvas.width = W;
        canvas.height = H;
        canvas.style.aspectRatio = `${image.naturalWidth} / ${image.naturalHeight}`;
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, W, H);
        ctx.drawImage(image, 0, 0, W, H);

        const fillAlpha = (opts && opts.fillAlpha) ?? 0.25;
        const strokeAlpha = (opts && opts.strokeAlpha) ?? 1;
        const lineWidth = (opts && opts.lineWidth) ?? Math.max(2, Math.round(longEdge / 600));

        for (const poly of polygons) {
            if (poly.points.length < 2) continue;
            ctx.beginPath();
            poly.points.forEach((pt, i) => {
                const x = pt.x * W;
                const y = pt.y * H;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            });
            ctx.closePath();
            ctx.fillStyle = getClassColor(poly.classId, fillAlpha);
            ctx.fill();
            ctx.strokeStyle = (opts && opts.strokeOverride) || getClassColor(poly.classId, strokeAlpha);
            ctx.lineWidth = lineWidth;
            ctx.stroke();
        }
    }

    function renderLegend(container, polygons, classes) {
        const used = new Map();
        for (const p of polygons) {
            used.set(p.classId, (used.get(p.classId) || 0) + 1);
        }
        if (used.size === 0) {
            container.innerHTML = '<span class="text-slate-500 italic">no polygons</span>';
            return;
        }
        const items = Array.from(used.entries())
            .sort((a, b) => a[0] - b[0])
            .map(([cid, count]) => {
                const name = classes[cid] != null ? classes[cid] : `class_${cid}`;
                return `<span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-800 border border-slate-700">
                    <span class="w-2.5 h-2.5 rounded-sm" style="background-color: ${getClassColor(cid)}"></span>
                    <span class="text-slate-300">${escapeHtml(name)}</span>
                    <span class="text-slate-500">×${count}</span>
                </span>`;
            });
        container.innerHTML = items.join('');
    }

    function escapeHtml(s) {
        return String(s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function editorUrl(dataset, imageUrl, labelUrl) {
        return `/editor/${encodeURIComponent(dataset)}?img=${encodeURIComponent(imageUrl)}&lbl=${encodeURIComponent(labelUrl)}`;
    }

    function makeThumbCard(entry, opts) {
        const onClick = opts && opts.onClick;
        const subtitleHtml = opts && opts.subtitle ? `<div class="text-[10px] text-slate-400">${opts.subtitle}</div>` : '';
        const card = document.createElement(onClick ? 'button' : 'a');
        if (onClick) {
            card.type = 'button';
            card.addEventListener('click', onClick);
        } else {
            card.href = editorUrl(entry.dataset, entry.image_url, entry.label_url);
            card.target = '_blank';
        }
        card.className = 'group block bg-slate-900/60 border border-slate-700 hover:border-indigo-400 rounded overflow-hidden text-left transition-colors';
        card.innerHTML = `
            <div class="relative bg-slate-800" style="aspect-ratio: 1 / 1;">
                <img loading="lazy" src="${entry.image_url}" alt="" class="absolute inset-0 w-full h-full object-contain">
                <span class="absolute top-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-slate-900/80 text-slate-300 uppercase tracking-wider">${entry.split}</span>
                ${entry.has_label ? '' : '<span class="absolute bottom-1 left-1 text-[10px] px-1.5 py-0.5 rounded bg-amber-500/90 text-slate-900 font-semibold">no label</span>'}
            </div>
            <div class="p-2">
                <div class="font-mono text-[10px] text-slate-300 truncate" title="${escapeHtml(entry.stem)}">${escapeHtml(entry.stem)}</div>
                ${subtitleHtml}
            </div>
        `;
        return card;
    }

    function renderOnlyList(listEl, items) {
        listEl.innerHTML = '';
        for (const it of items) {
            listEl.appendChild(makeThumbCard(it));
        }
    }

    function renderDiffList(items) {
        diffList.innerHTML = '';
        const filterText = diffFilter.value.trim().toLowerCase();
        let shown = 0;
        for (const it of items) {
            if (filterText && !it.stem.toLowerCase().includes(filterText)) continue;
            shown++;
            const subtitle = `${it.a.split} → ${it.b.split}`;
            const card = makeThumbCard(it.a, {
                subtitle,
                onClick: () => openViewer(it),
            });
            diffList.appendChild(card);
        }
        if (shown === 0) {
            diffList.innerHTML = '<div class="col-span-full text-slate-500 text-sm py-6 text-center">No matches.</div>';
        }
    }

    function renderClassDiff(classesA, classesB, nameA, nameB) {
        const setA = new Set(classesA);
        const setB = new Set(classesB);
        const onlyA = classesA.filter(c => !setB.has(c));
        const onlyB = classesB.filter(c => !setA.has(c));
        if (onlyA.length === 0 && onlyB.length === 0 && classesA.length === classesB.length
            && classesA.every((c, i) => c === classesB[i])) {
            classDiffPanel.classList.add('hidden');
            return;
        }
        classDiffPanel.classList.remove('hidden');
        const fmt = (names, label, color, otherSet) => `
            <div>
                <div class="text-xs uppercase tracking-wider text-slate-400 mb-2">${escapeHtml(label)}</div>
                <ol class="space-y-1 list-decimal list-inside font-mono text-xs">
                    ${names.map((n) => {
                        const inOther = otherSet.has(n);
                        const cls = inOther ? 'text-slate-300' : `${color} font-semibold`;
                        return `<li class="${cls}">${escapeHtml(n)}</li>`;
                    }).join('')}
                </ol>
            </div>
        `;
        classDiffEl.innerHTML =
            fmt(classesA, `${nameA} (A)`, 'text-amber-300', setB) +
            fmt(classesB, `${nameB} (B)`, 'text-amber-300', setA);
    }

    async function openViewer(entry) {
        currentDiffEntry = entry;
        viewer.classList.remove('hidden');
        viewerStem.textContent = entry.stem;
        viewerAName.textContent = `${entry.a.dataset} / ${entry.a.split}`;
        viewerBName.textContent = `${entry.b.dataset} / ${entry.b.split}`;
        viewerAEdit.href = editorUrl(entry.a.dataset, entry.a.image_url, entry.a.label_url);
        viewerBEdit.href = editorUrl(entry.b.dataset, entry.b.image_url, entry.b.label_url);
        viewer.scrollIntoView({ behavior: 'smooth', block: 'start' });
        await renderViewer();
    }

    async function renderViewer() {
        if (!currentDiffEntry) return;
        const entry = currentDiffEntry;
        const [imgA, imgB, polyA, polyB] = await Promise.all([
            loadImage(entry.a.image_url),
            loadImage(entry.b.image_url),
            fetchLabel(entry.a.label_url),
            fetchLabel(entry.b.label_url),
        ]);
        const classesA = lastResult ? lastResult.classes_a : [];
        const classesB = lastResult ? lastResult.classes_b : [];

        if (overlayBoth.checked) {
            // Draw image once, then layer A (solid) and B (dashed) on top so
            // shape divergence is immediately obvious without eye-jumping.
            const canvas = canvasA;
            const longEdge = Math.max(imgA.naturalWidth, imgA.naturalHeight);
            drawScene(canvas, imgA, [], classesA);
            const ctx = canvas.getContext('2d');
            const W = canvas.width, H = canvas.height;
            const drawSet = (polys, dashed) => {
                const lw = Math.max(2, Math.round(longEdge / 600));
                for (const poly of polys) {
                    if (poly.points.length < 2) continue;
                    ctx.beginPath();
                    poly.points.forEach((pt, i) => {
                        const x = pt.x * W, y = pt.y * H;
                        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                    });
                    ctx.closePath();
                    ctx.fillStyle = getClassColor(poly.classId, 0.15);
                    ctx.fill();
                    ctx.setLineDash(dashed ? [lw * 3, lw * 2] : []);
                    ctx.strokeStyle = getClassColor(poly.classId, 1);
                    ctx.lineWidth = lw;
                    ctx.stroke();
                }
                ctx.setLineDash([]);
            };
            drawSet(polyA, false);
            drawSet(polyB, true);
            // The right canvas mirrors the left so the labeled column hint
            // (A=solid, B=dashed) still has a place to show in the legend.
            drawScene(canvasB, imgB, [], classesB);
            renderLegend(legendA, polyA, classesA);
            renderLegend(legendB, polyB, classesB);
        } else {
            drawScene(canvasA, imgA, polyA, classesA);
            drawScene(canvasB, imgB, polyB, classesB);
            renderLegend(legendA, polyA, classesA);
            renderLegend(legendB, polyB, classesB);
        }
    }

    function closeViewer() {
        currentDiffEntry = null;
        viewer.classList.add('hidden');
    }

    function clearResults() {
        summary.classList.add('hidden');
        emptyResult.classList.add('hidden');
        classDiffPanel.classList.add('hidden');
        diffPanel.classList.add('hidden');
        onlyAPanel.classList.add('hidden');
        onlyBPanel.classList.add('hidden');
        closeViewer();
    }

    async function runCompare() {
        const a = selectA.value;
        const b = selectB.value;
        if (!a || !b) {
            alert('Pick two datasets.');
            return;
        }
        if (a === b) {
            alert('Pick two different datasets.');
            return;
        }
        const url = new URL(window.location.href);
        url.searchParams.set('a', a);
        url.searchParams.set('b', b);
        window.history.replaceState({}, '', url);

        clearResults();
        loading.classList.remove('hidden');
        compareBtn.disabled = true;
        compareBtnText.textContent = 'Comparing…';
        try {
            const resp = await fetch(`/api/compare?dataset_a=${encodeURIComponent(a)}&dataset_b=${encodeURIComponent(b)}`);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                alert(err.detail || 'Compare failed');
                return;
            }
            const data = await resp.json();
            lastResult = data;
            renderResult(data);
        } catch (e) {
            alert('Network error');
        } finally {
            loading.classList.add('hidden');
            compareBtn.disabled = false;
            compareBtnText.textContent = 'Compare';
        }
    }

    function renderResult(data) {
        document.getElementById('stat-a').textContent = data.counts.a_total;
        document.getElementById('stat-b').textContent = data.counts.b_total;
        document.getElementById('stat-only-a').textContent = data.counts.only_a;
        document.getElementById('stat-only-b').textContent = data.counts.only_b;
        document.getElementById('stat-label-diff').textContent = data.counts.label_diff;
        summary.classList.remove('hidden');

        renderClassDiff(data.classes_a, data.classes_b, data.dataset_a, data.dataset_b);

        if (data.counts.label_diff > 0) {
            diffPanel.classList.remove('hidden');
            diffCount.textContent = `(${data.counts.label_diff})`;
            renderDiffList(data.label_diff);
        }
        if (data.counts.only_a > 0) {
            onlyAPanel.classList.remove('hidden');
            onlyATitle.textContent = data.dataset_a;
            onlyACount.textContent = `(${data.counts.only_a})`;
            renderOnlyList(onlyAList, data.only_a);
        }
        if (data.counts.only_b > 0) {
            onlyBPanel.classList.remove('hidden');
            onlyBTitle.textContent = data.dataset_b;
            onlyBCount.textContent = `(${data.counts.only_b})`;
            renderOnlyList(onlyBList, data.only_b);
        }
        if (data.counts.only_a === 0 && data.counts.only_b === 0 && data.counts.label_diff === 0) {
            emptyResult.classList.remove('hidden');
        }
    }

    compareBtn.addEventListener('click', runCompare);
    viewerClose.addEventListener('click', closeViewer);
    overlayBoth.addEventListener('change', renderViewer);
    diffFilter.addEventListener('input', () => {
        if (lastResult) renderDiffList(lastResult.label_diff);
    });

    // Auto-run when both selectors are pre-populated via ?a=&b= URL params.
    if (selectA.value && selectB.value && selectA.value !== selectB.value) {
        runCompare();
    }
})();
