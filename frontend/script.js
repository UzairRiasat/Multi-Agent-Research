const backendUrl = 'http://127.0.0.1:8000';
const queryInput       = document.getElementById('queryInput');
const researchBtn      = document.getElementById('researchBtn');
const reportContainer  = document.getElementById('reportContainer');
const reportContent    = document.getElementById('reportContent');
const copyBtn          = document.getElementById('copyReportBtn');
const downloadBtn      = document.getElementById('downloadReportBtn');
const pipelineTracker  = document.getElementById('pipelineTracker');
const pipelineSubLabel = document.getElementById('pipelineSubLabel');

let currentRawMarkdown = '';
let streamingActive    = false;

// Auto-scroll state
// userScrolledUp: set to true the moment the user scrolls up during streaming,
// which disables auto-scroll so they can read freely.
// Reset to false at the start of each new research run.
let userScrolledUp = false;

window.addEventListener('scroll', () => {
    if (!streamingActive) return;
    const distFromBottom = document.documentElement.scrollHeight - window.scrollY - window.innerHeight;
    // If user is more than 80px above the bottom, treat it as intentional scroll up
    if (distFromBottom > 80) {
        userScrolledUp = true;
    } else {
        // They scrolled back down to the bottom — resume auto-scroll
        userScrolledUp = false;
    }
}, { passive: true });

const STEPS = [
    { id: 'step-planning',   step: 'planning'  },
    { id: 'step-searching',  step: 'searching' },
    { id: 'step-writing',    step: 'writing'   },
    { id: 'step-reviewing',  step: 'reviewing' },
    { id: 'step-done',       step: 'done'      },
];
const CONNECTORS = ['conn-1', 'conn-2', 'conn-3', 'conn-4'];
const iterationBadge = document.getElementById('iterationBadge');

function resetPipeline() {
    STEPS.forEach(s => document.getElementById(s.id).classList.remove('active', 'done'));
    CONNECTORS.forEach(c => document.getElementById(c).classList.remove('done'));
    pipelineSubLabel.textContent = '';
    iterationBadge.classList.add('hidden');
    iterationBadge.textContent = '';
}

function advancePipeline(stepKey, statusText, iteration) {
    const targetIdx = STEPS.findIndex(s => s.step === stepKey);
    if (targetIdx === -1) return;

    STEPS.forEach((s, i) => {
        const el = document.getElementById(s.id);
        el.classList.remove('active', 'done');
        if (i < targetIdx)        el.classList.add('done');
        else if (i === targetIdx) el.classList.add('active');
    });
    CONNECTORS.forEach((c, i) => {
        document.getElementById(c).classList.toggle('done', i < targetIdx);
    });

    pipelineSubLabel.textContent = statusText || '';

    if (iteration && iteration > 1) {
        iterationBadge.textContent = `Revision ${iteration - 1}`;
        iterationBadge.classList.remove('hidden');
    } else {
        iterationBadge.classList.add('hidden');
    }
}

function completePipeline() {
    STEPS.forEach(s => {
        const el = document.getElementById(s.id);
        el.classList.remove('active');
        el.classList.add('done');
    });
    CONNECTORS.forEach(c => document.getElementById(c).classList.add('done'));
    pipelineSubLabel.textContent = 'Complete';
    iterationBadge.classList.add('hidden');
}

// Copy button
copyBtn.addEventListener('click', async () => {
    if (!currentRawMarkdown) return;
    try {
        await navigator.clipboard.writeText(currentRawMarkdown);
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg> Copied!';
        copyBtn.classList.add('copy-success');
        setTimeout(() => { copyBtn.innerHTML = originalHTML; copyBtn.classList.remove('copy-success'); }, 2000);
    } catch (err) { alert('Could not copy report.'); }
});

// Download button
downloadBtn.addEventListener('click', () => {
    if (!currentRawMarkdown) return;
    const blob = new Blob([currentRawMarkdown], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `research_report_${new Date().toISOString().slice(0,19).replace(/:/g,'-')}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

// Main research handler
researchBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    const depth = (document.querySelector('input[name="depth"]:checked') || {}).value || 'standard';
    if (!query) { alert('Please enter a research question'); return; }
    if (streamingActive) return;

    researchBtn.disabled = true;
    researchBtn.classList.add('opacity-50', 'cursor-not-allowed');
    reportContainer.style.display = 'none';
    currentRawMarkdown = '';
    reportContent.innerHTML = '';

    userScrolledUp = false;
    pipelineTracker.style.display = 'block';
    resetPipeline();

    const isFast = depth === 'fast';
    document.getElementById('step-reviewing').style.display = isFast ? 'none' : '';
    document.getElementById('conn-4').style.display         = isFast ? 'none' : '';

    advancePipeline('planning', 'Planning research...', 1);

    try {
        const response = await fetch(`${backendUrl}/research/stream`, {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ query, depth }),
        });
        if (!response.body) throw new Error('No response body');

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer    = '';
        streamingActive = true;

        outer: while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                let data;
                try { data = JSON.parse(line.slice(6)); } catch { continue; }

                if (data.type === 'status') {
                    advancePipeline(data.step || 'planning', data.text, data.iteration);

                } else if (data.type === 'token') {
                    // First token — show the report container
                    if (!currentRawMarkdown) {
                        reportContainer.style.display = 'block';
                    }
                    currentRawMarkdown += data.text;
                    reportContent.innerHTML = marked.parse(currentRawMarkdown);
                    // Only auto-scroll if the user hasn't manually scrolled up.
                    // Use instant (not smooth) to avoid fighting the user's scroll
                    // position with overlapping animations on every token.
                    if (!userScrolledUp) {
                        window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'instant' });
                    }

                } else if (data.type === 'done') {
                    completePipeline();
                    break outer;

                } else if (data.type === 'error') {
                    alert('Error: ' + data.text);
                    break outer;
                }
            }
        }
    } catch (err) {
        console.error(err);
        alert('Error: Could not reach backend. Make sure the server is running.');
    } finally {
        researchBtn.disabled = false;
        researchBtn.classList.remove('opacity-50', 'cursor-not-allowed');
        streamingActive = false;
    }
});