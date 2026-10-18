<script lang="ts">
	import { onMount } from 'svelte';

	interface Span {
		seq: number;
		n: number;
		lane: number;
		worker: number;
		split: boolean;
		gather: number;
		pack: number;
		packed: number;
		infer: number;
		h2d_end: number;
		compute_end: number;
		run_end: number;
		post_start: number;
		post_end: number;
	}

	let { compact = false }: { compact?: boolean } = $props();
	const ROW_H = $derived(compact ? 18 : 26);

	const windows = [
		{ label: '100 ms', ms: 100 },
		{ label: '250 ms', ms: 250 },
		{ label: '1 s', ms: 1000 }
	];
	const LAG_MS = 300; // spans are recorded once unpacked, so draw a little behind now
	const LABEL_W = 64;
	const ROW_GAP = 6;
	const AXIS_H = 18;

	let windowMs = $state(250);
	let paused = $state(false);
	let lanes = $state(1);
	let maxBatch = $state(0);
	let error = $state(false);
	let stats = $state({ batches: 0, avg: 0, busy: 0, rate: 0 });
	let hover = $state<{ x: number; y: number; s: Span } | null>(null);

	let canvas: HTMLCanvasElement;
	let width = $state(0);
	let rootH = $state(0);
	let tipH = $state(0);

	let spans: Span[] = [];
	let lastSeq = -1;
	// server clock in ns lined up with performance.now() in ms
	let anchorServer = 0;
	let anchorClient = 0;
	let frozenEnd = 0;

	const rows = $derived(['Batcher', ...Array.from({ length: lanes }, (_, i) => `Lane ${i + 1}`), 'Unpack']);
	const height = $derived(rows.length * (ROW_H + ROW_GAP) + AXIS_H);

	function serverNow() {
		return anchorServer + (performance.now() - anchorClient) * 1e6;
	}

	function viewEnd() {
		return paused ? frozenEnd : serverNow() - LAG_MS * 1e6;
	}

	async function poll() {
		if (paused) return;
		const sent = performance.now();
		try {
			const r = await fetch(`/api/trace?since=${lastSeq}`).then((r) => r.json());
			const got = performance.now();
			anchorServer = r.now;
			anchorClient = (sent + got) / 2;
			lanes = r.lanes;
			maxBatch = r.max_batch;
			if (r.spans.length) {
				// a restarted server starts its sequence over
				if (r.spans[0].seq <= lastSeq) spans = [];
				spans.push(...r.spans);
				lastSeq = r.spans[r.spans.length - 1].seq;
			}
			const keepFrom = serverNow() - (LAG_MS + 2000) * 1e6;
			let i = 0;
			while (i < spans.length && spans[i].post_end < keepFrom) i++;
			if (i) spans = spans.slice(i);
			error = false;
		} catch {
			error = true;
		}
	}

	let colors: Record<string, string> = {};
	function readColors() {
		const css = getComputedStyle(document.documentElement);
		const v = (n: string) => css.getPropertyValue(n).trim();
		colors = {
			gather: v('--border'),
			pack: v('--cat-3'),
			h2d: v('--cat-4'),
			compute: v('--accent'),
			d2h: v('--cat-6'),
			post: v('--cat-7'),
			text: v('--text-muted'),
			fg: v('--fg'),
			grid: v('--border'),
			band: v('--bg')
		};
	}

	const legend = [
		{ css: '--border', label: 'Filling batch' },
		{ css: '--cat-3', label: 'Pack' },
		{ css: '--cat-4', label: 'Copy to GPU' },
		{ css: '--accent', label: 'GPU compute' },
		{ css: '--cat-6', label: 'Copy from GPU' },
		{ css: '--cat-7', label: 'Unpack' }
	];

	// segments each span draws: row index, start, end, color key
	function segments(s: Span): [number, number, number, string][] {
		const lane = 1 + s.lane;
		const post = rows.length - 1;
		return [
			[0, s.gather, s.pack, 'gather'],
			[0, s.pack, s.packed, 'pack'],
			[lane, s.infer, s.h2d_end, 'h2d'],
			[lane, s.h2d_end, s.compute_end, 'compute'],
			[lane, s.compute_end, s.run_end, 'd2h'],
			[post, s.post_start, s.post_end, 'post']
		];
	}

	function draw() {
		if (!canvas || !width) return;
		const dpr = window.devicePixelRatio || 1;
		if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
			canvas.width = Math.round(width * dpr);
			canvas.height = Math.round(height * dpr);
		}
		const ctx = canvas.getContext('2d')!;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
		ctx.clearRect(0, 0, width, height);

		const end = viewEnd();
		const span = windowMs * 1e6;
		const start = end - span;
		const plotW = width - LABEL_W;
		const x = (t: number) => LABEL_W + ((t - start) / span) * plotW;
		const rowY = (r: number) => r * (ROW_H + ROW_GAP);

		ctx.font = '11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
		ctx.textBaseline = 'middle';
		rows.forEach((name, r) => {
			ctx.fillStyle = colors.band;
			ctx.fillRect(LABEL_W, rowY(r), plotW, ROW_H);
			ctx.fillStyle = colors.text;
			ctx.fillText(name, 0, rowY(r) + ROW_H / 2);
		});

		// time grid, labelled back from the right edge
		const step = windowMs <= 100 ? 10 : windowMs <= 250 ? 25 : 100;
		const plotBottom = rowY(rows.length) - ROW_GAP;
		ctx.textAlign = 'center';
		for (let ms = 0; ms <= windowMs; ms += step) {
			const gx = LABEL_W + plotW - (ms / windowMs) * plotW;
			ctx.strokeStyle = colors.grid;
			ctx.globalAlpha = 0.6;
			ctx.beginPath();
			ctx.moveTo(gx, 0);
			ctx.lineTo(gx, plotBottom);
			ctx.stroke();
			ctx.globalAlpha = 1;
			if (ms % (step * 2) === 0) {
				ctx.fillStyle = colors.text;
				ctx.fillText(ms === 0 ? 'now' : `−${ms} ms`, Math.min(Math.max(gx, LABEL_W + 18), width - 18), plotBottom + AXIS_H / 2 + 2);
			}
		}
		ctx.textAlign = 'left';

		let batches = 0,
			images = 0,
			laneNs = 0;
		ctx.save();
		ctx.beginPath();
		ctx.rect(LABEL_W, 0, plotW, plotBottom);
		ctx.clip();
		for (const s of spans) {
			if (s.post_end < start || s.gather > end) continue;
			if (s.infer >= start && s.infer < end) {
				batches++;
				images += s.n;
			}
			laneNs += Math.max(0, Math.min(s.run_end, end) - Math.max(s.infer, start));
			for (const [r, a, b, key] of segments(s)) {
				if (b <= a) continue;
				const x0 = x(a);
				const w = Math.max(x(b) - x0, 1);
				ctx.fillStyle = colors[key];
				ctx.globalAlpha = key === 'post' ? 0.75 : 1;
				ctx.fillRect(x0, rowY(r) + 2, w, ROW_H - 4);
				ctx.globalAlpha = 1;
				if (key === 'compute' && w > 22 && !compact) {
					ctx.fillStyle = '#fff';
					ctx.fillText(String(s.n), x0 + 4, rowY(r) + ROW_H / 2);
				}
			}
			if (hover && hover.s.seq === s.seq) {
				ctx.strokeStyle = colors.fg;
				ctx.lineWidth = 1.5;
				for (const [r, a, b] of segments(s)) {
					if (b > a) ctx.strokeRect(x(a), rowY(r) + 1.5, Math.max(x(b) - x(a), 1), ROW_H - 3);
				}
				ctx.lineWidth = 1;
			}
		}
		ctx.restore();
		stats = {
			batches,
			avg: batches ? images / batches : 0,
			busy: (100 * laneNs) / (span * lanes),
			rate: images / (windowMs / 1000)
		};
	}

	function onMove(e: MouseEvent) {
		const rect = canvas.getBoundingClientRect();
		const px = e.clientX - rect.left;
		const py = e.clientY - rect.top;
		const top = canvas.offsetTop + py;
		const r = Math.floor(py / (ROW_H + ROW_GAP));
		if (px < LABEL_W || r < 0 || r >= rows.length) {
			hover = null;
			return;
		}
		const end = viewEnd();
		const t = end - windowMs * 1e6 + ((px - LABEL_W) / (width - LABEL_W)) * windowMs * 1e6;
		const hit = spans.find((s) => segments(s).some(([sr, a, b]) => sr === r && t >= a && t <= Math.max(b, a + 2e5)));
		hover = hit ? { x: px, y: top, s: hit } : null;
	}

	const ms = (a: number, b: number) => (Math.max(b - a, 0) / 1e6).toFixed(2);

	function togglePause() {
		frozenEnd = viewEnd();
		paused = !paused;
		if (!paused) hover = null;
	}

	onMount(() => {
		readColors();
		const themeWatch = new MutationObserver(readColors);
		themeWatch.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
		poll();
		const t = setInterval(poll, 100);
		let raf = 0;
		const loop = () => {
			draw();
			raf = requestAnimationFrame(loop);
		};
		raf = requestAnimationFrame(loop);
		return () => {
			clearInterval(t);
			cancelAnimationFrame(raf);
			themeWatch.disconnect();
		};
	});
</script>

<div class="relative" bind:clientHeight={rootH}>
	<div class="mb-3 flex flex-wrap items-center justify-between gap-3">
		<div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)] tabular-nums">
			<span><span class="text-[var(--fg)]">{stats.batches}</span> batches</span>
			<span>avg <span class="text-[var(--fg)]">{stats.avg.toFixed(1)}</span>{maxBatch ? ` of ${maxBatch}` : ''} per batch</span>
			<span>lanes busy <span class="text-[var(--fg)]">{Math.min(stats.busy, 100).toFixed(0)}%</span></span>
			<span><span class="text-[var(--fg)]">{Math.round(stats.rate).toLocaleString()}</span> img/s</span>
			{#if error}<span class="text-[var(--status-critical)]">disconnected</span>{/if}
		</div>
		<div class="flex items-center gap-2 text-xs">
			<div class="flex rounded-full bg-[var(--bg)] p-0.5 ring-1 ring-[var(--border)]">
				{#each windows as w}
					<button
						onclick={() => (windowMs = w.ms)}
						class="rounded-full px-2.5 py-0.5 {windowMs === w.ms
							? 'bg-[var(--fg)] text-[var(--bg)]'
							: 'text-[var(--text-muted)] hover:text-[var(--fg)]'}">{w.label}</button
					>
				{/each}
			</div>
			<button
				onclick={togglePause}
				class="rounded-full px-3 py-0.5 ring-1 {paused
					? 'bg-[var(--accent)] text-white ring-[var(--accent)]'
					: 'text-[var(--fg)] ring-[var(--border)] hover:ring-[var(--accent)]'}">{paused ? 'Resume' : 'Pause'}</button
			>
		</div>
	</div>

	<div bind:clientWidth={width}>
		<canvas
			bind:this={canvas}
			style="width: {width}px; height: {height}px"
			class="block"
			onmousemove={onMove}
			onmouseleave={() => (hover = null)}
		></canvas>
		{#if hover}
			{@const s = hover.s}
			<div
				bind:clientHeight={tipH}
				class="pointer-events-none absolute z-10 w-56 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs shadow-md"
				style="left: {Math.min(hover.x + 12, width - 232)}px; top: {Math.max(0, Math.min(hover.y - tipH / 2, rootH - tipH))}px"
			>
				<div class="mb-1 font-semibold text-[var(--fg)]">
					Batch {s.seq} · {s.n} image{s.n === 1 ? '' : 's'} · lane {s.lane + 1}
				</div>
				<div class="grid grid-cols-[1fr_auto] gap-x-3 text-[var(--text-muted)] tabular-nums">
					<span>Filling batch</span><span>{ms(s.gather, s.pack)} ms</span>
					<span>Pack</span><span>{ms(s.pack, s.packed)} ms</span>
					<span>Wait for lane</span><span>{ms(s.packed, s.infer)} ms</span>
					{#if s.split}
						<span>Copy to GPU</span><span>{ms(s.infer, s.h2d_end)} ms</span>
						<span>GPU compute</span><span>{ms(s.h2d_end, s.compute_end)} ms</span>
						<span>Copy from GPU</span><span>{ms(s.compute_end, s.run_end)} ms</span>
					{:else}
						<span>GPU run, copies included</span><span>{ms(s.infer, s.run_end)} ms</span>
					{/if}
					<span>Wait for worker</span><span>{ms(s.run_end, s.post_start)} ms</span>
					<span>Unpack</span><span>{ms(s.post_start, s.post_end)} ms</span>
					<span class="font-semibold text-[var(--fg)]">Total</span>
					<span class="font-semibold text-[var(--fg)]">{ms(s.gather, s.post_end)} ms</span>
				</div>
			</div>
		{/if}
	</div>

	<div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
		{#each legend as l}
			<span class="flex items-center gap-1.5">
				<span class="inline-block h-2 w-3 rounded-sm" style="background: var({l.css})"></span>
				{l.label}
			</span>
		{/each}
	</div>
</div>
