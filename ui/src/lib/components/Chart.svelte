<script lang="ts">
	export interface Series {
		name: string;
		color: string;
		values: number[];
	}

	interface Props {
		series: Series[];
		times?: number[];
		max?: number;
		unit?: string;
		height?: number;
		window?: number;
		compact?: boolean;
		stacked?: boolean;
		format?: (v: number) => string;
	}
	let {
		series,
		times = [],
		max,
		unit = '',
		height = 150,
		window = 120,
		compact = false,
		stacked = false,
		format = (v: number) =>
			v >= 100 || Number.isInteger(v) ? Math.round(v).toLocaleString() : v.toFixed(1)
	}: Props = $props();

	let width = $state(0);
	let hover = $state<number | null>(null);

	const pad = $derived(compact ? { l: 0, r: 0, t: 2, b: 2 } : { l: 66, r: 8, t: 6, b: 18 });
	const plotW = $derived(Math.max(1, width - pad.l - pad.r));
	const plotH = $derived(height - pad.t - pad.b);

	// keep only the samples inside the window
	const view = $derived(series.map((s) => ({ ...s, values: s.values.slice(-window) })));
	const t = $derived(times.slice(-window));
	const len = $derived(Math.max(0, ...view.map((s) => s.values.length)));

	// stacked series draw cumulative tops
	const tops = $derived.by(() => {
		if (!stacked) return view.map((s) => s.values);
		const acc = new Array(len).fill(0);
		return view.map((s) => s.values.map((v, i) => (acc[i] += v)));
	});

	function niceMax(v: number) {
		if (v <= 0) return 1;
		const p = 10 ** Math.floor(Math.log10(v));
		return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v) ?? v;
	}
	const top = $derived(max ?? niceMax(Math.max(0, ...tops.flat())));

	const x = (i: number) => pad.l + ((i + window - len) / (window - 1)) * plotW;
	const y = (v: number) => pad.t + plotH * (1 - Math.min(Math.max(v, 0), top) / top);

	const line = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join('');
	function band(upper: number[], lower: number[] | null) {
		if (upper.length < 2) return '';
		const back = lower
			? lower.map((v, i) => `L${x(i)},${y(v)}`).reverse().join('')
			: `L${x(upper.length - 1)},${y(0)}L${x(0)},${y(0)}`;
		return `${line(upper)}${back}Z`;
	}

	const clock = (ms: number) =>
		new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

	function onMove(e: PointerEvent) {
		const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
		const i = Math.round(((e.clientX - rect.left - pad.l) / plotW) * (window - 1)) - (window - len);
		hover = i >= 0 && i < len ? i : null;
	}
</script>

<div class="relative" bind:clientWidth={width}>
	<svg
		{width}
		{height}
		role="img"
		aria-label={series.map((s) => s.name).join(' and ')}
		onpointermove={compact ? undefined : onMove}
		onpointerleave={() => (hover = null)}
		class="block"
	>
		{#if !compact}
			{#each [0, 0.5, 1] as f}
				<line x1={pad.l} x2={pad.l + plotW} y1={y(top * f)} y2={y(top * f)} stroke="var(--border)" />
				<text x={pad.l - 6} y={y(top * f) + 3} text-anchor="end" font-size="10" fill="var(--text-muted)"
					>{format(top * f)}{unit}</text
				>
			{/each}
			{#if t.length > 0}
				<!-- axis spans the whole window so the left label is the window start not the first sample -->
				<text x={pad.l} y={height - 4} font-size="10" fill="var(--text-muted)"
					>{clock(t[len - 1] - (window - 1) * 1000)}</text
				>
				<text x={pad.l + plotW} y={height - 4} font-size="10" text-anchor="end" fill="var(--text-muted)"
					>{clock(t[len - 1])}</text
				>
			{/if}
		{:else}
			<rect x="0" y="0" {width} {height} fill="none" stroke="var(--border)" />
		{/if}

		{#if stacked}
			{#each view as s, k}
				<path d={band(tops[k], k ? tops[k - 1] : null)} fill={s.color} fill-opacity="0.85" />
				<path d={line(tops[k])} fill="none" stroke="var(--surface)" stroke-width="1" />
			{/each}
		{:else}
			{#if view.length === 1}
				<path d={band(view[0].values, null)} fill={view[0].color} fill-opacity="0.12" />
			{/if}
			{#each view as s}
				<path d={line(s.values)} fill="none" stroke={s.color} stroke-width={compact ? 1.5 : 2} stroke-linejoin="round" />
			{/each}
		{/if}

		{#if hover !== null}
			<line x1={x(hover)} x2={x(hover)} y1={pad.t} y2={pad.t + plotH} stroke="var(--text-muted)" />
			{#if !stacked}
				{#each view as s}
					{#if s.values[hover] !== undefined}
						<circle cx={x(hover)} cy={y(s.values[hover])} r="4" fill={s.color} stroke="var(--surface)" stroke-width="2" />
					{/if}
				{/each}
			{/if}
		{/if}
	</svg>

	{#if !compact && series.length > 1}
		<div class="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
			{#each series as s}
				<span class="flex items-center gap-1.5">
					<span class="inline-block h-2 w-3" style="background: {s.color}"></span>{s.name}
				</span>
			{/each}
		</div>
	{/if}

	{#if hover !== null}
		<div
			class="pointer-events-none absolute top-0 z-10 border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs whitespace-nowrap text-[var(--fg)] shadow-sm"
			style="left: {Math.min(x(hover) + 8, Math.max(0, width - 150))}px"
		>
			<div class="text-[var(--text-muted)]">{t[hover] ? clock(t[hover]) : `${len - 1 - hover}s ago`}</div>
			{#each [...view].reverse() as s}
				<div class="tabular-nums">
					{#if series.length > 1}<span style="color: {s.color}">■</span> {s.name}{/if}
					{format(s.values[hover] ?? 0)}{unit}
				</div>
			{/each}
		</div>
	{/if}
</div>
