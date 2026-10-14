<script lang="ts">
	export interface Series {
		name: string;
		color: string;
		values: number[];
	}

	interface Props {
		series: Series[];
		max?: number;
		unit?: string;
		height?: number;
		window?: number;
		compact?: boolean;
		format?: (v: number) => string;
	}
	let {
		series,
		max,
		unit = '',
		height = 180,
		window = 120,
		compact = false,
		format = (v: number) =>
			v >= 100 || Number.isInteger(v) ? Math.round(v).toLocaleString() : v.toFixed(1)
	}: Props = $props();

	let width = $state(0);
	let hover = $state<number | null>(null);

	const pad = $derived(compact ? { l: 0, r: 0, t: 2, b: 2 } : { l: 60, r: 8, t: 8, b: 20 });
	const plotW = $derived(Math.max(1, width - pad.l - pad.r));
	const plotH = $derived(height - pad.t - pad.b);

	function niceMax(v: number) {
		if (v <= 0) return 1;
		const p = 10 ** Math.floor(Math.log10(v));
		return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v) ?? v;
	}
	const top = $derived(max ?? niceMax(Math.max(0, ...series.flatMap((s) => s.values))));
	const len = $derived(Math.max(0, ...series.map((s) => s.values.length)));

	// newest sample sits on the right edge like Task Manager
	const x = (i: number, n: number) => pad.l + ((i + window - n) / (window - 1)) * plotW;
	const y = (v: number) => pad.t + plotH * (1 - Math.min(v, top) / top);

	function line(values: number[]) {
		return values.map((v, i) => `${i ? 'L' : 'M'}${x(i, values.length)},${y(v)}`).join('');
	}
	function area(values: number[]) {
		if (values.length < 2) return '';
		const n = values.length;
		return `${line(values)}L${x(n - 1, n)},${y(0)}L${x(0, n)},${y(0)}Z`;
	}

	function onMove(e: PointerEvent) {
		const rect = (e.currentTarget as SVGElement).getBoundingClientRect();
		const i = Math.round(((e.clientX - rect.left - pad.l) / plotW) * (window - 1)) - (window - len);
		hover = i >= 0 && i < len ? i : null;
	}
</script>

<div class="relative" bind:clientWidth={width}>
	{#if !compact && series.length > 1}
		<div class="mb-2 flex gap-4 text-xs text-[var(--text-muted)]">
			{#each series as s}
				<span class="flex items-center gap-1.5">
					<span class="inline-block h-0.5 w-3" style="background: {s.color}"></span>{s.name}
				</span>
			{/each}
		</div>
	{/if}
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
				<line
					x1={pad.l}
					x2={pad.l + plotW}
					y1={y(top * f)}
					y2={y(top * f)}
					stroke="var(--border)"
					stroke-width="1"
				/>
				<text
					x={pad.l - 6}
					y={y(top * f) + 3}
					text-anchor="end"
					font-size="10"
					fill="var(--text-muted)">{format(top * f)}{unit}</text
				>
			{/each}
			<text x={pad.l} y={height - 4} font-size="10" fill="var(--text-muted)">{window}s</text>
			<text x={pad.l + plotW} y={height - 4} font-size="10" text-anchor="end" fill="var(--text-muted)"
				>now</text
			>
		{:else}
			<rect x="0" y="0" {width} {height} fill="none" stroke="var(--border)" />
		{/if}

		{#if series.length === 1}
			<path d={area(series[0].values)} fill={series[0].color} fill-opacity="0.12" />
		{/if}
		{#each series as s}
			<path
				d={line(s.values)}
				fill="none"
				stroke={s.color}
				stroke-width={compact ? 1.5 : 2}
				stroke-linejoin="round"
			/>
		{/each}

		{#if hover !== null}
			<line
				x1={x(hover, len)}
				x2={x(hover, len)}
				y1={pad.t}
				y2={pad.t + plotH}
				stroke="var(--text-muted)"
				stroke-width="1"
			/>
			{#each series as s}
				{#if s.values[hover] !== undefined}
					<circle
						cx={x(hover, s.values.length)}
						cy={y(s.values[hover])}
						r="4"
						fill={s.color}
						stroke="var(--surface)"
						stroke-width="2"
					/>
				{/if}
			{/each}
		{/if}
	</svg>

	{#if hover !== null}
		<div
			class="pointer-events-none absolute top-0 border border-[var(--border)] bg-[var(--surface)] px-2 py-1 text-xs whitespace-nowrap text-[var(--fg)] shadow-sm"
			style="left: {Math.min(x(hover, len) + 8, width - 120)}px"
		>
			<div class="text-[var(--text-muted)]">{len - 1 - hover}s ago</div>
			{#each series as s}
				<div class="tabular-nums">
					{#if series.length > 1}<span style="color: {s.color}">■</span> {s.name}{/if}
					{format(s.values[hover] ?? 0)}{unit}
				</div>
			{/each}
		</div>
	{/if}
</div>
