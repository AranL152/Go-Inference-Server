<script lang="ts">
	import Panel from './Panel.svelte';
	import Chart from './Chart.svelte';
	import type { Sample } from '$lib/stats';

	let { samples, range }: { samples: Sample[]; range: number } = $props();

	const busy = $derived(samples.slice(-10).filter((x) => x.req_per_sec > 0));
	const mean = (f: (x: Sample) => number) => (busy.length ? busy.reduce((a, x) => a + f(x), 0) / busy.length : 0);
	const kb = (b: number) => (b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${(b / 1024).toFixed(1)} KB`);
	const us = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} ms` : `${v.toFixed(0)} µs`);

	const tiles = $derived([
		{ label: 'Allocated per request', value: kb(mean((x) => x.alloc_bytes_per_req)), f: (x: Sample) => x.alloc_bytes_per_req / 1024 },
		{ label: 'Objects per request', value: mean((x) => x.allocs_per_req).toFixed(0), f: (x: Sample) => x.allocs_per_req },
		{ label: 'GC cycles per second', value: mean((x) => x.gc_per_sec).toFixed(1), f: (x: Sample) => x.gc_per_sec },
		{ label: 'GC share of CPU', value: `${mean((x) => x.gc_cpu_pct).toFixed(1)}%`, f: (x: Sample) => x.gc_cpu_pct },
		{ label: 'GC pause p99', value: us(mean((x) => x.gc_pause_p99_us)), f: (x: Sample) => x.gc_pause_p99_us },
		{ label: 'Wait for CPU p99', value: us(mean((x) => x.sched_p99_us)), f: (x: Sample) => x.sched_p99_us }
	]);
	const finding = $derived(
		busy.length
			? `Each request allocates ${kb(mean((x) => x.alloc_bytes_per_req))} in ${mean((x) => x.allocs_per_req).toFixed(0)} objects; GC runs ${mean((x) => x.gc_per_sec).toFixed(1)} times a second and takes ${mean((x) => x.gc_cpu_pct).toFixed(1)}% of the server's CPU.`
			: ''
	);
</script>

<Panel title="Go runtime" sub="garbage per request, GC cost and scheduler delay from runtime/metrics" {finding}>
	{#if !busy.length}
		<p class="text-sm text-[var(--text-muted)]">No traffic in the last 10 s.</p>
	{:else}
		<div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
			{#each tiles as tile}
				<div class="rounded-xl bg-[var(--bg)] px-3 pt-2">
					<div class="text-xs text-[var(--text-muted)]">{tile.label}</div>
					<div class="text-lg font-semibold text-[var(--fg)] tabular-nums">{tile.value}</div>
					<Chart series={[{ name: tile.label, color: 'var(--series)', values: samples.map(tile.f) }]} window={range} height={28} compact frame={false} />
				</div>
			{/each}
		</div>
		<p class="mt-3 text-xs text-[var(--text-muted)]">Live heap {mean((x) => x.heap_live_mb).toFixed(0)} MB, averaged over the last 10 busy seconds.</p>
	{/if}
</Panel>
