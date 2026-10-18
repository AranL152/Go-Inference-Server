<script lang="ts">
	import Panel from './Panel.svelte';
	import type { Sample, ServerConfig } from '$lib/stats';

	let { samples, cfg }: { samples: Sample[]; cfg: ServerConfig | null } = $props();

	// average the last few busy seconds so the estimate does not flicker
	const recent = $derived(samples.slice(-5).filter((x) => x.req_per_sec > 0));
	const mean = (f: (x: Sample) => number) =>
		recent.length ? recent.reduce((a, x) => a + f(x), 0) / recent.length : 0;

	const rows = $derived.by(() => {
		if (!cfg || !recent.length) return [];
		const rps = mean((x) => x.req_per_sec);
		const out: { name: string; how: string; cap: number }[] = [];
		const decodeMs = mean((x) => x.decode_ms + x.letterbox_ms);
		const decoders = Math.min(cfg.cpu_decoders, cfg.cores);
		if (decoders > 0 && decodeMs > 0) {
			out.push({
				name: 'JPEG decode + letterbox',
				how: `${decoders} decoders × 1000 / ${decodeMs.toFixed(1)} ms per image`,
				cap: (decoders * 1000) / decodeMs
			});
		}
		const busy = mean((x) => x.gpu?.util ?? 0);
		if (busy > 3) {
			out.push({
				name: 'GPU',
				how: `${Math.round(rps)} img/s at ${busy.toFixed(0)}% busy`,
				cap: rps / (busy / 100)
			});
		}
		const postMs = mean((x) => x.post_ms);
		const batch = mean((x) => x.avg_batch);
		if (postMs > 0 && batch > 0) {
			out.push({
				name: 'Unpack',
				how: `${cfg.unpack_workers} workers × ${batch.toFixed(1)} images / ${postMs.toFixed(2)} ms per batch`,
				cap: (cfg.unpack_workers * batch * 1000) / postMs
			});
		}
		return out.sort((a, b) => a.cap - b.cap);
	});
	const rps = $derived(mean((x) => x.req_per_sec));
	// scale to the two tightest stages so a far larger ceiling does not flatten them
	const scale = $derived(Math.max(rps, rows[1]?.cap ?? rows[0]?.cap ?? 0) * 1.15 || 1);
	const finding = $derived.by(() => {
		if (!rows.length) return '';
		const lim = rows[0];
		const next = rows[1];
		return `${lim.name} caps throughput near ${Math.round(lim.cap).toLocaleString()} img/s. Serving ${Math.round(rps).toLocaleString()} img/s, ${Math.round((100 * rps) / lim.cap)}% of that ceiling${next ? `; ${next.name} could take ${Math.round(next.cap).toLocaleString()}` : ''}.`;
	});
	const n = (v: number) => Math.round(v).toLocaleString();
</script>

<Panel title="Bottleneck" sub="ceiling of each stage from its live cost per item" {finding}>
	{#if !rows.length}
		<p class="text-sm text-[var(--text-muted)]">No traffic in the last few seconds. Send load to estimate each stage's ceiling.</p>
	{:else}
		<div class="space-y-3">
			{#each rows as r, i}
				<div>
					<div class="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-sm">
						<span class="text-[var(--fg)] {i === 0 ? 'font-semibold' : ''}">{r.name}</span>
						<span class="text-xs text-[var(--text-muted)] tabular-nums">{r.how} = <span class="text-[var(--fg)]">{n(r.cap)} img/s</span></span>
					</div>
					<div class="relative h-3 rounded-full bg-[var(--bg)]">
						<div
							class="h-full rounded-full {i === 0 ? 'bg-[var(--accent)]' : 'bg-[var(--border)]'}"
							style="width: {Math.min(100, (100 * r.cap) / scale)}%"
						></div>
						<div
							class="absolute -top-1 h-5 w-0.5 bg-[var(--fg)]"
							style="left: {(100 * rps) / scale}%"
							title="current throughput"
						></div>
					</div>
				</div>
			{/each}
			<p class="text-xs text-[var(--text-muted)] tabular-nums">
				<span class="mr-1 inline-block h-3 w-0.5 translate-y-0.5 bg-[var(--fg)]"></span>current throughput {n(rps)} img/s
			</p>
		</div>
	{/if}
</Panel>
