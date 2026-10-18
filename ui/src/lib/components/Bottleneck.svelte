<script lang="ts">
	import Panel from './Panel.svelte';
	import type { Sample, ServerConfig } from '$lib/stats';

	// span is how many busy seconds to average: a few for live, the whole run for a job
	let { samples, cfg, span = 5 }: { samples: Sample[]; cfg: ServerConfig | null; span?: number } = $props();

	const recent = $derived(samples.filter((x) => x.req_per_sec > 0).slice(-span));
	const mean = (f: (x: Sample) => number) =>
		recent.length ? recent.reduce((a, x) => a + f(x), 0) / recent.length : 0;
	const lastBusy = $derived(recent.at(-1)?.t ?? 0);
	const stale = $derived(span <= 10 && lastBusy > 0 && (samples.at(-1)?.t ?? 0) - lastBusy > 5000);

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
			out.push({ name: 'GPU', how: `${Math.round(rps)} img/s at ${busy.toFixed(0)}% busy`, cap: rps / (busy / 100) });
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
	const n = (v: number) => Math.round(v).toLocaleString();
	const finding = $derived(
		rows.length
			? `Limit: ${rows[0].name} · ceiling ${n(rows[0].cap)} img/s · at ${Math.round((100 * rps) / rows[0].cap)}%${rows[1] ? ` · next ${rows[1].name} ${n(rows[1].cap)}` : ''}`
			: ''
	);
	const sub = $derived(
		stale ? `last busy at ${new Date(lastBusy).toLocaleTimeString()}` : `ceiling of each stage from its cost per item, ${span <= 10 ? `last ${recent.length} busy s` : 'whole run'}`
	);
</script>

<Panel title="Bottleneck" {sub} {finding}>
	{#if !rows.length}
		<p class="text-sm text-[var(--text-muted)]">No traffic yet.</p>
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
						<div class="absolute -top-1 h-5 w-0.5 bg-[var(--fg)]" style="left: {(100 * rps) / scale}%" title="throughput"></div>
					</div>
				</div>
			{/each}
			<p class="text-xs text-[var(--text-muted)] tabular-nums">
				<span class="mr-1 inline-block h-3 w-0.5 translate-y-0.5 bg-[var(--fg)]"></span>throughput {n(rps)} img/s
			</p>
		</div>
	{/if}
</Panel>
