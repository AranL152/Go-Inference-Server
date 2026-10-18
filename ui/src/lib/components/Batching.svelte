<script lang="ts">
	import { onMount } from 'svelte';
	import Panel from './Panel.svelte';
	import type { BatchingStats } from '$lib/stats';

	let b = $state<BatchingStats | null>(null);

	async function poll() {
		b = await fetch('/api/batching')
			.then((r) => r.json())
			.catch(() => b);
	}

	onMount(() => {
		poll();
		const t = setInterval(poll, 2000);
		return () => clearInterval(t);
	});

	const peak = $derived(Math.max(1, ...(b?.sizes ?? [])));
	const finding = $derived.by(() => {
		if (!b || !b.batches) return '';
		const out: string[] = [];
		const avg = b.avg_size.toFixed(1);
		if (b.avg_size / b.max_batch < 0.5 && b.fill_p50_ms >= 0.8 * b.max_wait_ms) {
			out.push(`Batches close on the ${b.max_wait_ms} ms timer at ${avg} of ${b.max_batch} images: requests arrive too sparsely to fill them, so most pay the full wait.`);
		} else if (b.full_pct > 80) {
			out.push(`${b.full_pct.toFixed(0)}% of batches are full, so a larger max batch could lift throughput.`);
		} else {
			out.push(`Average batch ${avg} of ${b.max_batch}, ${b.full_pct.toFixed(0)}% full.`);
		}
		if (b.lane_wait_p99_ms > 1) out.push(`Packed batches wait up to ${b.lane_wait_p99_ms.toFixed(1)} ms for a free lane.`);
		return out.join(' ');
	});
</script>

<Panel title="Batching" sub="dynamic batches formed in the last 10 s" {finding}>
	{#if !b || !b.batches}
		<p class="text-sm text-[var(--text-muted)]">No batches in the last 10 s.</p>
	{:else}
		<div class="flex h-28 items-end gap-0.5">
			{#each b.sizes as count, i}
				<div class="flex h-full flex-1 flex-col justify-end" title="{count} batches of {i + 1}">
					<div
						class="rounded-t-sm {i + 1 === b.max_batch ? 'bg-[var(--cat-3)]' : 'bg-[var(--accent)]'}"
						style="height: {(100 * count) / peak}%; min-height: {count ? 2 : 0}px"
					></div>
				</div>
			{/each}
		</div>
		<div class="mt-1 flex justify-between text-[10px] text-[var(--text-muted)] tabular-nums">
			<span>1</span><span>batch size</span><span>{b.max_batch}</span>
		</div>
		<dl class="mt-4 grid grid-cols-2 gap-x-6 gap-y-2 text-xs tabular-nums sm:grid-cols-4">
			{#each [['Batches', b.batches.toLocaleString()], ['Avg size', `${b.avg_size.toFixed(1)} of ${b.max_batch}`], ['Full', `${b.full_pct.toFixed(0)}%`], ['Max wait', `${b.max_wait_ms} ms`], ['Fill p50', `${b.fill_p50_ms.toFixed(2)} ms`], ['Fill p99', `${b.fill_p99_ms.toFixed(2)} ms`], ['Lane wait p50', `${b.lane_wait_p50_ms.toFixed(2)} ms`], ['Lane wait p99', `${b.lane_wait_p99_ms.toFixed(2)} ms`]] as [k, v]}
				<div>
					<dt class="text-[var(--text-muted)]">{k}</dt>
					<dd class="text-sm text-[var(--fg)]">{v}</dd>
				</div>
			{/each}
		</dl>
	{/if}
</Panel>
