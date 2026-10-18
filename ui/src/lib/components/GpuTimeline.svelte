<script lang="ts">
	import { onMount } from 'svelte';
	import Panel from './Panel.svelte';
	import type { NsysSummary } from '$lib/stats';

	let { model }: { model: string } = $props();
	let all = $state<NsysSummary[]>([]);

	onMount(async () => {
		const r = await fetch('/api/profiles?kind=nsys')
			.then((r) => r.json())
			.catch(() => ({ profiles: [] }));
		all = r.profiles ?? [];
	});

	const p = $derived(all.find((x) => x.model === model) ?? all[0]);
	const base = $derived((p?.model ?? model).replace(/\.onnx$/, ''));
	const finding = $derived(
		p
			? `GPU busy ${p.gpu_busy_pct}% of the capture (${p.compute_pct}% kernels, ${p.copy_pct}% copies); ${p.copy_hidden_pct}% of copy time overlaps compute. Inputs reach the GPU at ${p.h2d.gb_per_s} GB/s, ${p.h2d.pinned_pct}% from pinned memory.`
			: ''
	);
	const us = (v: number) => (v >= 1000 ? `${(v / 1000).toFixed(1)} ms` : `${v.toFixed(0)} µs`);
</script>

<Panel title="GPU timeline" sub="Nsight Systems trace of kernels and copies under steady load" {finding}>
	{#if !p}
		<p class="text-sm text-[var(--text-muted)]">No Nsight Systems capture yet. Run</p>
		<pre class="mt-2 overflow-x-auto rounded-lg bg-[var(--bg)] px-3 py-2 font-mono text-xs text-[var(--fg)]">./profile_gpu.sh {model || 'MODEL.onnx'} -device-binding -trt trt_cache</pre>
	{:else}
		<dl class="grid grid-cols-2 gap-x-6 gap-y-3 text-xs tabular-nums sm:grid-cols-4">
			{#each [['GPU busy', `${p.gpu_busy_pct}%`], ['Copies hidden by compute', `${p.copy_hidden_pct}%`], ['Idle gap p50', us(p.idle_gap_p50_us)], ['Idle gap p99', us(p.idle_gap_p99_us)], ['Host to GPU', `${p.h2d.gb_per_s} GB/s · ${p.h2d.avg_kb} KB avg`], ['Pinned share', `${p.h2d.pinned_pct}%`], ['GPU to host', `${p.d2h.gb_per_s} GB/s`], ['Kernels per second', p.kernels_per_s.toLocaleString()], ['Avg kernel', us(p.kernel_avg_us)], ['Kernels under 10 µs', `${p.short_kernel_pct}%`], ['Launch call', us(p.launch_avg_us)], ['Threads blocked on GPU', p.threads_blocked_on_gpu.toFixed(2)]] as [k, v]}
				<div>
					<dt class="text-[var(--text-muted)]">{k}</dt>
					<dd class="text-sm text-[var(--fg)]">{v}</dd>
				</div>
			{/each}
		</dl>
		<table class="mt-4 w-full text-xs tabular-nums">
			<thead>
				<tr class="border-b border-[var(--border)] text-left text-[var(--text-muted)]">
					<th class="py-1 font-normal">Kernel</th>
					<th class="py-1 text-right font-normal">Calls</th>
					<th class="py-1 text-right font-normal">Avg</th>
					<th class="py-1 text-right font-normal">Share</th>
				</tr>
			</thead>
			<tbody>
				{#each p.top_kernels as k}
					<tr class="border-b border-[var(--border)] text-[var(--fg)]">
						<td class="max-w-0 truncate py-1 pr-2 font-mono" title={k.name}>{k.name}</td>
						<td class="py-1 text-right">{k.calls.toLocaleString()}</td>
						<td class="py-1 text-right">{us(k.avg_us)}</td>
						<td class="py-1 text-right">{k.share_pct}%</td>
					</tr>
				{/each}
			</tbody>
		</table>
		<div class="mt-4 border-t border-[var(--border)] pt-3 text-xs text-[var(--text-muted)]">
			{p.seconds} s capture of {p.model} · open in Nsight Systems:
			<span class="ml-1 rounded-md bg-[var(--bg)] px-2 py-1 font-mono text-[var(--fg)] ring-1 ring-[var(--border)]">nsys-ui profiles/{base}.nsys-rep</span>
		</div>
	{/if}
</Panel>
