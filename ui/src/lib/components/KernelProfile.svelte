<script lang="ts">
	import { onMount } from 'svelte';

	interface Kernel {
		name: string;
		calls: number;
		ms_per_batch: number;
		share_pct: number;
		sm_pct: number;
		memory_pct: number;
		occupancy_pct: number;
		tensor_pct: number;
	}
	interface Profile {
		model: string;
		kernels_per_batch: number;
		gpu_ms_per_batch: number;
		sm_throughput_pct: number;
		memory_throughput_pct: number;
		occupancy_pct: number;
		tensor_active_pct: number;
		top_kernels: Kernel[];
	}

	let { model = '' }: { model?: string } = $props();
	let profiles = $state<Profile[]>([]);
	let chosen = $state('');

	onMount(async () => {
		const r = await fetch('/api/profiles').then((r) => r.json()).catch(() => ({ profiles: [] }));
		profiles = r.profiles;
	});
	// follow the running model until the user picks another
	const p = $derived(
		profiles.find((x) => x.model === chosen) ?? profiles.find((x) => x.model === model) ?? profiles[0]
	);
</script>

{#if profiles.length === 0}
	<p class="py-6 text-sm text-[var(--text-muted)]">
		No kernel profile yet. Run <code class="font-mono">./profile_kernels.sh MODEL</code> to profile one batch
		with Nsight Compute.
	</p>
{:else if p}
	<div class="mb-4 flex flex-wrap items-center gap-2 text-xs">
		{#each profiles as x}
			<button
				onclick={() => (chosen = x.model)}
				class="border px-2 py-0.5 font-mono {x.model === p?.model
					? 'border-[var(--accent)] text-[var(--fg)]'
					: 'border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--fg)]'}">{x.model}</button
			>
		{/each}
		<span class="text-[var(--text-muted)]">one batch of 16 profiled with Nsight Compute</span>
	</div>
	<div class="mb-5 grid grid-cols-3 gap-4 md:grid-cols-6">
		{#each [['GPU time per batch', `${p.gpu_ms_per_batch.toFixed(2)} ms`], ['Kernels per batch', p.kernels_per_batch], ['SM throughput', `${p.sm_throughput_pct}%`], ['Memory throughput', `${p.memory_throughput_pct}%`], ['Tensor cores active', `${p.tensor_active_pct}%`], ['Achieved occupancy', `${p.occupancy_pct}%`]] as [k, v]}
			<div>
				<div class="text-xs text-[var(--text-muted)]">{k}</div>
				<div class="text-xl font-semibold text-[var(--fg)] tabular-nums">{v}</div>
			</div>
		{/each}
	</div>
	<div class="overflow-x-auto">
		<table class="w-full text-xs">
			<thead>
				<tr class="border-b border-[var(--border)] text-left text-[var(--text-muted)]">
					<th class="py-1.5 pr-3 font-normal">Kernel</th>
					<th class="py-1.5 pr-3 text-right font-normal">Calls</th>
					<th class="py-1.5 pr-3 text-right font-normal">ms per batch</th>
					<th class="w-32 py-1.5 pr-3 font-normal">Share of GPU time</th>
					<th class="py-1.5 pr-3 text-right font-normal">SM</th>
					<th class="py-1.5 pr-3 text-right font-normal">Memory</th>
					<th class="py-1.5 pr-3 text-right font-normal">Tensor</th>
					<th class="py-1.5 text-right font-normal">Occupancy</th>
				</tr>
			</thead>
			<tbody>
				{#each p.top_kernels as k}
					<tr class="border-b border-[var(--border)] text-[var(--fg)] tabular-nums">
						<td class="max-w-80 truncate py-1.5 pr-3 font-mono" title={k.name}>{k.name}</td>
						<td class="py-1.5 pr-3 text-right">{k.calls}</td>
						<td class="py-1.5 pr-3 text-right">{k.ms_per_batch.toFixed(3)}</td>
						<td class="py-1.5 pr-3">
							<div class="flex items-center gap-2">
								<div class="h-1.5 flex-1 bg-[var(--border)]">
									<div class="h-1.5 bg-[var(--series)]" style="width: {k.share_pct}%"></div>
								</div>
								<span class="w-10 text-right">{k.share_pct}%</span>
							</div>
						</td>
						<td class="py-1.5 pr-3 text-right">{k.sm_pct}%</td>
						<td class="py-1.5 pr-3 text-right">{k.memory_pct}%</td>
						<td class="py-1.5 pr-3 text-right">{k.tensor_pct}%</td>
						<td class="py-1.5 text-right">{k.occupancy_pct}%</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}
