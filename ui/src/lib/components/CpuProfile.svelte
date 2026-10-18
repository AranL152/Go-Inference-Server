<script lang="ts">
	import { onMount } from 'svelte';
	import Panel from './Panel.svelte';
	import CaptureButton from './CaptureButton.svelte';
	import CaptureFooter from './CaptureFooter.svelte';
	import { latestCapture, shortFunc, startCapture, type CpuSummary } from '$lib/stats';

	const SECONDS = 3;
	let p = $state<CpuSummary | null>(null);
	let busy = $state(false);
	let error = $state('');

	// cgo time is spent in C libraries the Go profiler cannot see into
	const label = (f: string) => (f === 'runtime.cgocall' ? 'runtime.cgocall (inside C code)' : shortFunc(f));

	async function capture() {
		busy = true;
		error = '';
		try {
			p = await startCapture<CpuSummary>('cpu', SECONDS);
		} catch (e) {
			error = (e as Error).message;
		} finally {
			busy = false;
		}
	}

	onMount(async () => {
		p = await latestCapture<CpuSummary>('cpu');
	});

	const stages = $derived(p?.stages ?? []);
	const top = $derived(p?.top ?? []);
	const finding = $derived.by(() => {
		if (!p || !stages.length || !top.length) return '';
		const s = stages[0];
		return `${s.stage} takes ${s.pct.toFixed(0)}% of server CPU (${s.cores.toFixed(1)} of ${p.cores.toFixed(1)} busy cores). Hottest function: ${label(top[0].func)} at ${top[0].pct.toFixed(0)}%.`;
	});
</script>

<Panel title="CPU profile" sub="pprof samples split by pipeline stage label" {finding}>
	{#snippet actions()}<CaptureButton {busy} seconds={SECONDS} onclick={capture} />{/snippet}
	{#if error}<p class="mb-3 text-sm text-[var(--status-critical)]">{error}</p>{/if}
	{#if !p}
		<p class="text-sm text-[var(--text-muted)]">No profile yet. Capture one while the server is under load.</p>
	{:else}
		<div class="space-y-2">
			{#each stages as s}
				<div>
					<div class="mb-0.5 flex justify-between gap-2 text-xs">
						<span class="text-[var(--fg)]">{s.stage}</span>
						<span class="text-[var(--text-muted)] tabular-nums">{s.pct.toFixed(1)}% · {s.cores.toFixed(2)} cores</span>
					</div>
					<div class="h-2 overflow-hidden rounded-full bg-[var(--bg)]">
						<div class="h-full rounded-full bg-[var(--accent)]" style="width: {s.pct}%"></div>
					</div>
					{#if s.top?.length}
						<div class="mt-0.5 truncate font-mono text-[11px] text-[var(--text-muted)]">
							{s.top.map((f) => `${label(f.func)} ${f.pct.toFixed(1)}%`).join(' · ')}
						</div>
					{/if}
				</div>
			{/each}
		</div>
		<table class="mt-4 w-full text-xs tabular-nums">
			<thead>
				<tr class="border-b border-[var(--border)] text-left text-[var(--text-muted)]">
					<th class="py-1 font-normal">Function (flat)</th>
					<th class="py-1 text-right font-normal">CPU</th>
				</tr>
			</thead>
			<tbody>
				{#each top.slice(0, 8) as f}
					<tr class="border-b border-[var(--border)]">
						<td class="max-w-0 truncate py-1 pr-2 font-mono text-[var(--fg)]" title={f.func}>{label(f.func)}</td>
						<td class="py-1 text-right text-[var(--fg)]">{f.pct.toFixed(1)}%</td>
					</tr>
				{/each}
			</tbody>
		</table>
		<CaptureFooter file={p.file} taken={p.taken} command="go tool pprof -http=: profiles/{p.file}" />
	{/if}
</Panel>
