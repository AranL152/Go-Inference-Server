<script lang="ts">
	import { onMount } from 'svelte';
	import Panel from './Panel.svelte';
	import CaptureButton from './CaptureButton.svelte';
	import CaptureFooter from './CaptureFooter.svelte';
	import { latestCapture, startCapture, type StageTime, type TraceSummary } from '$lib/stats';

	let { data = null, live = true }: { data?: TraceSummary | null; live?: boolean } = $props();
	const SECONDS = 2;
	let captured = $state<TraceSummary | null>(null);
	const t = $derived(live ? captured : data);
	let busy = $state(false);
	let error = $state('');

	const parts = [
		{ key: 'running_pct', label: 'Running Go code', css: '--accent' },
		{ key: 'syscall_pct', label: 'In C or a syscall', css: '--cat-3' },
		{ key: 'runnable_pct', label: 'Waiting for a CPU', css: '--cat-4' },
		{ key: 'blocked_pct', label: 'Blocked', css: '--border' }
	] as const;

	async function capture() {
		busy = true;
		error = '';
		try {
			captured = await startCapture<TraceSummary>('trace', SECONDS);
		} catch (e) {
			error = (e as Error).message;
		} finally {
			busy = false;
		}
	}

	onMount(async () => {
		if (live) captured = await latestCapture<TraceSummary>('trace');
	});

	const stages = $derived(t?.stages ?? []);
	// stages with almost no goroutine time draw misleading full bars
	const shown = $derived(stages.filter((s) => s.seconds / (t?.seconds || 1) >= 0.05));
	const by = (name: string) => stages.find((s) => s.stage === name);
	const avgBusy = (s: StageTime) => s.seconds / (t?.seconds || 1);
</script>

<Panel
	title="Goroutine time by stage"
	sub="Go execution trace: where each stage's goroutines spent their time"
>
	{#snippet actions()}{#if live}<CaptureButton
				{busy}
				seconds={SECONDS}
				onclick={capture}
			/>{/if}{/snippet}
	{#if error}<p class="mb-3 text-sm text-[var(--status-critical)]">{error}</p>{/if}
	{#if !t}
		<p class="text-sm text-[var(--text-muted)]">
			{live
				? 'No trace yet. Capture one while the server is under load.'
				: 'The run ended before a trace could be taken.'}
		</p>
	{:else}
		<div class="space-y-3">
			{#each shown as s}
				<div>
					<div class="mb-0.5 flex flex-wrap justify-between gap-x-2 text-xs">
						<span class="text-[var(--fg)]"
							>{s.stage}
							<span class="text-[var(--text-muted)]"
								>· {s.goroutines} goroutines, {avgBusy(s).toFixed(1)} on average</span
							></span
						>
						<span class="text-[var(--text-muted)] tabular-nums"
							>CPU wait p99 {s.sched_p99_us >= 1000
								? `${(s.sched_p99_us / 1000).toFixed(1)} ms`
								: `${s.sched_p99_us.toFixed(0)} µs`}</span
						>
					</div>
					<div class="flex h-2.5 overflow-hidden rounded-full bg-[var(--bg)]">
						{#each parts as p}
							<div
								style="width: {s[p.key]}%; background: var({p.css})"
								title="{p.label} {s[p.key].toFixed(1)}%"
							></div>
						{/each}
					</div>
					{#if s.waits?.length && s.blocked_pct > 5}
						<div class="mt-0.5 text-[11px] text-[var(--text-muted)]">
							blocked on {s.waits.map((w) => `${w.reason} ${w.pct.toFixed(0)}%`).join(', ')}
						</div>
					{/if}
				</div>
			{/each}
		</div>
		<div class="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-[var(--text-muted)]">
			{#each parts as p}
				<span class="flex items-center gap-1.5"
					><span class="inline-block h-2 w-3 rounded-sm" style="background: var({p.css})"
					></span>{p.label}</span
				>
			{/each}
		</div>
		<CaptureFooter file={t.file} taken={t.taken} command="go tool trace profiles/{t.file}" />
	{/if}
</Panel>
