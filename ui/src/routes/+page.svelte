<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { api, type Config, type Job } from '$lib/api';
	import DatasetForm from '$lib/components/DatasetForm.svelte';
	import JobList from '$lib/components/JobList.svelte';
	import ResultsView from '$lib/components/ResultsView.svelte';
	import PerformanceView from '$lib/components/PerformanceView.svelte';
	import PipelineTimeline from '$lib/components/PipelineTimeline.svelte';

	type Tab = 'detect' | 'results' | 'performance';
	const heading: Record<Tab, { crumb: string; title: string }> = {
		detect: { crumb: 'Detection', title: 'Run detection' },
		results: { crumb: 'Detection', title: 'Results' },
		performance: { crumb: 'Monitoring', title: 'Inference server dashboard' }
	};

	let config = $state<Config | null>(null);
	let jobs = $state<Job[]>([]);
	let error = $state('');

	let activeTab = $derived<Tab>((page.url.searchParams.get('tab') as Tab) || 'detect');
	let selectedJob = $derived(page.url.searchParams.get('job') || '');
	let h = $derived(heading[activeTab] ?? heading.detect);

	function setTab(tab: Tab, job = '') {
		goto(`?tab=${tab}${job ? `&job=${job}` : ''}`, { keepFocus: true, noScroll: true });
	}

	async function refresh() {
		try {
			jobs = await api.jobs();
			error = '';
		} catch {
			error = 'Cannot reach the inference server.';
		}
	}

	async function cancel(id: string) {
		await api.cancel(id).catch(() => {});
		refresh();
	}

	onMount(() => {
		api
			.config()
			.then((c) => (config = c))
			.catch(() => (error = 'Cannot reach the inference server.'));
		refresh();
		const t = setInterval(refresh, 1000);
		return () => clearInterval(t);
	});
</script>

<div class="mx-auto max-w-[1400px]">
	<div class="mb-6">
		<div class="mb-2 flex items-center gap-1.5 text-xs text-[var(--text-muted)]">
			<span>Inference Server</span><span>›</span><span>{h.crumb}</span><span>›</span>
			<span class="text-[var(--fg)]">{h.title}</span>
		</div>
		<h1 class="text-2xl font-semibold tracking-tight text-[var(--fg)]">{h.title}</h1>
	</div>

	{#if error}
		<div class="mb-6 flex items-center gap-3 rounded-xl border border-[var(--status-critical)] bg-[var(--surface)] px-4 py-3">
			<span class="font-bold text-[var(--status-critical)]">✕</span>
			<p class="text-sm text-[var(--fg)]">{error}</p>
		</div>
	{/if}

	{#if activeTab === 'detect'}
		<div class="grid items-start gap-6 xl:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
			<div class="space-y-6">
				{#if config}
					<section class="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
						<div class="border-b border-[var(--border)] px-5 py-3">
							<h2 class="text-sm font-semibold text-[var(--fg)]">Model</h2>
						</div>
						<dl class="grid grid-cols-2 gap-x-6 gap-y-3 px-5 py-4 text-sm sm:grid-cols-3">
							{#each [['Model', config.model], ['Classes', `${config.labels.length}`], ['Max batch', `${config.max_batch}`], ['Confidence', `${config.conf}`], ['IoU', `${config.iou}`], ['Type', config.detector ? 'Detector' : 'Classifier']] as [k, v]}
								<div class="min-w-0">
									<dt class="text-xs text-[var(--text-muted)]">{k}</dt>
									<dd class="truncate font-mono text-[var(--fg)]" title={v}>{v}</dd>
								</div>
							{/each}
						</dl>
					</section>
					{#if config.detector}
						<DatasetForm labels={config.labels} onStarted={refresh} />
					{:else}
						<p class="text-sm text-[var(--text-muted)]">
							{config.model} is a classifier. Start the server with a detector model to run dataset jobs.
						</p>
					{/if}
				{/if}
			</div>
			<div class="min-w-0 space-y-6">
				<section class="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
					<div class="border-b border-[var(--border)] px-5 py-3">
						<h2 class="text-sm font-semibold text-[var(--fg)]">Live pipeline</h2>
					</div>
					<div class="px-5 py-4"><PipelineTimeline compact /></div>
				</section>
				<JobList {jobs} onCancel={cancel} onViewResults={(id) => setTab('results', id)} />
			</div>
		</div>
	{:else if activeTab === 'results'}
		<ResultsView {jobs} jobId={selectedJob} onSelectJob={(id) => setTab('results', id)} />
	{:else}
		<PerformanceView />
	{/if}
</div>
