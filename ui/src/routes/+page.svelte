<script lang="ts">
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { api, type Config, type Job } from '$lib/api';
	import DatasetForm from '$lib/components/DatasetForm.svelte';
	import JobList from '$lib/components/JobList.svelte';
	import ResultsView from '$lib/components/ResultsView.svelte';

	type Tab = 'detect' | 'results' | 'performance';
	const tabs: Tab[] = ['detect', 'results', 'performance'];

	let config = $state<Config | null>(null);
	let jobs = $state<Job[]>([]);
	let error = $state('');

	let activeTab = $derived<Tab>((page.url.searchParams.get('tab') as Tab) || 'detect');
	let selectedJob = $derived(page.url.searchParams.get('job') || '');

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

<div>
	<div class="mb-8 border-b border-[var(--border)]">
		<div class="flex gap-8">
			{#each tabs as tab}
				<button
					onclick={() => setTab(tab)}
					class="tab pb-3 text-sm tracking-wider uppercase {activeTab === tab
						? 'active text-[var(--fg)]'
						: 'text-[var(--text-muted)]'}"
				>
					{tab}
				</button>
			{/each}
		</div>
	</div>

	{#if error}
		<div class="mb-8 border border-red-500 p-4">
			<p class="text-sm text-red-600">{error}</p>
		</div>
	{/if}

	{#if activeTab === 'detect'}
		<div class="space-y-8">
			{#if config}
				{#if config.detector}
					<DatasetForm labels={config.labels} onStarted={refresh} />
				{:else}
					<p class="text-sm text-[var(--text-muted)]">
						{config.model} is a classifier. Start the server with a detector model to run dataset jobs.
					</p>
				{/if}
			{/if}
			<JobList {jobs} onCancel={cancel} onViewResults={(id) => setTab('results', id)} />
		</div>
	{:else if activeTab === 'results'}
		<ResultsView {jobs} jobId={selectedJob} onSelectJob={(id) => setTab('results', id)} />
	{:else}
		<p class="text-sm text-[var(--text-muted)]">Performance</p>
	{/if}
</div>
