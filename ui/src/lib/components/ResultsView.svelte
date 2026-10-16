<script lang="ts">
	import { getContext, untrack } from 'svelte';
	import type { Context } from 'svelte-simple-modal';
	import { api, type Category, type Job, type ResultItem } from '$lib/api';
	import DetectionImage from './DetectionImage.svelte';
	import ImageDetails from './ImageDetails.svelte';

	interface Props {
		jobs: Job[];
		jobId: string;
		onSelectJob: (id: string) => void;
	}
	let { jobs, jobId, onSelectJob }: Props = $props();
	const { open } = getContext<Context>('simple-modal');

	const PAGE = 48;
	let id = $derived(jobId || jobs[0]?.id || '');
	let job = $state<Job | null>(null);
	let categories = $state<Category[]>([]);
	let noObjects = $state(0);
	let cls = $state('');
	let items = $state<ResultItem[]>([]);
	let total = $state(0);
	let loading = $state(false);

	async function loadSummary() {
		if (!id) return;
		const r = await api.job(id);
		job = r.job;
		categories = r.categories;
		noObjects = r.no_objects;
	}

	async function loadPage(reset: boolean) {
		if (!id) return;
		loading = true;
		const r = await api.results(id, cls, reset ? 0 : items.length, PAGE);
		items = reset ? r.items : [...items, ...r.items];
		total = r.total;
		loading = false;
	}

	// reload when the job or class changes and poll the summary while running
	$effect(() => {
		id;
		cls;
		untrack(() => {
			loadSummary();
			loadPage(true);
		});
	});
	$effect(() => {
		if (job?.status !== 'running') return;
		const t = setInterval(() => {
			loadSummary();
			if (items.length < PAGE) loadPage(true);
		}, 1000);
		return () => clearInterval(t);
	});

	let maxImages = $derived(Math.max(1, ...categories.map((c) => c.images)));

	function show(item: ResultItem) {
		// @ts-expect-error svelte-simple-modal types predate Svelte 5 components
		open(ImageDetails, { src: api.imageURL(id, item.index), item, highlight: cls });
	}
</script>

{#if !id}
	<div class="rounded-2xl border border-dashed border-[var(--border)] py-16 text-center">
		<p class="text-sm text-[var(--text-muted)]">Run a detection job first.</p>
	</div>
{:else}
	<div class="space-y-6">
		<section class="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
			<div class="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--border)] px-5 py-3">
				<div class="flex min-w-0 items-center gap-3">
					<label for="job" class="text-sm font-semibold text-[var(--fg)]">Job</label>
					<select
						id="job"
						value={id}
						onchange={(e) => {
							cls = '';
							onSelectJob(e.currentTarget.value);
						}}
						class="max-w-full min-w-0 rounded-lg border border-[var(--border)] bg-[var(--bg)] py-1.5 pr-8 pl-3 font-mono text-sm text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
					>
						{#each jobs as j}
							<option value={j.id}>{j.id.slice(0, 8)} · {j.source}</option>
						{/each}
					</select>
				</div>
				{#if job}
					<span
						class="rounded-full px-3 py-1 text-xs font-medium {job.status === 'running'
							? 'bg-[var(--accent-soft)] text-[var(--accent)]'
							: 'bg-[var(--bg)] text-[var(--text-muted)]'}">{job.status}</span
					>
				{/if}
			</div>
			{#if job}
				<div class="grid grid-cols-2 divide-[var(--border)] md:grid-cols-3 md:divide-x">
					{#each [['Images', `${job.done.toLocaleString()} / ${job.total.toLocaleString()}`], ['Objects', job.objects.toLocaleString()], ['Throughput', `${Math.round(job.images_per_sec).toLocaleString()} img/s`]] as [k, v]}
						<div class="px-5 py-4">
							<div class="text-xs text-[var(--text-muted)]">{k}</div>
							<div class="mt-1 text-2xl font-semibold text-[var(--fg)] tabular-nums">{v}</div>
						</div>
					{/each}
				</div>
			{/if}
		</section>

		<div>
			<div class="mb-2 text-xs font-semibold text-[var(--text-muted)]">Categories</div>
			<div class="flex flex-wrap gap-2">
				{#each [{ class: '', images: job?.done ?? 0, objects: 0, label: 'All images' }, ...categories.map((c) => ({ ...c, label: c.class })), { class: '_none', images: noObjects, objects: 0, label: 'No objects' }] as c}
					<button
						onclick={() => (cls = c.class)}
						class="relative overflow-hidden rounded-full border px-3 py-1 text-sm {cls === c.class
							? 'border-[var(--accent)] bg-[var(--accent)] text-white'
							: 'border-[var(--border)] bg-[var(--surface)] text-[var(--fg)] hover:border-[var(--accent)]'}"
					>
						{#if c.class && c.class !== '_none' && cls !== c.class}
							<span
								class="absolute inset-y-0 left-0 bg-[var(--accent-soft)]"
								style="width: {(100 * c.images) / maxImages}%"
							></span>
						{/if}
						<span class="relative">{c.label}</span>
						<span class="relative ml-1 tabular-nums {cls === c.class ? 'opacity-80' : 'text-[var(--text-muted)]'}"
							>{c.images.toLocaleString()}</span
						>
					</button>
				{/each}
			</div>
		</div>

		<div>
			<div class="mb-3 text-xs text-[var(--text-muted)] tabular-nums">
				{total.toLocaleString()} images{cls && cls !== '_none' ? ` containing ${cls}` : ''}
			</div>
			<div class="columns-2 gap-4 md:columns-3 xl:columns-4">
				{#each items as item (item.index)}
					<button
						onclick={() => show(item)}
						class="group relative mb-4 block w-full break-inside-avoid overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)] text-left shadow-sm transition-shadow hover:shadow-md"
					>
						<DetectionImage src={api.imageURL(id, item.index)} {item} highlight={cls} labels={false} />
						<span
							class="absolute top-2 right-2 rounded-full bg-black/60 px-2 py-0.5 text-xs text-white tabular-nums backdrop-blur"
							>{item.detections.length}</span
						>
						<div class="truncate px-3 py-2 font-mono text-xs text-[var(--text-muted)] group-hover:text-[var(--fg)]">
							{item.path}
						</div>
					</button>
				{/each}
			</div>
			{#if items.length < total}
				<div class="mt-2 flex justify-center">
					<button
						onclick={() => loadPage(false)}
						disabled={loading}
						class="rounded-full border border-[var(--border)] bg-[var(--surface)] px-5 py-2 text-sm text-[var(--fg)] hover:border-[var(--accent)] disabled:opacity-30"
					>
						{loading ? 'Loading...' : `Load more (${(total - items.length).toLocaleString()} left)`}
					</button>
				</div>
			{/if}
		</div>
	</div>
{/if}
