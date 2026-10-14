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
	<p class="py-16 text-center text-sm text-[var(--text-muted)]">Run a detection job first.</p>
{:else}
	<div class="space-y-8">
		<div class="flex flex-wrap items-end justify-between gap-4">
			<div>
				<label for="job" class="mb-2 block text-xs tracking-wider text-[var(--text-muted)] uppercase"
					>Job</label
				>
				<select
					id="job"
					value={id}
					onchange={(e) => {
						cls = '';
						onSelectJob(e.currentTarget.value);
					}}
					class="border border-[var(--border)] bg-[var(--surface)] px-3 py-2 font-mono text-sm text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
				>
					{#each jobs as j}
						<option value={j.id}>{j.id.slice(0, 8)} · {j.source}</option>
					{/each}
				</select>
			</div>
			{#if job}
				<div class="flex gap-10 text-right">
					{#each [['Images', `${job.done.toLocaleString()} / ${job.total.toLocaleString()}`], ['Objects', job.objects.toLocaleString()], ['Throughput', `${Math.round(job.images_per_sec).toLocaleString()} img/s`], ['Status', job.status]] as [k, v]}
						<div>
							<div class="mb-1 text-xs tracking-wider text-[var(--text-muted)] uppercase">{k}</div>
							<div class="text-xl font-semibold text-[var(--fg)] tabular-nums">{v}</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<div class="grid grid-cols-[220px_1fr] gap-8">
			<nav class="self-start border border-[var(--border)] bg-[var(--surface)] p-4">
				<div class="mb-3 text-xs tracking-wider text-[var(--text-muted)] uppercase">Categories</div>
				{#each [{ class: '', images: job?.done ?? 0, objects: 0, label: 'All images' }, ...categories.map((c) => ({ ...c, label: c.class })), { class: '_none', images: noObjects, objects: 0, label: 'No objects' }] as c}
					<button
						onclick={() => (cls = c.class)}
						class="block w-full py-1.5 text-left text-sm {cls === c.class
							? 'text-[var(--fg)]'
							: 'text-[var(--text-muted)] hover:text-[var(--fg)]'}"
					>
						<div class="flex justify-between gap-2">
							<span class="truncate {cls === c.class ? 'font-semibold' : ''}">{c.label}</span>
							<span class="tabular-nums">{c.images.toLocaleString()}</span>
						</div>
						{#if c.class && c.class !== '_none'}
							<div class="mt-1 h-1 bg-[var(--border)]">
								<div
									class="h-1 bg-[var(--series)]"
									style="width: {(100 * c.images) / maxImages}%"
								></div>
							</div>
						{/if}
					</button>
				{/each}
			</nav>

			<div>
				<div class="mb-4 text-xs text-[var(--text-muted)] tabular-nums">
					{total.toLocaleString()} images{cls && cls !== '_none' ? ` containing ${cls}` : ''}
				</div>
				<div class="columns-3 gap-4">
					{#each items as item (item.index)}
						<button
							onclick={() => show(item)}
							class="mb-4 block w-full break-inside-avoid border border-[var(--border)] bg-[var(--surface)] text-left hover:border-[var(--accent)]"
						>
							<DetectionImage src={api.imageURL(id, item.index)} {item} highlight={cls} labels={false} />
							<div class="flex justify-between gap-2 px-2 py-1.5 text-xs">
								<span class="truncate font-mono text-[var(--text-muted)]">{item.path}</span>
								<span class="text-[var(--fg)] tabular-nums">{item.detections.length}</span>
							</div>
						</button>
					{/each}
				</div>
				{#if items.length < total}
					<button
						onclick={() => loadPage(false)}
						disabled={loading}
						class="mt-4 border border-[var(--border)] px-4 py-2 text-xs tracking-wider text-[var(--fg)] uppercase hover:border-[var(--accent)] disabled:opacity-30"
					>
						{loading ? 'Loading...' : `Load more (${(total - items.length).toLocaleString()} left)`}
					</button>
				{/if}
			</div>
		</div>
	</div>
{/if}
