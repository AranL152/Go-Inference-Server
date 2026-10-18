<script lang="ts">
	import { api, upload, parseClasses, type Job } from '$lib/api';

	interface Props {
		labels: string[];
		onStarted: (job: Job) => void;
	}
	let { labels, onStarted }: Props = $props();

	const presets: Record<string, string[]> = {
		'People & vehicles': ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'train'],
		Animals: ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
		Kitchen: ['bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'oven', 'sink']
	};

	let mode = $state<'path' | 'upload'>('path');
	let path = $state('');
	let files = $state<File[]>([]);
	let classesText = $state('');
	let dragging = $state(false);
	let submitting = $state(false);
	let uploadFrac = $state(0);
	let error = $state('');

	let known = $derived(new Set(labels.map((l) => l.toLowerCase())));
	let classes = $derived(parseClasses(classesText));
	let unknown = $derived(classes.filter((c) => !known.has(c)));

	function addFiles(list: FileList | null) {
		if (!list) return;
		const jpegs = [...list].filter((f) => /\.jpe?g$/i.test(f.name));
		files = [...files, ...jpegs];
	}

	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragging = false;
		addFiles(e.dataTransfer?.files ?? null);
	}

	async function submit(e: Event) {
		e.preventDefault();
		error = '';
		submitting = true;
		uploadFrac = 0;
		try {
			const job =
				mode === 'path'
					? await api.startPath(path.trim(), classes)
					: await upload(files, classes, (f) => (uploadFrac = f));
			files = [];
			onStarted(job);
		} catch (err) {
			error = (err as Error).message;
		} finally {
			submitting = false;
		}
	}

	let canSubmit = $derived(
		!submitting && unknown.length === 0 && (mode === 'path' ? path.trim() !== '' : files.length > 0)
	);
	let totalMB = $derived(files.reduce((n, f) => n + f.size, 0) / 1e6);
</script>

{#snippet step(n: number, title: string)}
	<div class="mb-3 flex items-center gap-3">
		<span
			class="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--accent-soft)] text-xs font-semibold text-[var(--accent)]"
			>{n}</span
		>
		<h3 class="text-sm font-semibold text-[var(--fg)]">{title}</h3>
	</div>
{/snippet}

<section class="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-sm">
	<div class="border-b border-[var(--border)] px-5 py-3">
		<h2 class="text-sm font-semibold text-[var(--fg)]">Detect Objects</h2>
	</div>

	<form onsubmit={submit}>
		<div class="space-y-6 px-5 py-5">
			<div>
				{@render step(1, 'Source')}
				<div class="mb-3 grid grid-cols-2 gap-3">
					{#each [{ m: 'path', label: 'Server folder', d: 'M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z' }, { m: 'upload', label: 'Upload images', d: 'M12 16V4M7 9l5-5 5 5M4 20h16' }] as o}
						<button
							type="button"
							onclick={() => (mode = o.m as 'path' | 'upload')}
							class="flex items-center gap-3 rounded-xl border px-4 py-3 text-left text-sm {mode === o.m
								? 'border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--fg)]'
								: 'border-[var(--border)] text-[var(--text-muted)] hover:border-[var(--text-muted)]'}"
						>
							<svg viewBox="0 0 24 24" class="h-5 w-5 shrink-0 {mode === o.m ? 'text-[var(--accent)]' : ''}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d={o.d} /></svg>
							{o.label}
						</button>
					{/each}
				</div>

				{#if mode === 'path'}
					<label for="path" class="mb-2 block text-xs text-[var(--text-muted)]">
						Folder of JPEGs on the server, searched recursively
					</label>
					<input
						id="path"
						type="text"
						bind:value={path}
						placeholder="Enter a folder path"
						class="w-full rounded-lg border border-[var(--border)] bg-[var(--bg)] px-3 py-2 font-mono text-sm text-[var(--fg)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
					/>
				{:else}
					<div
						role="region"
						aria-label="Drop images here"
						ondragover={(e) => {
							e.preventDefault();
							dragging = true;
						}}
						ondragleave={() => (dragging = false)}
						ondrop={onDrop}
						class="rounded-xl border-2 border-dashed px-6 py-8 text-center text-sm {dragging
							? 'border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--fg)]'
							: 'border-[var(--border)] bg-[var(--bg)] text-[var(--text-muted)]'}"
					>
						<p>Drop JPEG images here</p>
						<div class="mt-4 flex justify-center gap-3">
							<label
								class="cursor-pointer rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--fg)] hover:border-[var(--accent)]"
							>
								Choose files
								<input
									type="file"
									accept="image/jpeg"
									multiple
									class="hidden"
									onchange={(e) => addFiles(e.currentTarget.files)}
								/>
							</label>
							<label
								class="cursor-pointer rounded-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1 text-xs text-[var(--fg)] hover:border-[var(--accent)]"
							>
								Choose folder
								<input
									type="file"
									webkitdirectory
									class="hidden"
									onchange={(e) => addFiles(e.currentTarget.files)}
								/>
							</label>
						</div>
						{#if files.length > 0}
							<p class="mt-4 text-[var(--fg)] tabular-nums">
								{files.length} images · {totalMB.toFixed(1)} MB
								<button
									type="button"
									class="ml-2 text-xs text-[var(--text-muted)] underline"
									onclick={() => (files = [])}>clear</button
								>
							</p>
						{/if}
					</div>
				{/if}
			</div>

			<div>
				{@render step(2, 'Classes to sort into')}
				<div class="mb-2 flex flex-wrap gap-2 text-xs">
					{#each Object.entries(presets) as [name, list]}
						<button
							type="button"
							class="rounded-full bg-[var(--bg)] px-3 py-1 text-[var(--fg)] hover:bg-[var(--accent-soft)] hover:text-[var(--accent)]"
							onclick={() => (classesText = list.join(', '))}>+ {name}</button
						>
					{/each}
				</div>
				<textarea
					id="classes"
					bind:value={classesText}
					rows="2"
					placeholder="person, car, dog   (leave empty for all {labels.length} classes)"
					class="w-full rounded-lg border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm text-[var(--fg)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
				></textarea>
				{#if classes.length > 0}
					<div class="mt-2 flex flex-wrap gap-1.5 text-xs">
						{#each classes as c}
							<span
								class="rounded-md px-2 py-0.5 {known.has(c)
									? 'bg-[var(--accent-soft)] text-[var(--accent)]'
									: 'bg-red-500/10 text-red-600'}">{c}</span
							>
						{/each}
					</div>
				{/if}
				{#if unknown.length > 0}
					<p class="mt-2 text-xs text-red-600">
						Not in this model: {unknown.join(', ')}. The detector knows the {labels.length} COCO classes.
					</p>
				{/if}
			</div>

			{#if submitting && mode === 'upload'}
				<div>
					<div class="mb-1 text-xs text-[var(--text-muted)] tabular-nums">
						Uploading {Math.round(uploadFrac * 100)}%
					</div>
					<div class="h-1.5 w-full overflow-hidden rounded-full bg-[var(--border)]">
						<div class="h-full rounded-full bg-[var(--accent)]" style="width: {uploadFrac * 100}%"></div>
					</div>
				</div>
			{/if}

			{#if error}
				<p class="text-sm text-red-600">{error}</p>
			{/if}
		</div>

		<div class="flex items-center justify-between gap-4 border-t border-[var(--border)] bg-[var(--bg)] px-5 py-3">
			<span class="text-xs text-[var(--text-muted)]">
				{classes.length ? `${classes.length} classes` : `all ${labels.length} classes`}
			</span>
			<button
				type="submit"
				disabled={!canSubmit}
				class="rounded-full bg-[var(--accent)] px-5 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
			>
				{submitting ? 'Starting...' : 'Run detection'}
			</button>
		</div>
	</form>
</section>
