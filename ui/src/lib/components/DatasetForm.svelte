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

<div class="border border-[var(--border)] bg-[var(--surface)] p-8">
	<h2 class="mb-6 text-lg font-semibold text-[var(--fg)]">Detect Objects</h2>

	<form onsubmit={submit} class="space-y-6">
		<div>
			<div class="mb-2 flex gap-6 text-xs tracking-wider uppercase">
				<button
					type="button"
					class="tab pb-1 {mode === 'path' ? 'active text-[var(--fg)]' : 'text-[var(--text-muted)]'}"
					onclick={() => (mode = 'path')}>Server folder</button
				>
				<button
					type="button"
					class="tab pb-1 {mode === 'upload'
						? 'active text-[var(--fg)]'
						: 'text-[var(--text-muted)]'}"
					onclick={() => (mode = 'upload')}>Upload images</button
				>
			</div>

			{#if mode === 'path'}
				<label for="path" class="mb-2 block text-xs text-[var(--text-muted)]">
					Folder of JPEGs on the server, searched recursively
				</label>
				<input
					id="path"
					type="text"
					bind:value={path}
					placeholder="/home/you/datasets/val2017"
					class="w-full border border-[var(--border)] bg-[var(--surface)] px-3 py-2 font-mono text-sm text-[var(--fg)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
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
					class="border border-dashed p-8 text-center text-sm {dragging
						? 'border-[var(--accent)] text-[var(--fg)]'
						: 'border-[var(--border)] text-[var(--text-muted)]'}"
				>
					<p>Drop JPEG images here</p>
					<div class="mt-4 flex justify-center gap-3">
						<label
							class="cursor-pointer border border-[var(--border)] px-3 py-1 text-xs tracking-wider text-[var(--fg)] uppercase hover:border-[var(--accent)]"
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
							class="cursor-pointer border border-[var(--border)] px-3 py-1 text-xs tracking-wider text-[var(--fg)] uppercase hover:border-[var(--accent)]"
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
			<label
				for="classes"
				class="mb-2 block text-xs tracking-wider text-[var(--text-muted)] uppercase"
			>
				Classes to sort into
			</label>
			<textarea
				id="classes"
				bind:value={classesText}
				rows="2"
				placeholder="person, car, dog   (leave empty for all {labels.length} classes)"
				class="w-full border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--fg)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
			></textarea>
			<div class="mt-2 flex flex-wrap items-center gap-2 text-xs">
				{#each Object.entries(presets) as [name, list]}
					<button
						type="button"
						class="border border-[var(--border)] px-2 py-0.5 text-[var(--text-muted)] hover:border-[var(--accent)] hover:text-[var(--fg)]"
						onclick={() => (classesText = list.join(', '))}>{name}</button
					>
				{/each}
				{#each classes as c}
					<span
						class="border px-2 py-0.5 {known.has(c)
							? 'border-[var(--fg)] text-[var(--fg)]'
							: 'border-red-500 text-red-600'}">{c}</span
					>
				{/each}
			</div>
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
				<div class="h-1 w-full bg-[var(--border)]">
					<div class="h-1 bg-[var(--accent)]" style="width: {uploadFrac * 100}%"></div>
				</div>
			</div>
		{/if}

		{#if error}
			<p class="text-sm text-red-600">{error}</p>
		{/if}

		<button
			type="submit"
			disabled={!canSubmit}
			class="bg-[var(--fg)] px-6 py-2 text-[var(--bg)] transition-opacity hover:opacity-80 disabled:cursor-not-allowed disabled:opacity-30"
		>
			{submitting ? 'Starting...' : 'Run detection'}
		</button>
	</form>
</div>
