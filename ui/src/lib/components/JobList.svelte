<script lang="ts">
	import type { Job } from '$lib/api';
	import { formatDate } from '$lib/utils/date';

	interface Props {
		jobs: Job[];
		onCancel: (id: string) => void;
		onViewResults: (id: string) => void;
	}
	let { jobs, onCancel, onViewResults }: Props = $props();

	function statusColor(status: string) {
		switch (status) {
			case 'running':
				return 'border-[var(--accent)] text-[var(--accent)]';
			case 'completed':
				return 'border-[var(--fg)] text-[var(--fg)]';
			default:
				return 'border-[var(--text-muted)] text-[var(--text-muted)]';
		}
	}

	const percent = (j: Job) => (j.total ? Math.round((j.done / j.total) * 100) : 0);
</script>

<div class="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-8 shadow-sm">
	<h2 class="mb-6 text-lg font-semibold text-[var(--fg)]">Jobs</h2>

	{#if jobs.length === 0}
		<p class="py-8 text-center text-sm text-[var(--text-muted)]">
			No jobs yet. Point it at a folder of images above.
		</p>
	{:else}
		<div class="space-y-3">
			{#each jobs as job (job.id)}
				<div
					class="rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-4 shadow-sm transition-colors hover:border-[var(--accent)]"
				>
					<div class="flex items-start justify-between gap-4">
						<div class="min-w-0 flex-1">
							<div class="mb-2 flex items-center gap-3">
								<span
									class="rounded-full border px-2 py-0.5 text-xs tracking-wider uppercase {statusColor(job.status)}"
								>
									{job.status}
								</span>
								<span class="truncate text-xs text-[var(--text-muted)]">
									{job.classes.length ? job.classes.join(', ') : 'all classes'}
								</span>
							</div>
							<div class="truncate font-mono text-sm text-[var(--fg)]">{job.source}</div>

							<div class="mt-3">
								<div
									class="mb-2 flex flex-wrap items-center gap-2 text-xs text-[var(--text-muted)] tabular-nums"
								>
									<span>{percent(job)}%</span>
									<span>•</span>
									<span>{job.done.toLocaleString()} / {job.total.toLocaleString()} images</span>
									<span>•</span>
									<span>{job.objects.toLocaleString()} objects</span>
									<span>•</span>
									<span>{Math.round(job.images_per_sec).toLocaleString()} img/s</span>
									{#if job.failed > 0}
										<span>•</span>
										<span class="text-red-600">{job.failed} failed</span>
									{/if}
								</div>
								<div class="h-1 w-full overflow-hidden rounded-full bg-[var(--border)]">
									<div
										class="h-1 bg-[var(--accent)] transition-all"
										style="width: {percent(job)}%"
									></div>
								</div>
							</div>
						</div>

						<div class="flex shrink-0 flex-col items-end gap-2">
							<div class="flex gap-2">
								{#if job.status === 'running'}
									<button
										onclick={() => onCancel(job.id)}
										class="rounded-full border border-[var(--border)] px-3 py-1 text-xs tracking-wider text-[var(--fg)] uppercase hover:border-red-500"
									>
										Cancel
									</button>
								{/if}
								<button
									onclick={() => onViewResults(job.id)}
									class="rounded-full border border-[var(--border)] px-3 py-1 text-xs tracking-wider text-[var(--fg)] uppercase hover:border-[var(--accent)]"
								>
									View Results
								</button>
							</div>
							<div class="text-xs text-[var(--text-muted)]">{formatDate(job.created_at)}</div>
						</div>
					</div>
				</div>
			{/each}
		</div>
	{/if}
</div>
