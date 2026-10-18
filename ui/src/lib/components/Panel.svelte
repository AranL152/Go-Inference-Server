<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		sub?: string;
		finding?: string;
		wide?: boolean;
		actions?: Snippet;
		children: Snippet;
	}
	let { title, sub = '', finding = '', wide = false, actions, children }: Props = $props();
</script>

<section
	class="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--surface)] shadow-sm {wide
		? 'lg:col-span-2'
		: ''}"
>
	<div class="flex flex-wrap items-center justify-between gap-2 border-b border-[var(--border)] px-4 py-2.5">
		<div class="min-w-0">
			<h3 class="text-sm font-semibold text-[var(--fg)]">{title}</h3>
			{#if sub}<p class="truncate text-xs text-[var(--text-muted)]">{sub}</p>{/if}
		</div>
		{#if actions}<div class="flex items-center gap-2">{@render actions()}</div>{/if}
	</div>
	{#if finding}
		<div class="border-b border-[var(--border)] bg-[var(--accent-soft)] px-4 py-2 text-sm text-[var(--fg)]">
			{finding}
		</div>
	{/if}
	<div class="p-4">{@render children()}</div>
</section>
