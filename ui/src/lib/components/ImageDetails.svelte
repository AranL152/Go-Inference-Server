<script lang="ts">
	import { getContext } from 'svelte';
	import type { Context } from 'svelte-simple-modal';
	import type { ResultItem } from '$lib/api';
	import DetectionImage from './DetectionImage.svelte';

	interface Props {
		src: string;
		item: ResultItem;
		highlight: string;
	}
	let { src, item, highlight }: Props = $props();
	const { close } = getContext<Context>('simple-modal');
</script>

<div class="p-8">
	<div class="mb-6 flex items-start justify-between gap-4">
		<div class="min-w-0">
			<h2 class="truncate font-mono text-lg text-[var(--fg)]">{item.path}</h2>
			<p class="mt-1 text-xs text-[var(--text-muted)] tabular-nums">
				{item.width} × {item.height} · {item.detections.length} objects
			</p>
		</div>
		<button
			onclick={() => close()}
			class="border border-[var(--border)] px-3 py-1 text-xs tracking-wider text-[var(--fg)] uppercase hover:opacity-60"
			>Close</button
		>
	</div>

	<DetectionImage {src} {item} {highlight} />

	{#if item.detections.length > 0}
		<table class="mt-6 w-full text-sm">
			<thead>
				<tr class="border-b border-[var(--border)] text-left text-xs tracking-wider text-[var(--text-muted)] uppercase">
					<th class="py-2 font-normal">Class</th>
					<th class="py-2 text-right font-normal">Confidence</th>
					<th class="py-2 text-right font-normal">Box x1 y1 x2 y2</th>
				</tr>
			</thead>
			<tbody>
				{#each item.detections as d}
					<tr class="border-b border-[var(--border)] text-[var(--fg)]">
						<td class="py-2">{d.class_name}</td>
						<td class="py-2 text-right tabular-nums">{(d.confidence * 100).toFixed(1)}%</td>
						<td class="py-2 text-right font-mono text-xs text-[var(--text-muted)] tabular-nums">
							{d.box.map((v) => Math.round(v)).join(' ')}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	{/if}
</div>
