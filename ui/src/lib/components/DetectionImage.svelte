<script lang="ts">
	import type { ResultItem } from '$lib/api';

	interface Props {
		src: string;
		item: ResultItem;
		highlight?: string;
		labels?: boolean;
	}
	let { src, item, highlight = '', labels = true }: Props = $props();

	const pct = (v: number, of: number) => `${(100 * v) / of}%`;
	const active = (cls: string) => !highlight || highlight === cls || highlight === '_none';
</script>

<div class="relative">
	<img {src} alt={item.path} loading="lazy" class="block h-auto w-full" />
	{#if item.width && item.height}
		<svg
			viewBox="0 0 {item.width} {item.height}"
			preserveAspectRatio="none"
			class="pointer-events-none absolute inset-0 h-full w-full"
		>
			{#each item.detections as d}
				<rect
					x={d.box[0]}
					y={d.box[1]}
					width={d.box[2] - d.box[0]}
					height={d.box[3] - d.box[1]}
					fill="none"
					stroke={active(d.class_name) ? 'var(--series)' : 'var(--grey-light)'}
					stroke-width={active(d.class_name) ? 2 : 1}
					vector-effect="non-scaling-stroke"
				/>
			{/each}
		</svg>
		{#if labels}
			{#each item.detections.filter((d) => active(d.class_name)) as d}
				<span
					class="pointer-events-none absolute bg-[var(--series)] px-1 text-[10px] leading-4 whitespace-nowrap text-white"
					style="left: {pct(d.box[0], item.width)}; top: {pct(d.box[1], item.height)}"
				>
					{d.class_name}
					{Math.round(d.confidence * 100)}%
				</span>
			{/each}
		{/if}
	{/if}
</div>
