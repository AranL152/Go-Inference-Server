<script lang="ts">
	import { onMount, tick } from 'svelte';

	interface Line {
		seq: number;
		t: number;
		text: string;
	}
	let lines = $state<Line[]>([]);
	let filter = $state('');
	let box: HTMLDivElement;

	async function poll() {
		const since = lines.at(-1)?.seq ?? 0;
		try {
			const r = await fetch(`/api/logs?since=${since}`).then((r) => r.json());
			if (!r.lines.length) return;
			const atBottom = box && box.scrollHeight - box.scrollTop - box.clientHeight < 24;
			lines = [...lines, ...r.lines].slice(-500);
			await tick();
			if (atBottom && box) box.scrollTop = box.scrollHeight;
		} catch {
			// the dashboard shows the connection error
		}
	}

	onMount(() => {
		poll().then(() => box && (box.scrollTop = box.scrollHeight));
		const t = setInterval(poll, 2000);
		return () => clearInterval(t);
	});

	const shown = $derived(
		filter ? lines.filter((l) => l.text.toLowerCase().includes(filter.toLowerCase())) : lines
	);
	const time = (ms: number) =>
		new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
</script>

<div>
	<input
		bind:value={filter}
		placeholder="Filter log lines"
		class="mb-3 w-full border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs text-[var(--fg)] placeholder-[var(--text-muted)] focus:border-[var(--accent)] focus:outline-none"
	/>
	<div bind:this={box} class="h-64 overflow-y-auto border border-[var(--border)] font-mono text-xs">
		<table class="w-full">
			<thead class="sticky top-0 bg-[var(--surface)]">
				<tr class="border-b border-[var(--border)] text-left text-[var(--text-muted)]">
					<th class="w-24 px-3 py-1.5 font-normal">Timestamp</th>
					<th class="px-3 py-1.5 font-normal">Message</th>
				</tr>
			</thead>
			<tbody>
				{#each shown as l (l.seq)}
					<tr class="border-b border-[var(--border)] align-top">
						<td class="px-3 py-1 whitespace-nowrap text-[var(--text-muted)] tabular-nums">{time(l.t)}</td>
						<td class="px-3 py-1 break-all text-[var(--fg)]">{l.text}</td>
					</tr>
				{/each}
			</tbody>
		</table>
		{#if shown.length === 0}
			<p class="p-4 text-center text-[var(--text-muted)]">No log lines{filter ? ' match' : ''}.</p>
		{/if}
	</div>
</div>
