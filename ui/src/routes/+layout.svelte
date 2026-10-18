<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import Modal from 'svelte-simple-modal';

	let { children } = $props();
	let theme = $state('light');
	let online = $state(true);
	let live = $state<{
		gpu: string;
		util: number;
		memUsed: number;
		memTotal: number;
		temp: number;
		rps: number;
	} | null>(null);
	let lastT = 0;

	// sidebar: drag the right edge to resize, drag it nearly shut to collapse to icons
	const RAIL = 56;
	const MIN_W = 180;
	const MAX_W = 420;
	const SNAP = 120;
	let sideWidth = $state(240);
	let collapsed = $state(false);
	let dragging = $state(false);
	let widthBeforeDrag = 240;
	let shownWidth = $derived(collapsed ? RAIL : sideWidth);

	const nav = [
		{
			group: 'Detection',
			items: [
				{ tab: 'detect', label: 'Run detection', icon: 'M4 7V5a1 1 0 0 1 1-1h2M17 4h2a1 1 0 0 1 1 1v2M20 17v2a1 1 0 0 1-1 1h-2M7 20H5a1 1 0 0 1-1-1v-2M9 9h6v6H9z' },
				{ tab: 'results', label: 'Results', icon: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z' }
			]
		}
	];
	let active = $derived(page.url.searchParams.get('tab') || 'detect');

	function save(key: string, value: string) {
		try {
			localStorage.setItem(key, value);
		} catch {
			// storage can be blocked; the layout still works without it
		}
	}

	function toggleTheme() {
		theme = theme === 'light' ? 'dark' : 'light';
		document.documentElement.setAttribute('data-theme', theme);
		save('theme', theme);
	}

	function toggleSidebar() {
		collapsed = !collapsed;
		save('sidebar', JSON.stringify({ width: sideWidth, collapsed }));
	}

	function startResize(e: PointerEvent) {
		e.preventDefault();
		(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
		dragging = true;
		widthBeforeDrag = sideWidth;
	}

	function resize(e: PointerEvent) {
		if (!dragging) return;
		if (e.clientX < SNAP) {
			// expanding again later returns to the width from before this drag
			collapsed = true;
			sideWidth = widthBeforeDrag;
		} else {
			collapsed = false;
			sideWidth = Math.min(MAX_W, Math.max(MIN_W, e.clientX));
		}
	}

	function endResize() {
		if (!dragging) return;
		dragging = false;
		save('sidebar', JSON.stringify({ width: sideWidth, collapsed }));
	}

	async function pollLive() {
		try {
			const r = await fetch(`/api/system?since=${lastT}`).then((r) => r.json());
			const s = r.samples.at(-1);
			online = true;
			if (!s) return;
			lastT = s.t;
			live = {
				gpu: s.gpu?.name ?? 'GPU',
				util: s.gpu?.util ?? 0,
				memUsed: s.gpu?.mem_used_mb ?? 0,
				memTotal: s.gpu?.mem_total_mb ?? 1,
				temp: s.gpu?.temp_c ?? 0,
				rps: s.req_per_sec
			};
		} catch {
			online = false;
		}
	}

	onMount(() => {
		try {
			theme = localStorage.getItem('theme') || 'light';
			const sb = JSON.parse(localStorage.getItem('sidebar') || 'null');
			if (sb) {
				sideWidth = Math.min(MAX_W, Math.max(MIN_W, sb.width ?? 240));
				collapsed = !!sb.collapsed;
			}
		} catch {
			// fall back to defaults
		}
		document.documentElement.setAttribute('data-theme', theme);
		pollLive();
		const t = setInterval(pollLive, 2000);
		return () => clearInterval(t);
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>Inference Server</title>
</svelte:head>

{#snippet icon(d: string)}
	<svg viewBox="0 0 24 24" class="h-4 w-4 shrink-0" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path {d} /></svg>
{/snippet}

{#snippet meter(label: string, value: string, frac: number)}
	<div>
		<div class="flex justify-between text-xs">
			<span class="text-[var(--text-muted)]">{label}</span>
			<span class="text-[var(--fg)] tabular-nums">{value}</span>
		</div>
		<div class="mt-1 h-1.5 overflow-hidden rounded-full bg-[var(--border)]">
			<div class="h-full rounded-full bg-[var(--accent)]" style="width: {Math.min(100, frac * 100)}%"></div>
		</div>
	</div>
{/snippet}

<Modal
	styleWindow={{
		background: 'var(--bg)',
		border: '1px solid var(--border)',
		borderRadius: '16px',
		maxWidth: '1024px',
		width: '90vw',
		maxHeight: '90vh',
		overflow: 'auto'
	}}
	styleContent={{ padding: 0 }}
	styleCloseButton={{
		display: 'none'
	}}
	closeOnEsc={true}
	closeOnOuterClick={true}
>
	<div class="min-h-screen bg-[var(--bg)] {dragging ? 'cursor-col-resize select-none' : ''}">
		<header
			class="sticky top-0 z-20 flex h-10 items-center justify-between gap-3 border-b border-[var(--border)] bg-[var(--surface)] px-2 md:px-3"
		>
			<div class="flex min-w-0 items-center gap-2">
				<button
					onclick={toggleSidebar}
					title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
					aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
					class="hidden h-7 w-7 items-center justify-center rounded-md text-[var(--text-muted)] hover:bg-[var(--bg)] hover:text-[var(--fg)] md:flex"
				>
					<svg viewBox="0 0 24 24" class="h-4 w-4" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"
						><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" />{#if !collapsed}<path
								d="M5.5 8h1.5M5.5 11h1.5"
								stroke-linecap="round"
							/>{/if}</svg
					>
				</button>
				<span class="truncate text-sm font-semibold text-[var(--fg)]">Inference Server</span>
			</div>
			<button
				onclick={toggleTheme}
				class="rounded-full border border-[var(--border)] px-3 py-0.5 text-xs text-[var(--fg)] hover:border-[var(--accent)]"
			>
				{theme === 'light' ? 'Dark' : 'Light'}
			</button>
		</header>

		<!-- small screens: nav as a pill row -->
		<nav class="flex gap-2 overflow-x-auto border-b border-[var(--border)] px-4 py-2 md:hidden">
			{#each nav.flatMap((g) => g.items) as item}
				<button
					onclick={() => goto(`?tab=${item.tab}`)}
					class="flex shrink-0 items-center gap-2 rounded-full px-3 py-1.5 text-sm {active === item.tab
						? 'bg-[var(--accent-soft)] text-[var(--accent)]'
						: 'text-[var(--text-muted)]'}">{@render icon(item.icon)}{item.label}</button
				>
			{/each}
		</nav>

		<div class="flex">
			<aside
				style="width: {shownWidth}px"
				class="sticky top-10 hidden h-[calc(100vh-2.5rem)] shrink-0 flex-col justify-between overflow-hidden border-r border-[var(--border)] bg-[var(--surface)] p-2 md:flex {dragging
					? ''
					: 'transition-[width] duration-200'}"
			>
				<nav class="space-y-4">
					{#each nav as g}
						<div>
							{#if collapsed}
								<div class="mx-2 mb-2 border-t border-[var(--border)] first:border-0"></div>
							{:else}
								<div class="mb-1 truncate px-3 text-[11px] font-semibold tracking-wide text-[var(--text-muted)]">
									{g.group}
								</div>
							{/if}
							{#each g.items as item}
								<button
									onclick={() => goto(`?tab=${item.tab}`)}
									title={collapsed ? item.label : undefined}
									class="flex w-full items-center gap-3 rounded-lg py-2 text-left text-sm {collapsed
										? 'justify-center px-0'
										: 'px-3'} {active === item.tab
										? 'bg-[var(--accent-soft)] font-medium text-[var(--accent)]'
										: 'text-[var(--fg)] hover:bg-[var(--bg)]'}"
								>
									{@render icon(item.icon)}{#if !collapsed}<span class="truncate">{item.label}</span>{/if}
								</button>
							{/each}
						</div>
					{/each}
				</nav>

				{#if collapsed}
					<div
						class="flex flex-col items-center gap-1 rounded-lg bg-[var(--bg)] py-2 text-[10px] text-[var(--text-muted)] tabular-nums"
						title="{live?.gpu ?? 'GPU'} · GPU busy {live?.util ?? 0}%"
					>
						<span class="h-2 w-2 rounded-full {online ? 'bg-[var(--status-good)]' : 'bg-[var(--status-critical)]'}"></span>
						<span class="text-[var(--fg)]">{live?.util ?? 0}%</span>
						<span>GPU</span>
					</div>
				{:else}
					<div class="space-y-3 rounded-xl border border-[var(--border)] bg-[var(--bg)] p-3">
						<div class="flex items-center gap-2 text-xs font-semibold text-[var(--fg)]">
							<span class="h-2 w-2 shrink-0 rounded-full {online ? 'bg-[var(--status-good)]' : 'bg-[var(--status-critical)]'}"></span>
							<span class="truncate">{live?.gpu ?? 'GPU'}</span>
						</div>
						{@render meter('GPU busy', `${live?.util ?? 0}%`, (live?.util ?? 0) / 100)}
						{@render meter(
							'GPU memory',
							`${((live?.memUsed ?? 0) / 1024).toFixed(1)} GB`,
							(live?.memUsed ?? 0) / (live?.memTotal ?? 1)
						)}
						<div class="flex justify-between text-xs">
							<span class="text-[var(--text-muted)]">Throughput</span>
							<span class="text-[var(--fg)] tabular-nums">{Math.round(live?.rps ?? 0).toLocaleString()} img/s</span>
						</div>
						<div class="flex justify-between text-xs">
							<span class="text-[var(--text-muted)]">Temperature</span>
							<span class="text-[var(--fg)] tabular-nums">{live?.temp ?? 0}°C</span>
						</div>
					</div>
				{/if}
			</aside>

			<!-- resize sash: highlights on hover, drag to resize, drag near the edge to collapse, double-click to toggle -->
			<div
				role="separator"
				aria-orientation="vertical"
				aria-label="Resize sidebar"
				title="Drag to resize, double-click to collapse"
				onpointerdown={startResize}
				onpointermove={resize}
				onpointerup={endResize}
				onpointercancel={endResize}
				ondblclick={toggleSidebar}
				class="group sticky top-10 z-10 -ml-[3px] hidden h-[calc(100vh-2.5rem)] w-[6px] shrink-0 cursor-col-resize md:block"
			>
				<div
					class="mx-auto h-full w-[2px] transition-colors delay-150 duration-150 {dragging
						? 'bg-[var(--accent)]'
						: 'bg-transparent group-hover:bg-[var(--accent)]'}"
				></div>
			</div>

			<main class="min-w-0 flex-1 px-4 py-6 md:px-8 lg:px-10">
				{@render children?.()}
			</main>
		</div>
	</div>
</Modal>
