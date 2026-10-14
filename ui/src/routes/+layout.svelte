<script lang="ts">
	import '../app.css';
	import favicon from '$lib/assets/favicon.svg';
	import { onMount } from 'svelte';
	import Modal from 'svelte-simple-modal';

	let { children } = $props();
	let theme = $state('light');

	function toggleTheme() {
		theme = theme === 'light' ? 'dark' : 'light';
		document.documentElement.setAttribute('data-theme', theme);
		localStorage.setItem('theme', theme);
	}

	onMount(() => {
		const saved = localStorage.getItem('theme') || 'light';
		theme = saved;
		document.documentElement.setAttribute('data-theme', theme);
	});
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
	<title>ToyReduce</title>
</svelte:head>

<Modal
	styleWindow={{
		background: 'var(--bg)',
		border: '1px solid var(--border)',
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
	<div class="min-h-screen bg-[var(--bg)]">
		<!-- Clean header -->
		<header class="border-b border-[var(--border)]">
			<div class="mx-auto flex max-w-6xl items-center justify-between px-6 py-8">
				<div>
					<h1 class="text-2xl font-semibold tracking-tight text-[var(--fg)]">ToyReduce</h1>
					<p class="mt-1 text-sm text-[var(--text-muted)]">MapReduce Job Server</p>
				</div>
				<button
					onclick={toggleTheme}
					class="border border-[var(--border)] px-3 py-1 text-xs tracking-wider text-[var(--fg)] uppercase hover:opacity-60"
				>
					{theme === 'light' ? 'Dark' : 'Light'}
				</button>
			</div>
		</header>

		<!-- Content -->
		<main class="mx-auto max-w-6xl px-6 py-12">
			{@render children?.()}
		</main>
	</div>
</Modal>
