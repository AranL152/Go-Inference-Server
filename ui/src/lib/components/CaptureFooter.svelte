<script lang="ts">
	let { file, command, taken }: { file: string; command: string; taken: string } = $props();
	let copied = $state(false);

	function copy() {
		navigator.clipboard?.writeText(command).then(() => {
			copied = true;
			setTimeout(() => (copied = false), 1200);
		});
	}
</script>

<div class="mt-4 flex flex-wrap items-center gap-2 border-t border-[var(--border)] pt-3 text-xs text-[var(--text-muted)]">
	<span>captured {new Date(taken).toLocaleTimeString()}</span>
	<a class="underline hover:text-[var(--fg)]" href="/api/profile/files/{file}">download {file}</a>
	<button
		onclick={copy}
		title="copy"
		class="min-w-0 truncate rounded-md bg-[var(--bg)] px-2 py-1 font-mono text-[var(--fg)] ring-1 ring-[var(--border)] hover:ring-[var(--accent)]"
		>{copied ? 'copied' : command}</button
	>
</div>
