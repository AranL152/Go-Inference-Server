<script lang="ts">
	let { busy, seconds, onclick }: { busy: boolean; seconds: number; onclick: () => void } = $props();
</script>

<button
	{onclick}
	disabled={busy}
	class="rounded-full bg-[var(--accent)] px-3 py-1 text-xs font-medium text-white hover:opacity-90 disabled:opacity-50"
>
	{busy ? `Capturing ${seconds} s...` : `Capture ${seconds} s`}
</button>
