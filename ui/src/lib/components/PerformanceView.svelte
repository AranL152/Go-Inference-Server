<script lang="ts">
	import { onMount } from 'svelte';
	import Chart from './Chart.svelte';

	interface Sample {
		t: number;
		gpu: null | {
			name: string;
			util: number;
			mem_util: number;
			mem_used_mb: number;
			mem_total_mb: number;
			temp_c: number;
			power_w: number;
			power_limit_w: number;
			sm_clock_mhz: number;
			sm_clock_max_mhz: number;
			mem_clock_mhz: number;
			pcie_gen: number;
			pcie_width: number;
		};
		cpu: number;
		cores: number[];
		mem_used_mb: number;
		mem_total_mb: number;
		proc_cpu: number;
		proc_rss_mb: number;
		goroutines: number;
		req_per_sec: number;
		avg_batch: number;
		queue_ms: number;
		run_ms: number;
		lane_busy: number;
		p50_ms: number;
		p99_ms: number;
	}

	type Panel = 'gpu' | 'cpu' | 'memory' | 'pipeline';
	let samples = $state<Sample[]>([]);
	let panel = $state<Panel>('gpu');
	let error = $state('');

	async function poll() {
		const since = samples.at(-1)?.t ?? 0;
		try {
			const r = await fetch(`/api/system?since=${since}`).then((r) => r.json());
			samples = [...samples, ...r.samples].slice(-120);
			error = '';
		} catch {
			error = 'Lost connection to the server.';
		}
	}

	onMount(() => {
		poll();
		const t = setInterval(poll, 1000);
		return () => clearInterval(t);
	});

	const s = $derived(samples.at(-1));
	const col = <K extends keyof Sample>(k: K) => samples.map((x) => x[k] as number);
	const gcol = (f: (g: NonNullable<Sample['gpu']>) => number) => samples.map((x) => (x.gpu ? f(x.gpu) : 0));
	const gb = (mb: number) => (mb / 1024).toFixed(1);

	const blue = 'var(--series)';
	const orange = 'var(--series-2)';

	const nav = $derived([
		{
			id: 'gpu' as Panel,
			title: 'GPU',
			sub: s?.gpu ? `${s.gpu.util}%  ${s.gpu.temp_c}°C` : 'no data',
			values: gcol((g) => g.util),
			max: 100
		},
		{ id: 'cpu' as Panel, title: 'CPU', sub: s ? `${s.cpu.toFixed(0)}%` : '', values: col('cpu'), max: 100 },
		{
			id: 'memory' as Panel,
			title: 'Memory',
			sub: s ? `${gb(s.mem_used_mb)} / ${gb(s.mem_total_mb)} GB` : '',
			values: col('mem_used_mb'),
			max: s?.mem_total_mb || 1
		},
		{
			id: 'pipeline' as Panel,
			title: 'Inference',
			sub: s ? `${Math.round(s.req_per_sec).toLocaleString()} img/s` : '',
			values: col('req_per_sec'),
			max: undefined
		}
	]);
</script>

{#snippet stat(label: string, value: string)}
	<div>
		<div class="text-xs text-[var(--text-muted)]">{label}</div>
		<div class="text-lg font-semibold text-[var(--fg)] tabular-nums">{value}</div>
	</div>
{/snippet}

{#snippet card(title: string, children: () => any)}
	<div class="border border-[var(--border)] bg-[var(--surface)] p-6">
		<div class="mb-4 text-xs tracking-wider text-[var(--text-muted)] uppercase">{title}</div>
		{@render children()}
	</div>
{/snippet}

{#if error}
	<div class="mb-8 border border-red-500 p-4 text-sm text-red-600">{error}</div>
{/if}

<div class="grid grid-cols-[240px_1fr] gap-8">
	<nav class="space-y-2 self-start">
		{#each nav as n}
			<button
				onclick={() => (panel = n.id)}
				class="flex w-full items-center gap-3 border p-3 text-left {panel === n.id
					? 'border-[var(--accent)] bg-[var(--surface)]'
					: 'border-[var(--border)] hover:border-[var(--text-muted)]'}"
			>
				<div class="w-20 shrink-0">
					<Chart series={[{ name: n.title, color: blue, values: n.values }]} max={n.max} height={40} compact />
				</div>
				<div class="min-w-0">
					<div class="text-sm font-semibold text-[var(--fg)]">{n.title}</div>
					<div class="truncate text-xs text-[var(--text-muted)] tabular-nums">{n.sub}</div>
				</div>
			</button>
		{/each}
	</nav>

	<div class="min-w-0 space-y-6">
		{#if !s}
			<p class="py-16 text-center text-sm text-[var(--text-muted)]">Collecting samples...</p>
		{:else if panel === 'gpu'}
			<div class="flex items-baseline justify-between">
				<h2 class="text-2xl font-semibold text-[var(--fg)]">GPU</h2>
				<span class="text-sm text-[var(--text-muted)]">{s.gpu?.name ?? 'nvidia-smi unavailable'}</span>
			</div>
			{#snippet util()}
				<Chart series={[{ name: 'GPU utilization', color: blue, values: gcol((g) => g.util) }]} max={100} unit="%" height={220} />
			{/snippet}
			{@render card('Utilization', util)}
			<div class="grid grid-cols-2 gap-6">
				{#snippet vram()}
					<Chart
						series={[{ name: 'VRAM', color: blue, values: gcol((g) => g.mem_used_mb / 1024) }]}
						max={(s.gpu?.mem_total_mb ?? 1024) / 1024}
						unit=" GB"
						height={120}
					/>
				{/snippet}
				{@render card('Dedicated memory', vram)}
				{#snippet power()}
					<Chart
						series={[{ name: 'Power', color: blue, values: gcol((g) => g.power_w) }]}
						max={s.gpu?.power_limit_w || undefined}
						unit=" W"
						height={120}
					/>
				{/snippet}
				{@render card('Power draw', power)}
			</div>
			{#if s.gpu}
				<div class="grid grid-cols-4 gap-6 border border-[var(--border)] bg-[var(--surface)] p-6">
					{@render stat('Utilization', `${s.gpu.util}%`)}
					{@render stat('Memory', `${gb(s.gpu.mem_used_mb)} / ${gb(s.gpu.mem_total_mb)} GB`)}
					{@render stat('Temperature', `${s.gpu.temp_c}°C`)}
					{@render stat('Power', `${Math.round(s.gpu.power_w)} / ${Math.round(s.gpu.power_limit_w)} W`)}
					{@render stat('SM clock', `${s.gpu.sm_clock_mhz} / ${s.gpu.sm_clock_max_mhz} MHz`)}
					{@render stat('Memory clock', `${s.gpu.mem_clock_mhz} MHz`)}
					{@render stat('Memory controller', `${s.gpu.mem_util}%`)}
					{@render stat('PCIe', `Gen ${s.gpu.pcie_gen} x${s.gpu.pcie_width}`)}
				</div>
			{/if}
		{:else if panel === 'cpu'}
			<div class="flex items-baseline justify-between">
				<h2 class="text-2xl font-semibold text-[var(--fg)]">CPU</h2>
				<span class="text-sm text-[var(--text-muted)]">{s.cores.length} logical processors</span>
			</div>
			{#snippet cores()}
				<div class="grid grid-cols-4 gap-3">
					{#each s.cores as _, i}
						<div>
							<Chart
								series={[{ name: `CPU ${i}`, color: blue, values: samples.map((x) => x.cores[i] ?? 0) }]}
								max={100}
								height={56}
								compact
							/>
							<div class="mt-1 flex justify-between text-[10px] text-[var(--text-muted)] tabular-nums">
								<span>CPU {i}</span><span>{s.cores[i].toFixed(0)}%</span>
							</div>
						</div>
					{/each}
				</div>
			{/snippet}
			{@render card('Utilization per logical processor', cores)}
			<div class="grid grid-cols-4 gap-6 border border-[var(--border)] bg-[var(--surface)] p-6">
				{@render stat('Total', `${s.cpu.toFixed(0)}%`)}
				{@render stat('Inference server', `${s.proc_cpu.toFixed(0)}%`)}
				{@render stat('Goroutines', s.goroutines.toLocaleString())}
				{@render stat('Busiest core', `${Math.max(...s.cores).toFixed(0)}%`)}
			</div>
		{:else if panel === 'memory'}
			<h2 class="text-2xl font-semibold text-[var(--fg)]">Memory</h2>
			{#snippet mem()}
				<Chart
					series={[{ name: 'Memory in use', color: blue, values: col('mem_used_mb').map((v) => v / 1024) }]}
					max={s.mem_total_mb / 1024}
					unit=" GB"
					height={220}
				/>
			{/snippet}
			{@render card('Memory in use', mem)}
			<div class="grid grid-cols-3 gap-6 border border-[var(--border)] bg-[var(--surface)] p-6">
				{@render stat('In use', `${gb(s.mem_used_mb)} GB`)}
				{@render stat('Total', `${gb(s.mem_total_mb)} GB`)}
				{@render stat('Inference server', `${gb(s.proc_rss_mb)} GB`)}
			</div>
		{:else}
			<h2 class="text-2xl font-semibold text-[var(--fg)]">Inference</h2>
			{#snippet rate()}
				<Chart series={[{ name: 'Images per second', color: blue, values: col('req_per_sec') }]} height={220} unit="/s" />
			{/snippet}
			{@render card('Throughput', rate)}
			<div class="grid grid-cols-2 gap-6">
				{#snippet lat()}
					<Chart
						series={[
							{ name: 'p50', color: blue, values: col('p50_ms') },
							{ name: 'p99', color: orange, values: col('p99_ms') }
						]}
						unit=" ms"
						height={120}
					/>
				{/snippet}
				{@render card('Request latency', lat)}
				{#snippet batch()}
					<Chart series={[{ name: 'Batch size', color: blue, values: col('avg_batch') }]} height={120} />
				{/snippet}
				{@render card('Average batch size', batch)}
			</div>
			<div class="grid grid-cols-4 gap-6 border border-[var(--border)] bg-[var(--surface)] p-6">
				{@render stat('Throughput', `${Math.round(s.req_per_sec).toLocaleString()} img/s`)}
				{@render stat('Latency p50 / p99', `${s.p50_ms.toFixed(0)} / ${s.p99_ms.toFixed(0)} ms`)}
				{@render stat('GPU run per batch', `${s.run_ms.toFixed(1)} ms`)}
				{@render stat('Queue wait', `${s.queue_ms.toFixed(1)} ms`)}
				{@render stat('Batch size', s.avg_batch.toFixed(1))}
				{@render stat('Infer lanes busy', `${s.lane_busy.toFixed(0)}%`)}
			</div>
		{/if}
	</div>
</div>
