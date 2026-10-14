<script lang="ts">
	import { onMount } from 'svelte';
	import Chart, { type Series } from './Chart.svelte';
	import KernelProfile from './KernelProfile.svelte';
	import LogWidget from './LogWidget.svelte';

	interface Gpu {
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
		pcie_rx_mbs: number;
		pcie_tx_mbs: number;
		throttle: string[];
		power_violation_pct: number;
		thermal_violation_pct: number;
	}
	interface Sample {
		t: number;
		gpu: Gpu | null;
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
		decode_ms: number;
		letterbox_ms: number;
		h2d_ms: number;
		compute_ms: number;
		d2h_ms: number;
		post_ms: number;
	}

	const ranges = [
		{ label: '1m', n: 60 },
		{ label: '5m', n: 300 },
		{ label: '15m', n: 900 }
	];
	let range = $state(300);
	let samples = $state<Sample[]>([]);
	let model = $state('');
	let error = $state('');

	async function poll() {
		const since = samples.at(-1)?.t ?? 0;
		try {
			const r = await fetch(`/api/system?since=${since}`).then((r) => r.json());
			samples = [...samples, ...r.samples].slice(-900);
			error = '';
		} catch {
			error = 'Lost connection to the server.';
		}
	}

	onMount(() => {
		fetch('/api/config')
			.then((r) => r.json())
			.then((c) => (model = c.model))
			.catch(() => {});
		poll();
		const t = setInterval(poll, 1000);
		return () => clearInterval(t);
	});

	const s = $derived(samples.at(-1));
	const times = $derived(samples.map((x) => x.t));
	const col = (f: (x: Sample) => number) => samples.map(f);
	const gcol = (f: (g: Gpu) => number) => samples.map((x) => (x.gpu ? f(x.gpu) : 0));
	const gb = (mb: number) => (mb / 1024).toFixed(1);
	const one = (name: string, values: number[], color = 'var(--series)'): Series[] => [{ name, color, values }];

	const stages = [
		{ key: 'decode_ms', name: 'JPEG decode' },
		{ key: 'letterbox_ms', name: 'Letterbox' },
		{ key: 'queue_ms', name: 'Queue wait' },
		{ key: 'h2d_ms', name: 'Copy to GPU' },
		{ key: 'compute_ms', name: 'GPU compute' },
		{ key: 'd2h_ms', name: 'Copy from GPU' },
		{ key: 'post_ms', name: 'Decode boxes' }
	] as const;
	const stageSeries = $derived(
		stages.map((st, i) => ({
			name: st.name,
			color: `var(--cat-${i + 1})`,
			values: col((x) => x[st.key] ?? 0)
		}))
	);
	const stageNow = $derived(stages.map((st) => (s ? (s[st.key] ?? 0) : 0)));
	const stageTotal = $derived(stageNow.reduce((a, b) => a + b, 0));

	type Level = 'ok' | 'warning' | 'alarm' | 'nodata';
	const alarms = $derived.by((): { name: string; level: Level; detail: string }[] => {
		if (!s) return [{ name: 'Waiting for samples', level: 'nodata', detail: '' }];
		const g = s.gpu;
		const throttled = g?.throttle ?? [];
		return [
			{
				name: 'GPU clock throttling',
				level: throttled.length ? 'warning' : 'ok',
				detail: throttled.length ? throttled.join(', ') : 'clocks free to boost'
			},
			{
				name: 'GPU temperature',
				level: !g ? 'nodata' : g.temp_c >= 83 ? 'alarm' : g.temp_c >= 75 ? 'warning' : 'ok',
				detail: g ? `${g.temp_c}°C` : 'no GPU data'
			},
			{
				name: 'Power limit',
				level: !g ? 'nodata' : g.power_violation_pct > 0 ? 'warning' : 'ok',
				detail: g ? `${g.power_violation_pct.toFixed(0)}% of time capped` : ''
			},
			{
				name: 'p99 latency',
				level: s.p99_ms > 500 ? 'alarm' : s.p99_ms > 250 ? 'warning' : 'ok',
				detail: `${s.p99_ms.toFixed(0)} ms`
			},
			{
				name: 'CPU saturation',
				level: s.cpu > 90 ? 'alarm' : s.cpu > 75 ? 'warning' : 'ok',
				detail: `${s.cpu.toFixed(0)}% of ${s.cores.length} cores`
			},
			{
				name: 'Host memory',
				level: s.mem_used_mb / s.mem_total_mb > 0.9 ? 'alarm' : 'ok',
				detail: `${gb(s.mem_used_mb)} of ${gb(s.mem_total_mb)} GB`
			}
		];
	});
	const levelStyle: Record<Level, { color: string; icon: string; label: string }> = {
		ok: { color: 'var(--status-good)', icon: '✓', label: 'OK' },
		warning: { color: 'var(--status-warning)', icon: '▲', label: 'Warning' },
		alarm: { color: 'var(--status-critical)', icon: '✕', label: 'In alarm' },
		nodata: { color: 'var(--text-muted)', icon: '–', label: 'No data' }
	};

	const numbers = $derived(
		s
			? [
					{ label: 'Throughput', value: Math.round(s.req_per_sec).toLocaleString(), unit: 'img/s' },
					{ label: 'p99 latency', value: s.p99_ms.toFixed(0), unit: 'ms' },
					{ label: 'GPU busy', value: `${s.gpu?.util ?? 0}`, unit: '%' },
					{ label: 'GPU memory', value: s.gpu ? gb(s.gpu.mem_used_mb) : '0', unit: 'GB' },
					{ label: 'Host to GPU', value: Math.round(s.gpu?.pcie_rx_mbs ?? 0).toLocaleString(), unit: 'MB/s' },
					{ label: 'GPU power', value: Math.round(s.gpu?.power_w ?? 0).toString(), unit: 'W' }
				]
			: []
	);
</script>

{#snippet card(title: string, sub: string, body: () => any, wide = false)}
	<div class="border border-[var(--border)] bg-[var(--surface)] p-4 {wide ? 'md:col-span-2' : ''}">
		<div class="mb-3 flex items-baseline justify-between gap-2">
			<h3 class="text-sm font-semibold text-[var(--fg)]">{title}</h3>
			<span class="text-xs text-[var(--text-muted)]">{sub}</span>
		</div>
		{@render body()}
	</div>
{/snippet}

{#snippet section(title: string)}
	<h2 class="mt-4 border-b border-[var(--border)] pb-2 text-lg font-semibold text-[var(--fg)]">{title}</h2>
{/snippet}

<div class="space-y-4">
	<div class="flex flex-wrap items-center justify-between gap-4">
		<div>
			<h2 class="text-xl font-semibold text-[var(--fg)]">Inference server dashboard</h2>
			<p class="text-xs text-[var(--text-muted)]">
				{s?.gpu?.name ?? 'GPU'} · {model || 'model'} · refreshes every second
			</p>
		</div>
		<div class="flex border border-[var(--border)] text-xs">
			{#each ranges as r}
				<button
					onclick={() => (range = r.n)}
					class="px-3 py-1.5 {range === r.n
						? 'bg-[var(--fg)] text-[var(--bg)]'
						: 'text-[var(--text-muted)] hover:text-[var(--fg)]'}">{r.label}</button
				>
			{/each}
		</div>
	</div>

	{#if error}
		<div class="border border-[var(--status-critical)] p-3 text-sm text-[var(--fg)]">✕ {error}</div>
	{/if}

	<div class="grid grid-cols-2 gap-4 md:grid-cols-6">
		{#each numbers as n}
			<div class="border border-[var(--border)] bg-[var(--surface)] p-4">
				<div class="text-xs text-[var(--text-muted)]">{n.label}</div>
				<div class="mt-1 text-3xl font-semibold text-[var(--fg)] tabular-nums">
					{n.value}<span class="ml-1 text-sm font-normal text-[var(--text-muted)]">{n.unit}</span>
				</div>
			</div>
		{/each}
	</div>

	<div class="grid gap-4 md:grid-cols-3">
		{#snippet alarmBody()}
			<ul class="space-y-2">
				{#each alarms as a}
					<li class="flex items-start gap-3 border border-[var(--border)] px-3 py-2">
						<span class="mt-0.5 w-4 text-center font-bold" style="color: {levelStyle[a.level].color}"
							>{levelStyle[a.level].icon}</span
						>
						<div class="min-w-0 flex-1">
							<div class="flex justify-between gap-2 text-sm text-[var(--fg)]">
								<span>{a.name}</span>
								<span class="text-xs text-[var(--text-muted)]">{levelStyle[a.level].label}</span>
							</div>
							<div class="truncate text-xs text-[var(--text-muted)]">{a.detail}</div>
						</div>
					</li>
				{/each}
			</ul>
		{/snippet}
		{@render card('Alarm status', `${alarms.filter((a) => a.level === 'alarm').length} in alarm`, alarmBody)}

		{#snippet breakdown()}
			<div class="grid gap-4 lg:grid-cols-[1fr_200px]">
				<Chart series={stageSeries} {times} window={range} stacked unit=" ms" height={190} />
				<table class="self-start text-xs tabular-nums">
					<tbody>
						{#each [...stages].reverse() as st, ri}
							{@const i = stages.length - 1 - ri}
							<tr class="border-b border-[var(--border)]">
								<td class="py-1 pr-2">
									<span class="mr-1.5 inline-block h-2 w-2" style="background: var(--cat-{i + 1})"></span
									>{st.name}
								</td>
								<td class="py-1 pr-2 text-right text-[var(--fg)]">{stageNow[i].toFixed(1)}</td>
								<td class="py-1 text-right text-[var(--text-muted)]"
									>{stageTotal ? Math.round((100 * stageNow[i]) / stageTotal) : 0}%</td
								>
							</tr>
						{/each}
						<tr><td class="py-1 pr-2 font-semibold">Total</td><td class="py-1 pr-2 text-right font-semibold">{stageTotal.toFixed(1)} ms</td><td></td></tr>
					</tbody>
				</table>
			</div>
		{/snippet}
		{@render card('Where the time goes', 'decode and letterbox per image, GPU stages per batch', breakdown, true)}
	</div>

	{@render section('Inference')}
	<div class="grid gap-4 md:grid-cols-3">
		{#snippet thr()}<Chart series={one('Images per second', col((x) => x.req_per_sec))} {times} window={range} unit="/s" />{/snippet}
		{@render card('Throughput', 'images per second', thr)}
		{#snippet lat()}
			<Chart
				series={[
					{ name: 'p50', color: 'var(--series)', values: col((x) => x.p50_ms) },
					{ name: 'p99', color: 'var(--series-2)', values: col((x) => x.p99_ms) }
				]}
				{times}
				window={range}
				unit=" ms"
			/>
		{/snippet}
		{@render card('Request latency', 'milliseconds', lat)}
		{#snippet bat()}<Chart series={one('Batch size', col((x) => x.avg_batch))} {times} window={range} />{/snippet}
		{@render card('Batch size', 'images per GPU run', bat)}
		{#snippet que()}<Chart series={one('Queue wait', col((x) => x.queue_ms))} {times} window={range} unit=" ms" />{/snippet}
		{@render card('Queue wait', 'submit to GPU start', que)}
		{#snippet gpuRun()}<Chart series={one('GPU compute', col((x) => x.compute_ms))} {times} window={range} unit=" ms" />{/snippet}
		{@render card('GPU compute per batch', 'milliseconds', gpuRun)}
		{#snippet lanes()}<Chart series={one('Lanes busy', col((x) => x.lane_busy))} {times} window={range} max={100} unit="%" />{/snippet}
		{@render card('Inference lanes busy', 'percent of time', lanes)}
	</div>

	{@render section('GPU')}
	<div class="grid gap-4 md:grid-cols-3">
		{#snippet gu()}<Chart series={one('GPU busy', gcol((g) => g.util))} {times} window={range} max={100} unit="%" />{/snippet}
		{@render card('GPU busy', 'time any kernel ran', gu)}
		{#snippet gm()}
			<Chart series={one('GPU memory', gcol((g) => g.mem_used_mb / 1024))} {times} window={range} max={(s?.gpu?.mem_total_mb ?? 1024) / 1024} unit=" GB" />
		{/snippet}
		{@render card('GPU memory', `of ${s?.gpu ? gb(s.gpu.mem_total_mb) : '0'} GB`, gm)}
		{#snippet pci()}
			<Chart
				series={[
					{ name: 'Host to GPU', color: 'var(--series)', values: gcol((g) => g.pcie_rx_mbs) },
					{ name: 'GPU to host', color: 'var(--series-2)', values: gcol((g) => g.pcie_tx_mbs) }
				]}
				{times}
				window={range}
				unit=" MB/s"
			/>
		{/snippet}
		{@render card('PCIe traffic', s?.gpu ? `Gen ${s.gpu.pcie_gen} x${s.gpu.pcie_width}` : '', pci)}
		{#snippet pw()}<Chart series={one('Power', gcol((g) => g.power_w))} {times} window={range} max={s?.gpu?.power_limit_w || undefined} unit=" W" />{/snippet}
		{@render card('Power draw', `limit ${Math.round(s?.gpu?.power_limit_w ?? 0)} W`, pw)}
		{#snippet tmp()}<Chart series={one('Temperature', gcol((g) => g.temp_c))} {times} window={range} max={100} unit="°C" />{/snippet}
		{@render card('Temperature', 'degrees Celsius', tmp)}
		{#snippet clk()}<Chart series={one('SM clock', gcol((g) => g.sm_clock_mhz))} {times} window={range} max={s?.gpu?.sm_clock_max_mhz || undefined} unit=" MHz" />{/snippet}
		{@render card('SM clock', `max ${s?.gpu?.sm_clock_max_mhz ?? 0} MHz`, clk)}
	</div>

	{@render section('CPU and memory')}
	<div class="grid gap-4 md:grid-cols-3">
		{#snippet cpu()}
			<Chart
				series={[
					{ name: 'All processes', color: 'var(--series)', values: col((x) => x.cpu) },
					{ name: 'Inference server', color: 'var(--series-2)', values: col((x) => x.proc_cpu) }
				]}
				{times}
				window={range}
				max={100}
				unit="%"
			/>
		{/snippet}
		{@render card('CPU utilization', `${s?.cores.length ?? 0} logical processors`, cpu)}
		{#snippet ram()}
			<Chart
				series={[
					{ name: 'All processes', color: 'var(--series)', values: col((x) => x.mem_used_mb / 1024) },
					{ name: 'Inference server', color: 'var(--series-2)', values: col((x) => x.proc_rss_mb / 1024) }
				]}
				{times}
				window={range}
				max={(s?.mem_total_mb ?? 1024) / 1024}
				unit=" GB"
			/>
		{/snippet}
		{@render card('Host memory', `of ${s ? gb(s.mem_total_mb) : '0'} GB`, ram)}
		{#snippet gor()}<Chart series={one('Goroutines', col((x) => x.goroutines))} {times} window={range} />{/snippet}
		{@render card('Goroutines', 'server concurrency', gor)}
		{#snippet cores()}
			<div class="grid grid-cols-4 gap-2 md:grid-cols-8">
				{#each s?.cores ?? [] as c, i}
					<div>
						<Chart series={one(`CPU ${i}`, samples.map((x) => x.cores[i] ?? 0))} window={range} max={100} height={40} compact />
						<div class="mt-0.5 flex justify-between text-[10px] text-[var(--text-muted)] tabular-nums">
							<span>CPU {i}</span><span>{c.toFixed(0)}%</span>
						</div>
					</div>
				{/each}
			</div>
		{/snippet}
		{@render card('Per core', 'percent busy', cores, true)}
	</div>

	{@render section('GPU kernels (Nsight Compute)')}
	{#snippet kp()}<KernelProfile {model} />{/snippet}
	{@render card('Kernel profile', 'hardware counters from one profiled batch', kp)}

	{@render section('Logs')}
	{#snippet logs()}<LogWidget />{/snippet}
	{@render card('Server log', 'latest 500 lines', logs)}
</div>
