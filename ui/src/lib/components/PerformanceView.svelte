<script lang="ts">
	import { onMount } from 'svelte';
	import Chart, { type Series } from './Chart.svelte';
	import KernelProfile from './KernelProfile.svelte';
	import LogWidget from './LogWidget.svelte';
	import Panel from './Panel.svelte';
	import Bottleneck from './Bottleneck.svelte';
	import CpuProfile from './CpuProfile.svelte';
	import GoTrace from './GoTrace.svelte';
	import Batching from './Batching.svelte';
	import GoRuntime from './GoRuntime.svelte';
	import GpuTimeline from './GpuTimeline.svelte';
	import { page } from '$app/state';
	import { goto } from '$app/navigation';
	import { api, type Job } from '$lib/api';
	import { formatDate } from '$lib/utils/date';
	import type { Gpu, RunPerf, Sample, ServerConfig } from '$lib/stats';

	const ranges = [
		{ label: '1m', n: 60 },
		{ label: '5m', n: 300 },
		{ label: '15m', n: 900 }
	];
	let range = $state(300);
	let liveSamples = $state<Sample[]>([]);
	let cfg = $state<ServerConfig | null>(null);
	let error = $state('');
	let jobs = $state<Job[]>([]);

	// ?run=<job id> shows that job's recorded run instead of the live server
	const runId = $derived(page.url.searchParams.get('run') ?? '');
	let run = $state<RunPerf | null>(null);
	let runError = $state('');
	$effect(() => {
		const id = runId;
		run = null;
		runError = '';
		if (!id) return;
		fetch(`/api/jobs/${id}/perf`)
			.then(async (r) => {
				const body = await r.json();
				if (!r.ok) throw new Error(body.error);
				return body as RunPerf;
			})
			.then((p) => runId === id && (run = p))
			.catch((e) => (runError = (e as Error).message));
	});
	const samples = $derived(runId ? (run?.samples ?? []) : liveSamples);
	const win = $derived(runId ? Math.max(samples.length, 10) : range);
	const runJob = $derived(jobs.find((j) => j.id === runId));

	function pickRun(id: string) {
		goto(`?tab=performance${id ? `&run=${id}` : ''}`, { keepFocus: true, noScroll: true });
	}

	async function poll() {
		const since = liveSamples.at(-1)?.t ?? 0;
		try {
			const r = await fetch(`/api/system?since=${since}`).then((r) => r.json());
			liveSamples = [...liveSamples, ...r.samples].slice(-900);
			error = '';
		} catch {
			error = 'Lost connection to the server.';
		}
	}

	onMount(() => {
		fetch('/api/config')
			.then((r) => r.json())
			.then((c) => (cfg = c))
			.catch(() => {});
		const loadJobs = () =>
			api
				.jobs()
				.then((j) => (jobs = j))
				.catch(() => {});
		loadJobs();
		poll();
		const t = setInterval(poll, 1000);
		const tj = setInterval(loadJobs, 5000);
		return () => {
			clearInterval(t);
			clearInterval(tj);
		};
	});

	const model = $derived(cfg?.model ?? '');
	const s = $derived(samples.at(-1));
	const times = $derived(samples.map((x) => x.t));
	const col = (f: (x: Sample) => number) => samples.map(f);
	const gcol = (f: (g: Gpu) => number) => samples.map((x) => (x.gpu ? f(x.gpu) : 0));
	const gb = (mb: number) => (mb / 1024).toFixed(1);
	const one = (name: string, values: number[], color = 'var(--series)'): Series[] => [
		{ name, color, values }
	];

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
					{
						label: 'Throughput',
						value: Math.round(s.req_per_sec).toLocaleString(),
						unit: 'img/s',
						spark: col((x) => x.req_per_sec)
					},
					{
						label: 'p99 latency',
						value: s.p99_ms.toFixed(0),
						unit: 'ms',
						spark: col((x) => x.p99_ms)
					},
					{
						label: 'GPU busy',
						value: `${s.gpu?.util ?? 0}`,
						unit: '%',
						spark: gcol((g) => g.util)
					},
					{
						label: 'GPU memory',
						value: s.gpu ? gb(s.gpu.mem_used_mb) : '0',
						unit: 'GB',
						spark: gcol((g) => g.mem_used_mb)
					},
					{
						label: 'Host to GPU',
						value: Math.round(s.gpu?.pcie_rx_mbs ?? 0).toLocaleString(),
						unit: 'MB/s',
						spark: gcol((g) => g.pcie_rx_mbs)
					},
					{
						label: 'GPU power',
						value: Math.round(s.gpu?.power_w ?? 0).toString(),
						unit: 'W',
						spark: gcol((g) => g.power_w)
					}
				]
			: []
	);

	const views = [
		{ id: 'analysis', label: 'Analysis' },
		{ id: 'series', label: 'Time series' },
		{ id: 'logs', label: 'Logs' }
	] as const;
	let view = $state<(typeof views)[number]['id']>('analysis');
</script>

{#snippet card(title: string, sub: string, body: () => any, wide = false)}
	<Panel {title} {sub} {wide}>{@render body()}</Panel>
{/snippet}

<div class="space-y-5">
	<div class="flex flex-wrap items-center justify-between gap-3">
		<div
			class="flex gap-1 overflow-x-auto rounded-xl bg-[var(--surface)] p-1 shadow-sm ring-1 ring-[var(--border)]"
		>
			{#each views as v}
				<button
					onclick={() => (view = v.id)}
					class="shrink-0 rounded-lg px-3 py-1.5 text-sm {view === v.id
						? 'bg-[var(--accent)] text-white'
						: 'text-[var(--text-muted)] hover:text-[var(--fg)]'}">{v.label}</button
				>
			{/each}
		</div>
		<div class="flex flex-wrap items-center gap-3">
			<select
				value={runId}
				onchange={(e) => pickRun(e.currentTarget.value)}
				class="max-w-72 rounded-lg border border-[var(--border)] bg-[var(--surface)] py-1 pr-8 pl-3 text-sm text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
			>
				<option value="">Live</option>
				{#each jobs as j}
					<option value={j.id}>Run {formatDate(j.created_at)} · {j.source.split('/').at(-1)}</option
					>
				{/each}
			</select>
			{#if !runId}
				<div
					class="flex rounded-full bg-[var(--surface)] p-0.5 text-xs ring-1 ring-[var(--border)]"
				>
					{#each ranges as r}
						<button
							onclick={() => (range = r.n)}
							class="rounded-full px-3 py-1 {range === r.n
								? 'bg-[var(--fg)] text-[var(--bg)]'
								: 'text-[var(--text-muted)] hover:text-[var(--fg)]'}">{r.label}</button
						>
					{/each}
				</div>
			{/if}
		</div>
	</div>

	{#if runId}
		<div
			class="flex flex-wrap gap-x-5 gap-y-1 rounded-xl bg-[var(--surface)] px-4 py-2 text-xs text-[var(--text-muted)] tabular-nums ring-1 ring-[var(--border)]"
		>
			{#if runError}
				<span class="text-[var(--status-critical)]">{runError}</span>
			{:else if !run}
				<span>Loading run…</span>
			{:else}
				<span
					>{new Date(run.start).toLocaleTimeString()} to {new Date(run.end).toLocaleTimeString()} · {(
						(Date.parse(run.end) - Date.parse(run.start)) /
						1000
					).toFixed(1)} s</span
				>
				{#if runJob}
					<span class="font-mono">{runJob.source}</span>
					<span
						>{runJob.done.toLocaleString()} images · {Math.round(
							runJob.images_per_sec
						).toLocaleString()} img/s · {runJob.status}</span
					>
				{/if}
			{/if}
		</div>
	{/if}

	{#if error}
		<div
			class="flex items-center gap-3 rounded-xl border border-[var(--status-critical)] bg-[var(--surface)] px-4 py-3 text-sm text-[var(--fg)]"
		>
			<span class="font-bold text-[var(--status-critical)]">✕</span>{error}
		</div>
	{/if}

	{#if view === 'analysis'}
		{#key runId}
			<Bottleneck {samples} {cfg} span={runId ? samples.length : 5} run={!!runId} />

			<div class="grid items-start gap-4 lg:grid-cols-2">
				<CpuProfile live={!runId} data={run?.cpu ?? null} />
				<GoTrace live={!runId} data={run?.trace ?? null} />
			</div>

			<div class="grid items-start gap-4 lg:grid-cols-2">
				<Batching live={!runId} data={run?.batching ?? null} />
				<GoRuntime {samples} range={win} span={runId ? samples.length : 10} />
			</div>
		{/key}

		{#snippet breakdown()}
			<Chart series={stageSeries} {times} window={win} stacked unit=" ms" height={180} />
			<div class="mt-4 grid grid-cols-2 gap-x-6 gap-y-1.5 text-xs tabular-nums sm:grid-cols-4">
				{#each stages as st, i}
					<div class="flex items-center justify-between gap-2">
						<span class="flex min-w-0 items-center gap-1.5 text-[var(--text-muted)]">
							<span
								class="inline-block h-2 w-2 shrink-0 rounded-full"
								style="background: var(--cat-{i + 1})"
							></span>
							<span class="truncate">{st.name}</span>
						</span>
						<span class="text-[var(--fg)]">{stageNow[i].toFixed(1)}</span>
					</div>
				{/each}
				<div class="flex justify-between gap-2 font-semibold text-[var(--fg)]">
					<span>Total</span><span>{stageTotal.toFixed(1)} ms</span>
				</div>
			</div>
		{/snippet}
		{@render card(
			'Latency budget',
			'decode and letterbox per image, GPU stages per batch',
			breakdown
		)}

		<GpuTimeline {model} />
		{#snippet kp()}<KernelProfile {model} />{/snippet}
		{@render card('GPU kernels', 'Nsight Compute hardware counters for one batch', kp)}

		<div>
			<h2 class="mb-3 text-sm font-semibold text-[var(--text-muted)]">Hardware health</h2>
			<div class="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
				{#each numbers as n}
					<div class="overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--surface)]">
						<div class="px-3 pt-2">
							<div class="text-xs text-[var(--text-muted)]">{n.label}</div>
							<div class="text-lg font-semibold text-[var(--fg)] tabular-nums">
								{n.value}<span class="ml-1 text-xs font-normal text-[var(--text-muted)]"
									>{n.unit}</span
								>
							</div>
						</div>
						<div class="px-1 pb-1">
							<Chart
								series={one(n.label, n.spark)}
								window={win}
								height={28}
								compact
								frame={false}
							/>
						</div>
					</div>
				{/each}
			</div>
			<div class="mt-3 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
				{#each alarms as a}
					<div
						class="rounded-xl border border-[var(--border)] bg-[var(--surface)] px-3 py-2"
						style="border-left: 4px solid {levelStyle[a.level].color}"
					>
						<div class="flex items-center justify-between gap-2">
							<span class="truncate text-xs text-[var(--fg)]">{a.name}</span>
							<span class="text-xs font-bold" style="color: {levelStyle[a.level].color}"
								>{levelStyle[a.level].icon}</span
							>
						</div>
						<div class="truncate text-[11px] text-[var(--text-muted)]">{a.detail}</div>
					</div>
				{/each}
			</div>
		</div>
	{:else if view === 'series'}
		<h2 class="text-sm font-semibold text-[var(--text-muted)]">Pipeline</h2>
		<div class="grid gap-4 lg:grid-cols-2">
			{#snippet thr()}<Chart
					series={one(
						'Images per second',
						col((x) => x.req_per_sec)
					)}
					{times}
					window={win}
					unit="/s"
				/>{/snippet}
			{@render card('Throughput', 'images per second', thr)}
			{#snippet lat()}
				<Chart
					series={[
						{ name: 'p50', color: 'var(--series)', values: col((x) => x.p50_ms) },
						{ name: 'p99', color: 'var(--series-2)', values: col((x) => x.p99_ms) }
					]}
					{times}
					window={win}
					unit=" ms"
				/>
			{/snippet}
			{@render card('Request latency', 'milliseconds', lat)}
			{#snippet bat()}<Chart
					series={one(
						'Batch size',
						col((x) => x.avg_batch)
					)}
					{times}
					window={win}
				/>{/snippet}
			{@render card('Batch size', 'images per GPU run', bat)}
			{#snippet que()}<Chart
					series={one(
						'Queue wait',
						col((x) => x.queue_ms)
					)}
					{times}
					window={win}
					unit=" ms"
				/>{/snippet}
			{@render card('Queue wait', 'submit to GPU start', que)}
			{#snippet gpuRun()}<Chart
					series={one(
						'GPU compute',
						col((x) => x.compute_ms)
					)}
					{times}
					window={win}
					unit=" ms"
				/>{/snippet}
			{@render card('GPU compute per batch', 'milliseconds', gpuRun)}
			{#snippet lanes()}<Chart
					series={one(
						'Lanes busy',
						col((x) => x.lane_busy)
					)}
					{times}
					window={win}
					max={100}
					unit="%"
				/>{/snippet}
			{@render card('Inference lanes busy', 'percent of time', lanes)}
		</div>

		<h2 class="text-sm font-semibold text-[var(--text-muted)]">Go runtime</h2>
		<div class="grid gap-4 lg:grid-cols-2">
			{#snippet alloc()}<Chart
					series={one(
						'Allocated per request',
						col((x) => x.alloc_bytes_per_req / 1024)
					)}
					{times}
					window={win}
					unit=" KB"
				/>{/snippet}
			{@render card('Allocated per request', 'kilobytes', alloc)}
			{#snippet gccpu()}<Chart
					series={one(
						'GC share of CPU',
						col((x) => x.gc_cpu_pct)
					)}
					{times}
					window={win}
					unit="%"
				/>{/snippet}
			{@render card('GC share of CPU', 'percent of the CPU the server used', gccpu)}
			{#snippet pause()}<Chart
					series={one(
						'GC pause p99',
						col((x) => x.gc_pause_p99_us)
					)}
					{times}
					window={win}
					unit=" µs"
				/>{/snippet}
			{@render card('GC pause p99', 'microseconds', pause)}
			{#snippet sched()}<Chart
					series={one(
						'Wait for CPU p99',
						col((x) => x.sched_p99_us)
					)}
					{times}
					window={win}
					unit=" µs"
				/>{/snippet}
			{@render card('Goroutine wait for CPU p99', 'microseconds from runnable to running', sched)}
		</div>

		<h2 class="text-sm font-semibold text-[var(--text-muted)]">GPU</h2>
		<div class="grid gap-4 lg:grid-cols-2">
			{#snippet gu()}<Chart
					series={one(
						'GPU busy',
						gcol((g) => g.util)
					)}
					{times}
					window={win}
					max={100}
					unit="%"
				/>{/snippet}
			{@render card('GPU busy', 'time any kernel ran', gu)}
			{#snippet gm()}
				<Chart
					series={one(
						'GPU memory',
						gcol((g) => g.mem_used_mb / 1024)
					)}
					{times}
					window={win}
					max={(s?.gpu?.mem_total_mb ?? 1024) / 1024}
					unit=" GB"
				/>
			{/snippet}
			{@render card('GPU memory', `of ${s?.gpu ? gb(s.gpu.mem_total_mb) : '0'} GB`, gm)}
			{#snippet pci()}
				<Chart
					series={[
						{ name: 'Host to GPU', color: 'var(--series)', values: gcol((g) => g.pcie_rx_mbs) },
						{ name: 'GPU to host', color: 'var(--series-2)', values: gcol((g) => g.pcie_tx_mbs) }
					]}
					{times}
					window={win}
					unit=" MB/s"
				/>
			{/snippet}
			{@render card(
				'PCIe traffic',
				s?.gpu ? `Gen ${s.gpu.pcie_gen} x${s.gpu.pcie_width}` : '',
				pci
			)}
			{#snippet pw()}<Chart
					series={one(
						'Power',
						gcol((g) => g.power_w)
					)}
					{times}
					window={win}
					max={s?.gpu?.power_limit_w || undefined}
					unit=" W"
				/>{/snippet}
			{@render card('Power draw', `limit ${Math.round(s?.gpu?.power_limit_w ?? 0)} W`, pw)}
			{#snippet tmp()}<Chart
					series={one(
						'Temperature',
						gcol((g) => g.temp_c)
					)}
					{times}
					window={win}
					max={100}
					unit="°C"
				/>{/snippet}
			{@render card('Temperature', 'degrees Celsius', tmp)}
			{#snippet clk()}<Chart
					series={one(
						'SM clock',
						gcol((g) => g.sm_clock_mhz)
					)}
					{times}
					window={win}
					max={s?.gpu?.sm_clock_max_mhz || undefined}
					unit=" MHz"
				/>{/snippet}
			{@render card('SM clock', `max ${s?.gpu?.sm_clock_max_mhz ?? 0} MHz`, clk)}
		</div>

		<h2 class="text-sm font-semibold text-[var(--text-muted)]">Host</h2>
		<div class="grid gap-4 lg:grid-cols-2">
			{#snippet cpu()}
				<Chart
					series={[
						{ name: 'All processes', color: 'var(--series)', values: col((x) => x.cpu) },
						{ name: 'Inference server', color: 'var(--series-2)', values: col((x) => x.proc_cpu) }
					]}
					{times}
					window={win}
					max={100}
					unit="%"
				/>
			{/snippet}
			{@render card('CPU utilization', `${s?.cores.length ?? 0} logical processors`, cpu)}
			{#snippet ram()}
				<Chart
					series={[
						{
							name: 'All processes',
							color: 'var(--series)',
							values: col((x) => x.mem_used_mb / 1024)
						},
						{
							name: 'Inference server',
							color: 'var(--series-2)',
							values: col((x) => x.proc_rss_mb / 1024)
						}
					]}
					{times}
					window={win}
					max={(s?.mem_total_mb ?? 1024) / 1024}
					unit=" GB"
				/>
			{/snippet}
			{@render card('Host memory', `of ${s ? gb(s.mem_total_mb) : '0'} GB`, ram)}
			{#snippet cores()}
				<div class="grid grid-cols-4 gap-2 md:grid-cols-8">
					{#each s?.cores ?? [] as c, i}
						<div>
							<Chart
								series={one(
									`CPU ${i}`,
									samples.map((x) => x.cores[i] ?? 0)
								)}
								window={win}
								max={100}
								height={40}
								compact
							/>
							<div
								class="mt-0.5 flex justify-between text-[10px] text-[var(--text-muted)] tabular-nums"
							>
								<span>CPU {i}</span><span>{c.toFixed(0)}%</span>
							</div>
						</div>
					{/each}
				</div>
			{/snippet}
			{@render card('Per core', 'percent busy', cores, true)}
			{#snippet gor()}<Chart
					series={one(
						'Goroutines',
						col((x) => x.goroutines)
					)}
					{times}
					window={win}
				/>{/snippet}
			{@render card('Goroutines', 'server concurrency', gor, true)}
		</div>
	{:else}
		{#snippet logs()}<LogWidget />{/snippet}
		{@render card('Server log', 'latest 500 lines', logs)}
	{/if}
</div>
