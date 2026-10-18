export interface Gpu {
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

export interface Sample {
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
	alloc_bytes_per_req: number;
	allocs_per_req: number;
	gc_per_sec: number;
	gc_cpu_pct: number;
	gc_pause_p99_us: number;
	sched_p99_us: number;
	heap_live_mb: number;
}

export interface ServerConfig {
	model: string;
	detector: boolean;
	max_batch: number;
	max_wait_ms: number;
	infer_lanes: number;
	unpack_workers: number;
	cpu_decoders: number;
	cores: number;
}

export interface StageTime {
	stage: string;
	goroutines: number;
	seconds: number;
	running_pct: number;
	runnable_pct: number;
	blocked_pct: number;
	syscall_pct: number;
	waits: { reason: string; pct: number }[] | null;
	sched_p50_us: number;
	sched_p99_us: number;
}

export interface TraceSummary {
	file: string;
	taken: string;
	seconds: number;
	stages: StageTime[] | null;
	gcs: number;
	gc_mark_pct: number;
	stw_ms: number;
}

export interface FuncShare {
	func: string;
	pct: number;
}

export interface CpuSummary {
	file: string;
	taken: string;
	seconds: number;
	cores: number;
	stages: { stage: string; pct: number; cores: number; top: FuncShare[] | null }[] | null;
	top: FuncShare[] | null;
}

export interface BatchingStats {
	max_batch: number;
	max_wait_ms: number;
	batches: number;
	sizes: number[];
	avg_size: number;
	full_pct: number;
	fill_p50_ms: number;
	fill_p99_ms: number;
	lane_wait_p50_ms: number;
	lane_wait_p99_ms: number;
}

export interface CopyStats {
	count: number;
	mb: number;
	avg_kb: number;
	gb_per_s: number;
	pinned_pct: number;
}

export interface NsysSummary {
	model: string;
	seconds: number;
	gpu_busy_pct: number;
	compute_pct: number;
	copy_pct: number;
	copy_hidden_pct: number;
	idle_gaps: number;
	idle_gap_p50_us: number;
	idle_gap_p99_us: number;
	kernels_per_s: number;
	kernel_avg_us: number;
	short_kernel_pct: number;
	launch_avg_us: number;
	threads_blocked_on_gpu: number;
	h2d: CopyStats;
	d2h: CopyStats;
	top_kernels: { name: string; calls: number; total_ms: number; avg_us: number; share_pct: number }[];
}

// short function names: drop the package path and keep receiver and name
export function shortFunc(f: string) {
	const i = f.lastIndexOf('/');
	return i >= 0 ? f.slice(i + 1) : f;
}

export async function startCapture<T>(kind: 'cpu' | 'trace', seconds: number): Promise<T> {
	const r = await fetch(`/api/profile/${kind}?seconds=${seconds}`, { method: 'POST' });
	const body = await r.json().catch(() => ({}));
	if (!r.ok) throw new Error(body.error || r.statusText);
	return body as T;
}

export async function latestCapture<T>(kind: 'cpu' | 'trace'): Promise<T | null> {
	return fetch(`/api/profile/${kind}`)
		.then((r) => r.json())
		.catch(() => null);
}
