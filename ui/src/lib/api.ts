export interface Detection {
	class_index: number;
	class_name: string;
	confidence: number;
	box: [number, number, number, number];
}

export interface Job {
	id: string;
	source: string;
	classes: string[];
	status: 'running' | 'completed' | 'cancelled';
	total: number;
	done: number;
	failed: number;
	objects: number;
	images_per_sec: number;
	created_at: string;
	finished_at?: string;
}

export interface Category {
	class: string;
	images: number;
	objects: number;
}

export interface ResultItem {
	index: number;
	path: string;
	width: number;
	height: number;
	detections: Detection[];
	error?: string;
}

export interface Config {
	model: string;
	detector: boolean;
	labels: string[];
	max_batch: number;
	conf: number;
	iou: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
	const res = await fetch(path, init);
	const body = await res.json().catch(() => ({}));
	if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
	return body as T;
}

export const api = {
	config: () => request<Config>('/api/config'),
	jobs: () => request<{ jobs: Job[] }>('/api/jobs').then((r) => r.jobs),
	job: (id: string) =>
		request<{ job: Job; categories: Category[]; no_objects: number }>(`/api/jobs/${id}`),
	results: (id: string, cls: string, offset: number, limit: number) =>
		request<{ total: number; items: ResultItem[] }>(
			`/api/jobs/${id}/results?class=${encodeURIComponent(cls)}&offset=${offset}&limit=${limit}`
		),
	startPath: (path: string, classes: string[]) =>
		request<Job>('/api/jobs', {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ path, classes })
		}),
	cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
	imageURL: (id: string, index: number) => `/api/jobs/${id}/images/${index}`
};

// upload uses XHR so the form can show upload progress
export function upload(
	files: File[],
	classes: string[],
	onProgress: (fraction: number) => void
): Promise<Job> {
	const form = new FormData();
	form.append('classes', classes.join(','));
	for (const f of files) form.append('files', f, f.name);
	return new Promise((resolve, reject) => {
		const xhr = new XMLHttpRequest();
		xhr.open('POST', '/api/jobs/upload');
		xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
		xhr.onload = () => {
			const body = JSON.parse(xhr.responseText || '{}');
			if (xhr.status >= 200 && xhr.status < 300) resolve(body);
			else reject(new Error(body.error || `upload failed ${xhr.status}`));
		};
		xhr.onerror = () => reject(new Error('upload failed'));
		xhr.send(form);
	});
}

export function parseClasses(text: string): string[] {
	return text
		.split(/[,\n]/)
		.map((s) => s.trim().toLowerCase())
		.filter(Boolean);
}
