import { EventSourceParserStream } from 'eventsource-parser/stream';

import { WEBUI_API_BASE_URL } from '$lib/constants';

// -----------------------------------------------------------------------------
// SSE consumption helper
//
// This mirrors the pattern already used for chat completion streaming in
// `$lib/apis/streaming` (`createOpenAITextStream`) and the raw-response +
// `AbortController` tuple pattern used by streaming fetches elsewhere (e.g.
// `pullModel` in `$lib/apis/ollama` and `downloadChatStats` in
// `$lib/apis/chats`): the fetch function itself returns the raw `Response`
// (not `.json()`-parsed) alongside an `AbortController` so the caller can
// cancel the stream, and a small shared async generator turns
// `res.body` into parsed SSE events via `TextDecoderStream` +
// `EventSourceParserStream`. The Benchmarks SSE payloads aren't OpenAI
// chat-completion shaped, so this is a generic version of
// `createOpenAITextStream` rather than a reuse of it.
// -----------------------------------------------------------------------------

export type BenchmarksSSEEvent<T = any> = {
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	event: string;
	data: T | null;
};

export async function* parseBenchmarksEventStream<T = any>(
	body: ReadableStream<Uint8Array>
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
): AsyncGenerator<BenchmarksSSEEvent<T>> {
	const reader = body
		.pipeThrough(new TextDecoderStream())
		.pipeThrough(new EventSourceParserStream())
		.getReader();

	while (true) {
		const { value, done } = await reader.read();
		if (done) {
			break;
		}
		if (!value) {
			continue;
		}

		const eventName = value.event ?? 'message';

		let data: T | null = null;
		if (value.data) {
			try {
				data = JSON.parse(value.data);
			} catch (e) {
				console.error('Error parsing benchmarks SSE event:', e);
			}
		}

		yield { event: eventName, data };

		if (eventName === 'done') {
			break;
		}
	}
}

// -----------------------------------------------------------------------------
// Serve
// -----------------------------------------------------------------------------

export const getServeProfiles = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/serve/profiles`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getServeProfile = async (token: string = '', name: string) => {
	let error = null;

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/serve/profile/${encodeURIComponent(name)}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export type ServeStartForm = {
	profile?: string;
	ngl?: number;
	ctx?: number;
	threads?: number;
	parallel?: number;
	ot?: string;
	reasoning?: string;
	spec?: string;
};

export const startServe = async (token: string = '', form: ServeStartForm = {}) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/serve/start`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(form)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const stopServe = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/serve/stop`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const checkServe = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/serve/check`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// Returns the raw [Response, AbortController] tuple, matching the
// established convention for streaming fetches in this codebase (see
// `pullModel` in `$lib/apis/ollama` and `downloadChatStats` in
// `$lib/apis/chats`). Callers pass `res.body` into `parseBenchmarksEventStream`
// above to get parsed `{ event, data }` updates, the same way
// `createOpenAITextStream(res.body, ...)` is used for chat streaming.
export const streamServe = async (
	token: string = ''
): Promise<[Response | null, AbortController]> => {
	const controller = new AbortController();
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/serve/stream`, {
		signal: controller.signal,
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	}).catch((err) => {
		console.error(err);
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return [res, controller];
};

// -----------------------------------------------------------------------------
// Live
// -----------------------------------------------------------------------------

export const getLive = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/live/`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

export const getTestOptions = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/tests/options`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export type TestRunForm = {
	suite: string;
	profile?: string;
	benchmark?: string;
	system?: string;
	slice?: string;
	resume?: boolean;
};

export const startTestRun = async (token: string = '', form: TestRunForm) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/tests/run`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(form)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const cancelTestRun = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/tests/cancel`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// See the comment above `streamServe` — same [Response, AbortController]
// convention, consumed via `parseBenchmarksEventStream`.
export const streamTestRun = async (
	token: string = ''
): Promise<[Response | null, AbortController]> => {
	const controller = new AbortController();
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/tests/stream`, {
		signal: controller.signal,
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	}).catch((err) => {
		console.error(err);
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return [res, controller];
};

// -----------------------------------------------------------------------------
// Compare
// -----------------------------------------------------------------------------

export type CompareBy = 'config' | 'benchmark' | 'failures' | 'serving';

// Response shape varies with `by` (rows-based for config/failures, columns +
// rows for benchmark, derived_columns/derived for serving) — left untyped so
// the caller narrows it based on the `by` it passed.
export const getCompare = async (
	token: string = '',
	{
		by,
		tier = null,
		baseline = null
	}: { by: CompareBy; tier?: string | null; baseline?: string | null }
) => {
	let error = null;

	const searchParams = new URLSearchParams();
	searchParams.append('by', by);
	if (tier) searchParams.append('tier', tier);
	if (baseline) searchParams.append('baseline', baseline);

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/compare/?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const exportAnswers = async (token: string = '', run: string) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/compare/export-answers`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ run })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// -----------------------------------------------------------------------------
// Answers
// -----------------------------------------------------------------------------

export const getAnswerRuns = async (token: string = '', limit: number = 20) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (limit) searchParams.append('limit', limit.toString());

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/answers/runs?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getAnswers = async (
	token: string = '',
	run: string,
	filter: 'failures' | 'all' | 'pass' = 'failures'
) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (run) searchParams.append('run', run);
	if (filter) searchParams.append('filter', filter);

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/answers/?${searchParams.toString()}`, {
		method: 'GET',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getAnswerOne = async (
	token: string = '',
	{
		run,
		benchmark,
		item_id,
		thinking = false
	}: { run: string; benchmark: string; item_id: string; thinking?: boolean }
) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (run) searchParams.append('run', run);
	if (benchmark) searchParams.append('benchmark', benchmark);
	if (item_id) searchParams.append('item_id', item_id);
	searchParams.append('thinking', thinking ? 'true' : 'false');

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/answers/one?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// -----------------------------------------------------------------------------
// Report
// -----------------------------------------------------------------------------

export type ReportForm = {
	tier?: string;
	model?: string;
	benchmark?: string;
	figures?: boolean;
};

export const generateReport = async (token: string = '', form: ReportForm = {}) => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/report/`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify({ figures: true, ...form })
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// The backend returns `markdown_url`/`figures` from `generateReport` as
// origin-relative paths (e.g. "/api/v1/benchmarks/report/files/<run>/<file>")
// which is identical to what this builds (WEBUI_API_BASE_URL already equals
// `${WEBUI_BASE_URL}/api/v1`) — so those response fields can be used as-is,
// e.g. directly as an `<img src>`. This helper exists for building the same
// URL when you already know `runDir`/`filename` without re-calling
// `generateReport` (e.g. revisiting a past report from `getRecentSweeps`/
// history).
export const reportFileUrl = (runDir: string, filename: string): string =>
	`${WEBUI_API_BASE_URL}/benchmarks/report/files/${encodeURIComponent(runDir)}/${encodeURIComponent(
		filename
	)}`;

// `<img>`/`<a>` tags can't attach an `authorization` header, so an `<img
// src={reportFileUrl(...)}>` only works if the browser sends the session
// cookie for this same-origin request (this app's `getFileContentById` in
// `$lib/apis/files` relies on the same thing via `credentials: 'include'` for
// binary content it can't token-auth from a plain URL). These endpoints are
// admin-only, so if cookie auth isn't enabled/sufficient for a given
// deployment, figures embedded via `reportFileUrl` may 401 — fetch the bytes
// with this helper (which does carry the Bearer token) and render them via an
// object URL instead. For the markdown body specifically, always use this
// helper (or a plain fetch with the Authorization header) rather than a raw
// `<img>`-style URL, since markdown needs to be read as text, not displayed.
export const getReportFileText = async (token: string = '', runDir: string, filename: string) => {
	let error = null;

	const res = await fetch(reportFileUrl(runDir, filename), {
		method: 'GET',
		headers: {
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json().catch(() => ({ detail: res.statusText }));
			return res.text();
		})
		.catch((err) => {
			error = err?.detail ?? err;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// -----------------------------------------------------------------------------
// Tune
// -----------------------------------------------------------------------------

export type TuneStartForm = {
	profile?: string;
	tier?: string;
	benchmark?: string;
	system?: string;
	grid?: string;
	budget?: string;
	candidates?: number;
	round_items?: number;
	eta?: number;
	seed?: number;
	stages?: string[];
};

export const startTune = async (token: string = '', form: TuneStartForm = {}) => {
	let error = null;

	const body = {
		tier: 'smoke',
		budget: 'interactive',
		eta: 2,
		seed: 0,
		stages: ['explore', 'refine'],
		...form
	};

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/tune/start`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		},
		body: JSON.stringify(body)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const resumeTune = async (token: string = '', sweepId: string | null = null) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (sweepId) searchParams.append('sweep_id', sweepId);

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/tune/resume?${searchParams.toString()}`,
		{
			method: 'POST',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const stopTune = async (token: string = '') => {
	let error = null;

	const res = await fetch(`${WEBUI_API_BASE_URL}/benchmarks/tune/stop`, {
		method: 'POST',
		headers: {
			Accept: 'application/json',
			'Content-Type': 'application/json',
			authorization: `Bearer ${token}`
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getTuneStatus = async (token: string = '', sweepId: string | null = null) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (sweepId) searchParams.append('sweep_id', sweepId);

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/tune/status?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getRecentSweeps = async (token: string = '', limit: number = 20) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (limit) searchParams.append('limit', limit.toString());

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/tune/recent?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			error = err.detail;
			console.error(err);
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

// See the comment above `streamServe` — same [Response, AbortController]
// convention, consumed via `parseBenchmarksEventStream`.
export const streamTuneLog = async (
	token: string = '',
	sweepId: string | null = null
): Promise<[Response | null, AbortController]> => {
	const controller = new AbortController();
	let error = null;

	const searchParams = new URLSearchParams();
	if (sweepId) searchParams.append('sweep_id', sweepId);

	const res = await fetch(
		`${WEBUI_API_BASE_URL}/benchmarks/tune/log?${searchParams.toString()}`,
		{
			signal: controller.signal,
			method: 'GET',
			headers: {
				Accept: 'application/json',
				'Content-Type': 'application/json',
				authorization: `Bearer ${token}`
			}
		}
	).catch((err) => {
		console.error(err);
		error = err;
		return null;
	});

	if (error) {
		throw error;
	}

	return [res, controller];
};
