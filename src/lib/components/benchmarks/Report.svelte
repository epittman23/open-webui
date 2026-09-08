<script lang="ts">
	import { getContext, onDestroy } from 'svelte';
	import { marked } from 'marked';

	import { generateReport, getReportFileText } from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');

	const errDetail = (err: unknown, fallback: string): string => {
		if (typeof err === 'string') return err;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		return (err as any)?.detail ?? fallback;
	};

	// -----------------------------------------------------------------------------
	// Filter form
	// -----------------------------------------------------------------------------

	let tier = '';
	let model = '';
	let benchmark = '';
	let figures = true;

	// -----------------------------------------------------------------------------
	// Generation state
	// -----------------------------------------------------------------------------

	let loading = false;
	let error: string | null = null;

	let markdownHtml: string | null = null;

	// Each entry pairs the backend's origin-relative figure URL with a local
	// blob object URL created after an authenticated fetch (see loadFigures
	// below for why we don't just point <img src> at the backend URL
	// directly).
	let figureEntries: { url: string; objectUrl: string }[] = [];

	const revokeFigureUrls = () => {
		for (const entry of figureEntries) {
			URL.revokeObjectURL(entry.objectUrl);
		}
		figureEntries = [];
	};

	// `<img>` tags can't attach an `authorization` header, and whether this
	// deployment's session cookie also authorizes these admin-only endpoints
	// for a plain same-origin `<img src>` request is unclear from here (see
	// the comment above `reportFileUrl`/`getReportFileText` in the API
	// client). Fetching each figure with the Bearer token and rendering it
	// via `URL.createObjectURL` works regardless of that, so that's the
	// approach used here rather than relying on cookie auth.
	const loadFigures = async (urls: string[]) => {
		revokeFigureUrls();

		const entries = await Promise.all(
			urls.map(async (url) => {
				try {
					const res = await fetch(url, {
						headers: { authorization: `Bearer ${localStorage.token}` }
					});
					if (!res.ok) return null;
					const blob = await res.blob();
					return { url, objectUrl: URL.createObjectURL(blob) };
				} catch (e) {
					console.error('Failed to load report figure:', url, e);
					return null;
				}
			})
		);

		figureEntries = entries.filter((e): e is { url: string; objectUrl: string } => e !== null);
	};

	// `markdown_url` looks like `/api/v1/benchmarks/report/files/<run_dir>/report.md`;
	// getReportFileText needs the run dir and filename split back out of it.
	const splitMarkdownUrl = (markdownUrl: string): { runDir: string; filename: string } | null => {
		const parts = markdownUrl.split('/').filter(Boolean);
		if (parts.length < 2) return null;
		const filename = decodeURIComponent(parts[parts.length - 1]);
		const runDir = decodeURIComponent(parts[parts.length - 2]);
		return { runDir, filename };
	};

	const generate = async () => {
		loading = true;
		error = null;
		markdownHtml = null;
		revokeFigureUrls();

		try {
			const form = {
				...(tier.trim() ? { tier: tier.trim() } : {}),
				...(model.trim() ? { model: model.trim() } : {}),
				...(benchmark.trim() ? { benchmark: benchmark.trim() } : {}),
				figures
			};

			const res = await generateReport(localStorage.token, form).catch((err) => {
				error = errDetail(err, $i18n.t('Failed to generate report'));
				return null;
			});

			if (!res) {
				// error already set above
			} else if (res.error) {
				error = res.error;
			} else {
				const parsed = splitMarkdownUrl(res.markdown_url);
				if (!parsed) {
					error = $i18n.t('Unexpected report response');
				} else {
					const text = await getReportFileText(localStorage.token, parsed.runDir, parsed.filename).catch(
						(err) => {
							error = errDetail(err, $i18n.t('Failed to load report'));
							return null;
						}
					);

					if (text !== null) {
						markdownHtml = marked.parse(text) as string;

						if (Array.isArray(res.figures) && res.figures.length > 0) {
							await loadFigures(res.figures);
						}
					}
				}
			}
		} finally {
			loading = false;
		}
	};

	onDestroy(() => {
		revokeFigureUrls();
	});
</script>

<div class="flex flex-col gap-4">
	<div>
		<h2 class="text-lg font-medium text-gray-900 dark:text-white">{$i18n.t('Report')}</h2>
		<div class="text-xs text-gray-500 dark:text-gray-400">
			{$i18n.t('Generate a statistical report and figures from recorded benchmark runs.')}
		</div>
	</div>

	<form
		class="flex flex-wrap items-end gap-3"
		on:submit|preventDefault={generate}
	>
		<div class="flex flex-col gap-1">
			<label for="report-tier" class="text-xs text-gray-500 dark:text-gray-400"
				>{$i18n.t('Tier')}</label
			>
			<input
				id="report-tier"
				type="text"
				bind:value={tier}
				placeholder={$i18n.t('e.g. smoke, standard, full')}
				class="text-sm px-2.5 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-850 outline-none w-40"
			/>
		</div>

		<div class="flex flex-col gap-1">
			<label for="report-model" class="text-xs text-gray-500 dark:text-gray-400"
				>{$i18n.t('Model')}</label
			>
			<input
				id="report-model"
				type="text"
				bind:value={model}
				placeholder={$i18n.t('Optional model filter')}
				class="text-sm px-2.5 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-850 outline-none w-48"
			/>
		</div>

		<div class="flex flex-col gap-1">
			<label for="report-benchmark" class="text-xs text-gray-500 dark:text-gray-400"
				>{$i18n.t('Benchmark')}</label
			>
			<input
				id="report-benchmark"
				type="text"
				bind:value={benchmark}
				placeholder={$i18n.t('Optional benchmark filter')}
				class="text-sm px-2.5 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-850 outline-none w-48"
			/>
		</div>

		<label class="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 pb-1.5">
			<input type="checkbox" bind:checked={figures} class="size-4" />
			{$i18n.t('Generate figures')}
		</label>

		<button
			type="submit"
			disabled={loading}
			class="px-3.5 py-1.5 text-sm font-medium rounded-lg bg-black text-white dark:bg-white dark:text-black disabled:opacity-50 flex items-center gap-2"
		>
			{#if loading}
				<Spinner className="size-3.5" />
			{/if}
			{$i18n.t('Generate')}
		</button>
	</form>

	{#if loading}
		<div class="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400 my-6">
			<Spinner className="size-4" />
			{$i18n.t('Generating report… this can take a while for large runs.')}
		</div>
	{:else if error}
		<div
			class="text-sm text-red-700 dark:text-red-200 bg-red-500/10 rounded-lg px-3 py-2 whitespace-pre-wrap"
		>
			{error}
		</div>
	{:else if markdownHtml}
		{#if figureEntries.length > 0}
			<div class="grid sm:grid-cols-2 gap-3">
				{#each figureEntries as entry (entry.url)}
					<img
						src={entry.objectUrl}
						alt={entry.url}
						class="rounded-lg border border-gray-200 dark:border-gray-800 w-full"
					/>
				{/each}
			</div>
		{/if}

		<div class="prose dark:prose-invert max-w-none text-sm">
			{@html markdownHtml}
		</div>
	{:else}
		<div class="text-sm text-gray-500 dark:text-gray-400 my-6">
			{$i18n.t('Set optional filters and click Generate to build a report.')}
		</div>
	{/if}
</div>
