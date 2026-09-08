<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import DOMPurify from 'dompurify';
	import { marked } from 'marked';
	import { getAnswerRuns, getAnswers, getAnswerOne } from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import NativeSelect from '$lib/components/common/NativeSelect.svelte';
	import Badge from '$lib/components/common/Badge.svelte';

	const i18n = getContext('i18n');

	type AnswerRun = {
		suite_run_id: string;
		started_at: number;
		ended_at?: number;
		model: string;
		tier: string;
		attempted: number;
		passed: number;
	};

	type AnswerRow = {
		benchmark: string;
		item_id: string;
		outcome: string;
		reason?: string;
		reasoning_chars?: number;
	};

	// -------------------------------------------------------------------
	// Run picker
	// -------------------------------------------------------------------
	let runs: AnswerRun[] = [];
	let selectedRun = '';
	let loadingRuns = true;
	let runsError: string | null = null;

	const formatRunLabel = (run: AnswerRun): string => {
		const date = run.started_at ? new Date(run.started_at * 1000).toLocaleString() : '—';
		return `${date} · ${run.model} · ${run.tier} · ${run.passed}/${run.attempted}`;
	};

	$: runOptions = runs.map((run) => ({ value: run.suite_run_id, label: formatRunLabel(run) }));

	const loadRuns = async () => {
		loadingRuns = true;
		runsError = null;
		try {
			const res = await getAnswerRuns(localStorage.token, 20);
			runs = res?.runs ?? [];
			if (!selectedRun && runs.length > 0) {
				selectedRun = runs[0].suite_run_id;
			}
		} catch (err) {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			runsError = (err as any)?.detail ?? String(err);
		}
		loadingRuns = false;
	};

	// -------------------------------------------------------------------
	// Results table
	// -------------------------------------------------------------------
	let filter: 'failures' | 'all' | 'pass' = 'failures';
	$: filterOptions = [
		{ value: 'failures', label: $i18n.t('Failures') },
		{ value: 'all', label: $i18n.t('All') },
		{ value: 'pass', label: $i18n.t('Pass') }
	];

	let rows: AnswerRow[] = [];
	let loadingRows = false;
	let rowsError: string | null = null;

	const loadRows = async () => {
		if (!selectedRun) {
			rows = [];
			return;
		}
		loadingRows = true;
		rowsError = null;
		try {
			const res = await getAnswers(localStorage.token, selectedRun, filter);
			rows = res?.rows ?? [];
		} catch (err) {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			rowsError = (err as any)?.detail ?? String(err);
			rows = [];
		}
		loadingRows = false;
	};

	// Reload the results table whenever the selected run or filter changes.
	$: if (selectedRun || filter) {
		loadRows();
		selectedRow = null;
		answerMarkdown = '';
	}

	const outcomeBadgeType = (outcome: string): string => {
		const o = (outcome ?? '').toLowerCase();
		if (o.includes('pass')) return 'success';
		if (o.includes('fail')) return 'error';
		return 'muted';
	};

	// -------------------------------------------------------------------
	// Answer detail
	// -------------------------------------------------------------------
	let selectedRow: AnswerRow | null = null;
	let thinking = false;
	let answerMarkdown = '';
	let loadingAnswer = false;
	let answerError: string | null = null;

	const loadAnswer = async (row: AnswerRow) => {
		if (!selectedRun) return;
		loadingAnswer = true;
		answerError = null;
		try {
			const res = await getAnswerOne(localStorage.token, {
				run: selectedRun,
				benchmark: row.benchmark,
				item_id: row.item_id,
				thinking
			});
			answerMarkdown = res?.markdown ?? '';
		} catch (err) {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			answerError = (err as any)?.detail ?? String(err);
			answerMarkdown = '';
		}
		loadingAnswer = false;
	};

	const selectRow = (row: AnswerRow) => {
		selectedRow = row;
		loadAnswer(row);
	};

	// Re-fetch the current answer when "show thinking" is toggled.
	const onThinkingChange = () => {
		if (selectedRow) {
			loadAnswer(selectedRow);
		}
	};

	$: renderedAnswer = answerMarkdown ? DOMPurify.sanitize(marked.parse(answerMarkdown)) : '';

	onMount(() => {
		loadRuns();
	});
</script>

<div class="flex items-center justify-between mb-2 gap-2 flex-wrap">
	<h2 class="text-sm font-medium text-gray-900 dark:text-white shrink-0">
		{$i18n.t('Answers')}
	</h2>

	<div class="flex items-center gap-2 flex-wrap justify-end min-w-0">
		{#if loadingRuns}
			<Spinner className="size-4" />
		{:else if runOptions.length > 0}
			<NativeSelect
				bind:value={selectedRun}
				options={runOptions}
				className="w-fit max-w-[28rem] truncate rounded-sm px-2 py-1 text-xs bg-transparent outline-none border border-gray-100 dark:border-gray-800"
			/>
		{:else}
			<span class="text-xs text-gray-400">{$i18n.t('No runs found')}</span>
		{/if}

		<NativeSelect
			bind:value={filter}
			options={filterOptions}
			className="w-fit rounded-sm px-2 py-1 text-xs bg-transparent outline-none border border-gray-100 dark:border-gray-800"
		/>

		<label class="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-300 select-none">
			<input type="checkbox" bind:checked={thinking} on:change={onThinkingChange} />
			{$i18n.t('Show thinking')}
		</label>
	</div>
</div>

{#if runsError}
	<div class="text-xs text-red-600 dark:text-red-400 px-0.5 py-2">{runsError}</div>
{/if}

{#if loadingRows}
	<div class="my-10 flex justify-center">
		<Spinner className="size-5" />
	</div>
{:else}
	<div class="scrollbar-hidden relative whitespace-nowrap overflow-x-auto max-w-full">
		<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
			<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
				<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
					<th scope="col" class="px-2.5 py-2">{$i18n.t('Benchmark')}</th>
					<th scope="col" class="px-2.5 py-2">{$i18n.t('Item')}</th>
					<th scope="col" class="px-2.5 py-2">{$i18n.t('Outcome')}</th>
					<th scope="col" class="px-2.5 py-2">{$i18n.t('Reason')}</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as row (row.benchmark + '::' + row.item_id)}
					<tr
						class="dark:border-gray-850 text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors {selectedRow &&
						selectedRow.benchmark === row.benchmark &&
						selectedRow.item_id === row.item_id
							? 'bg-gray-50 dark:bg-gray-800'
							: ''}"
						on:click={() => selectRow(row)}
					>
						<td class="px-3 py-1 font-normal text-gray-900 dark:text-white whitespace-normal">
							{row.benchmark}
						</td>
						<td class="px-3 py-1 whitespace-normal">{row.item_id}</td>
						<td class="px-3 py-1"><Badge type={outcomeBadgeType(row.outcome)} content={row.outcome} /></td>
						<td class="px-3 py-1 whitespace-normal max-w-[24rem] truncate" title={row.reason ?? ''}>
							{row.reason ?? '—'}
						</td>
					</tr>
				{/each}
				{#if rows.length === 0}
					<tr>
						<td colspan="4" class="px-3 py-2 text-center text-gray-400">{$i18n.t('No data')}</td>
					</tr>
				{/if}
			</tbody>
		</table>
	</div>
{/if}

<div class="mt-6 pt-4 border-t border-gray-50 dark:border-gray-850/30">
	<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1.5 px-0.5">
		{$i18n.t('Answer')}
	</div>

	{#if loadingAnswer}
		<div class="my-6 flex justify-center">
			<Spinner className="size-5" />
		</div>
	{:else if answerError}
		<div class="text-xs text-red-600 dark:text-red-400 px-0.5">{answerError}</div>
	{:else if renderedAnswer}
		<div class="prose dark:prose-invert max-w-none text-sm">
			{@html renderedAnswer}
		</div>
	{:else}
		<div class="text-xs text-gray-400 px-0.5">
			{$i18n.t('Select a row above to view its answer.')}
		</div>
	{/if}
</div>
