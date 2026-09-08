<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { getCompare, exportAnswers, getTestOptions, type CompareBy } from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import NativeSelect from '$lib/components/common/NativeSelect.svelte';
	import ChevronUp from '$lib/components/icons/ChevronUp.svelte';
	import ChevronDown from '$lib/components/icons/ChevronDown.svelte';

	const i18n = getContext('i18n');

	// -------------------------------------------------------------------
	// Filters
	// -------------------------------------------------------------------
	let by: CompareBy = 'config';
	let tier = '';
	let baseline = '';

	let tierOptions: ({ label?: string; value: string } | string)[] = [
		{ value: '', label: $i18n.t('All') }
	];

	const loadTierOptions = async () => {
		try {
			const res = await getTestOptions(localStorage.token);
			tierOptions = [{ value: '', label: $i18n.t('All') }, ...(res?.tiers ?? [])];
		} catch (err) {
			console.error('Failed to load tier options:', err);
		}
	};

	$: byOptions = [
		{ value: 'config', label: $i18n.t('Config') },
		{ value: 'benchmark', label: $i18n.t('Benchmark') },
		{ value: 'failures', label: $i18n.t('Failures') },
		{ value: 'serving', label: $i18n.t('Serving') }
	];

	// -------------------------------------------------------------------
	// Data
	// -------------------------------------------------------------------
	let loading = true;
	let loadError: string | null = null;
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let data: any = null;

	// The response shape genuinely varies with `by`:
	//   - config / failures -> { rows, notes }
	//   - benchmark         -> { columns, rows, notes }
	//   - serving           -> { derived_columns, derived, notes }
	// `columns`/`rows` below normalize all four into one generic
	// (columns: string[], rows: Record<string, any>[]) shape that the
	// table markup renders without caring which mode produced it. For
	// config/failures there's no explicit column list from the backend,
	// so the columns are derived from the keys of the first row (in the
	// order the backend returned them) per the instruction not to
	// hardcode a fixed column list.
	$: columns =
		by === 'benchmark'
			? (data?.columns ?? [])
			: by === 'serving'
				? (data?.derived_columns ?? [])
				: (data?.rows?.[0] ? Object.keys(data.rows[0]) : []);

	$: rows = by === 'serving' ? (data?.derived ?? []) : (data?.rows ?? []);
	$: notes = [...new Set(data?.notes ?? [])] as string[];

	// -------------------------------------------------------------------
	// Sorting (generic, keyed by whatever column header was clicked)
	// -------------------------------------------------------------------
	let orderBy: string | null = null;
	let direction: 'asc' | 'desc' = 'asc';

	const toggleSort = (col: string) => {
		if (orderBy === col) {
			direction = direction === 'asc' ? 'desc' : 'asc';
		} else {
			orderBy = col;
			direction = 'asc';
		}
	};

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	const compareValues = (a: any, b: any): number => {
		if (a === undefined || a === null) return b === undefined || b === null ? 0 : -1;
		if (b === undefined || b === null) return 1;
		if (typeof a === 'number' && typeof b === 'number') return a - b;
		const an = Number(a);
		const bn = Number(b);
		if (!Number.isNaN(an) && !Number.isNaN(bn) && a !== '' && b !== '') return an - bn;
		return String(a).localeCompare(String(b));
	};

	$: sortedRows = orderBy
		? [...rows].sort((a, b) => {
				const cmp = compareValues(a?.[orderBy as string], b?.[orderBy as string]);
				return direction === 'asc' ? cmp : -cmp;
			})
		: rows;

	// Column set changes across `by` modes, so a previously-selected sort
	// column may no longer exist; reset rather than sort on `undefined`.
	$: {
		by;
		orderBy = null;
		direction = 'asc';
	}

	const humanizeHeader = (col: string): string =>
		col
			.replace(/[_-]+/g, ' ')
			.replace(/\b\w/g, (c) => c.toUpperCase());

	// Percentage-shaped columns (a 0-1 fraction, or exactly 1) read much
	// faster as "87.5%" than as "0.875" or a bare "1" that could be
	// mistaken for a count.
	const isRateColumn = (col: string): boolean => /(^|_)(pass_rate|rate)$/i.test(col);

	const formatRate = (value: number): string => `${(value * 100).toFixed(1).replace(/\.0$/, '')}%`;

	// Both known nested-object cells (`flags`: a plain key/value map of
	// serving overrides; `per_benchmark`: benchmark name -> {passed,
	// attempted, pass_rate}) render as a short inline summary instead of
	// the default `String(value)` -> "[object Object]".
	const formatObject = (value: Record<string, unknown>): string => {
		const entries = Object.entries(value);
		if (!entries.length) return '—';
		return entries
			.map(([key, val]) => {
				if (val && typeof val === 'object' && !Array.isArray(val)) {
					const cell = val as Record<string, unknown>;
					if ('passed' in cell && 'attempted' in cell) return `${key} ${cell.passed}/${cell.attempted}`;
					return `${key}: ${formatObject(cell)}`;
				}
				return `${key}=${formatCell(val)}`;
			})
			.join(', ');
	};

	const formatCell = (value: unknown, col: string = ''): string => {
		if (value === null || value === undefined || value === '') return '—';
		if (typeof value === 'boolean') return value ? $i18n.t('Yes') : $i18n.t('No');
		if (typeof value === 'number') {
			if (isRateColumn(col) && value >= 0 && value <= 1) return formatRate(value);
			return Number.isInteger(value) ? value.toString() : value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
		}
		if (Array.isArray(value)) return value.length ? value.map((v) => formatCell(v)).join(', ') : '—';
		if (typeof value === 'object') return formatObject(value as Record<string, unknown>);
		return String(value);
	};

	const isWarningNote = (note: string): boolean => /^\s*warning\b/i.test(note);

	// -------------------------------------------------------------------
	// Load
	// -------------------------------------------------------------------
	const loadCompare = async () => {
		loading = true;
		loadError = null;
		try {
			data = await getCompare(localStorage.token, {
				by,
				tier: tier.trim() ? tier.trim() : null,
				baseline: by === 'config' && baseline.trim() ? baseline.trim() : null
			});
		} catch (err) {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			loadError = (err as any)?.detail ?? String(err);
			data = null;
		}
		loading = false;
	};

	onMount(() => {
		loadCompare();
		loadTierOptions();
	});

	// Reload automatically whenever the mode changes (distinct request
	// shape); tier/baseline are applied explicitly via the Apply button
	// below so free typing doesn't fire a request per keystroke.
	$: if (by) {
		loadCompare();
	}

	// -------------------------------------------------------------------
	// Export answers
	// -------------------------------------------------------------------
	let exportRun = '';
	let exportLoading = false;
	let exportError: string | null = null;
	let exportResult: { exported: number; directory: string } | null = null;

	const doExportAnswers = async () => {
		if (!exportRun.trim()) return;
		exportLoading = true;
		exportError = null;
		exportResult = null;
		try {
			exportResult = await exportAnswers(localStorage.token, exportRun.trim());
		} catch (err) {
			// eslint-disable-next-line @typescript-eslint/no-explicit-any
			exportError = (err as any)?.detail ?? String(err);
		}
		exportLoading = false;
	};
</script>

<div class="flex items-center justify-between mb-2 gap-2 flex-wrap">
	<h2 class="text-sm font-medium text-gray-900 dark:text-white shrink-0">
		{$i18n.t('Compare')}
	</h2>

	<div class="flex items-center gap-2 flex-wrap justify-end min-w-0">
		<NativeSelect
			bind:value={by}
			options={byOptions}
			className="w-fit rounded-sm px-2 py-1 text-xs bg-transparent outline-none border border-gray-100 dark:border-gray-800"
		/>

		<NativeSelect
			bind:value={tier}
			options={tierOptions}
			className="w-36 rounded-sm px-2 py-1 text-xs bg-transparent outline-none border border-gray-100 dark:border-gray-800"
		/>

		{#if by === 'config'}
			<input
				type="text"
				bind:value={baseline}
				placeholder={$i18n.t('Baseline config id')}
				on:keydown={(e) => e.key === 'Enter' && loadCompare()}
				class="w-44 rounded-sm px-2 py-1 text-xs bg-transparent outline-none border border-gray-100 dark:border-gray-800"
			/>
		{/if}

		<button
			class="px-2.5 py-1 text-xs rounded-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 transition"
			on:click={loadCompare}
		>
			{$i18n.t('Apply')}
		</button>
	</div>
</div>

{#if loading}
	<div class="my-10 flex justify-center">
		<Spinner className="size-5" />
	</div>
{:else if loadError}
	<div class="text-xs text-red-600 dark:text-red-400 px-0.5 py-4">
		{loadError}
	</div>
{:else}
	<div class="relative">
		<div class="relative whitespace-nowrap overflow-x-auto max-w-full">
			<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
				<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
					<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
						{#each columns as col}
							<th
								scope="col"
								class="px-2.5 py-2 cursor-pointer select-none"
								on:click={() => toggleSort(col)}
							>
								<div class="flex gap-1.5 items-center">
									{humanizeHeader(col)}
									{#if orderBy === col}
										<span class="font-normal">
											{#if direction === 'asc'}<ChevronUp className="size-2" />{:else}<ChevronDown
													className="size-2"
												/>{/if}
										</span>
									{:else}
										<span class="invisible"><ChevronUp className="size-2" /></span>
									{/if}
								</div>
							</th>
						{/each}
					</tr>
				</thead>
				<tbody>
					{#each sortedRows as row, idx (idx)}
						<tr class="dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors">
							{#each columns as col}
								<td class="px-3 py-1">{formatCell(row?.[col], col)}</td>
							{/each}
						</tr>
					{/each}
					{#if sortedRows.length === 0}
						<tr>
							<td colspan={Math.max(columns.length, 1)} class="px-3 py-2 text-center text-gray-400">
								{$i18n.t('No data')}
							</td>
						</tr>
					{/if}
				</tbody>
			</table>
		</div>
		<div
			class="pointer-events-none absolute top-0 right-0 bottom-0 w-8 bg-linear-to-l from-white dark:from-gray-900 to-transparent"
		></div>
	</div>

	{#if notes.length > 0}
		<ul class="mt-2 space-y-1 px-0.5">
			{#each notes as note}
				{#if isWarningNote(note)}
					<li
						class="text-xs bg-yellow-500/20 text-yellow-700 dark:text-yellow-200 rounded-sm px-2 py-1 w-fit"
					>
						{note}
					</li>
				{:else}
					<li class="text-xs text-gray-500 dark:text-gray-400">{note}</li>
				{/if}
			{/each}
		</ul>
	{/if}
{/if}

<div class="mt-6 pt-4 border-t border-gray-50 dark:border-gray-850/30">
	<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1.5 px-0.5">
		{$i18n.t('Export Answers')}
	</div>
	<div class="flex items-center gap-2 flex-wrap">
		<input
			type="text"
			bind:value={exportRun}
			placeholder={$i18n.t('Suite run id')}
			on:keydown={(e) => e.key === 'Enter' && doExportAnswers()}
			class="w-56 rounded-sm px-2 py-1 text-xs bg-transparent outline-none border border-gray-100 dark:border-gray-800"
		/>
		<button
			class="px-2.5 py-1 text-xs rounded-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-200 transition disabled:opacity-50 disabled:cursor-not-allowed"
			disabled={exportLoading || !exportRun.trim()}
			on:click={doExportAnswers}
		>
			{#if exportLoading}
				<Spinner className="size-3" />
			{:else}
				{$i18n.t('Export')}
			{/if}
		</button>

		{#if exportResult}
			<span class="text-xs text-green-700 dark:text-green-300">
				{$i18n.t('Exported {{count}} answers to {{directory}}', {
					count: exportResult.exported,
					directory: exportResult.directory
				})}
			</span>
		{/if}

		{#if exportError}
			<span class="text-xs text-red-600 dark:text-red-400">{exportError}</span>
		{/if}
	</div>
</div>
