<script lang="ts">
	import { onDestroy, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import {
		getTestOptions,
		startTestRun,
		cancelTestRun,
		streamTestRun,
		parseBenchmarksEventStream,
		type TestRunForm
	} from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Badge from '$lib/components/common/Badge.svelte';
	import NativeSelect from '$lib/components/common/NativeSelect.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	const inputClass =
		'w-full h-8 rounded-lg border border-gray-100/50 bg-gray-50/40 px-2.5 text-xs text-gray-700 outline-hidden transition-colors placeholder:text-gray-300 focus:border-blue-400 dark:border-white/[0.04] dark:bg-white/[0.03] dark:text-gray-300 dark:placeholder:text-gray-700 dark:focus:border-blue-500';

	const buttonClass =
		'px-3.5 py-1.5 text-sm font-normal bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full disabled:opacity-40 disabled:cursor-not-allowed';

	const secondaryButtonClass =
		'px-3.5 py-1.5 text-sm font-normal bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-gray-100 transition rounded-full disabled:opacity-40 disabled:cursor-not-allowed';

	// -------------------------------------------------------------------
	// Options
	// -------------------------------------------------------------------
	let tiers: string[] = [];
	let benchmarks: string[] = [];
	let systems: string[] = [];
	let optionsLoading = true;
	let error: string | null = null;

	let suite = '';
	let benchmark = '';
	let system = '';
	let resume = false;
	let slice = '';

	const loadOptions = async () => {
		optionsLoading = true;
		try {
			const res = await getTestOptions(localStorage.token);
			tiers = res?.tiers ?? [];
			benchmarks = res?.benchmarks ?? [];
			systems = res?.systems ?? [];
			if (tiers.length > 0 && !suite) suite = tiers[0];
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to load options');
		}
		optionsLoading = false;
	};

	loadOptions();

	// -------------------------------------------------------------------
	// Run state
	// -------------------------------------------------------------------
	type ItemEvent = {
		type: 'item';
		benchmark: string;
		item_id: string;
		outcome: string;
		reason?: string;
		i: number;
		total: number;
		passed: number;
		attempted: number;
	};

	let running = false;
	let cancelling = false;
	let starting = false;

	let suiteRunId: string | null = null;
	let total = 0;
	let skipped = 0;
	let latest: ItemEvent | null = null;
	let items: ItemEvent[] = [];
	let done: { passed: number; attempted: number; cancelled: boolean } | null = null;
	let streamError: string | null = null;

	let abortController: AbortController | null = null;
	let logContainer: HTMLDivElement | null = null;

	const outcomeBadgeType = (outcome: string): 'success' | 'error' | 'muted' | 'info' => {
		const o = (outcome ?? '').toLowerCase();
		if (o === 'pass' || o === 'passed') return 'success';
		if (o === 'fail' || o === 'failed') return 'error';
		if (o === 'skip' || o === 'skipped') return 'muted';
		return 'info';
	};

	const scrollToBottom = async () => {
		await new Promise((r) => setTimeout(r, 0));
		if (logContainer) {
			logContainer.scrollTop = logContainer.scrollHeight;
		}
	};

	const handleRun = async () => {
		starting = true;
		error = null;
		streamError = null;
		done = null;
		items = [];
		latest = null;
		suiteRunId = null;
		total = 0;
		skipped = 0;

		try {
			const form: TestRunForm = { suite };
			if (benchmark) form.benchmark = benchmark;
			if (system) form.system = system;
			if (slice.trim()) form.slice = slice.trim();
			if (resume) form.resume = true;

			await startTestRun(localStorage.token, form);
			startStream();
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to start test run');
		}
		starting = false;
	};

	const handleCancel = async () => {
		cancelling = true;
		try {
			await cancelTestRun(localStorage.token);
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to cancel test run');
		}
		cancelling = false;
	};

	const startStream = async () => {
		stopStream();
		running = true;

		const [res, controller] = await streamTestRun(localStorage.token);
		abortController = controller;

		if (!res || !res.body) {
			running = false;
			return;
		}

		try {
			for await (const { event, data } of parseBenchmarksEventStream<
				| { type: 'start'; suite_run_id: string; total: number; skipped: number }
				| ItemEvent
				| { type: 'done'; passed: number; attempted: number; cancelled: boolean }
				| { type: 'error'; message: string }
			>(res.body)) {
				if (event === 'done') {
					break;
				}
				if (!data) continue;

				if (data.type === 'start') {
					suiteRunId = data.suite_run_id;
					total = data.total;
					skipped = data.skipped;
				} else if (data.type === 'item') {
					latest = data;
					items = [...items, data];
					scrollToBottom();
				} else if (data.type === 'done') {
					done = { passed: data.passed, attempted: data.attempted, cancelled: data.cancelled };
				} else if (data.type === 'error') {
					streamError = data.message;
					break;
				}
			}
		} catch (err: any) {
			console.error(err);
		}
		running = false;
	};

	const stopStream = () => {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		running = false;
	};

	onDestroy(() => {
		stopStream();
	});
</script>

<div class="flex flex-col gap-4">
	<div class="flex items-center justify-between">
		<h2 class="text-sm font-medium text-gray-900 dark:text-white">{$i18n.t('Tests')}</h2>
	</div>

	{#if error}
		<div
			class="text-xs text-red-700 dark:text-red-200 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
		>
			{typeof error === 'string' ? error : JSON.stringify(error)}
		</div>
	{/if}

	{#if optionsLoading}
		<div class="flex justify-center py-4">
			<Spinner className="size-4" />
		</div>
	{:else}
		<div class="grid md:grid-cols-2 lg:grid-cols-4 gap-2">
			<div class="flex flex-col gap-1">
				<label class="text-[0.6875rem] text-gray-400" for="tests-tier">{$i18n.t('Tier')}</label>
				<NativeSelect
					bind:value={suite}
					options={tiers}
					placeholder={$i18n.t('Select a tier')}
					className="{inputClass} pr-8"
				/>
			</div>
			<div class="flex flex-col gap-1">
				<label class="text-[0.6875rem] text-gray-400" for="tests-benchmark"
					>{$i18n.t('Benchmark')}</label
				>
				<NativeSelect
					bind:value={benchmark}
					options={benchmarks}
					placeholder={$i18n.t('All benchmarks')}
					className="{inputClass} pr-8"
				/>
			</div>
			<div class="flex flex-col gap-1">
				<label class="text-[0.6875rem] text-gray-400" for="tests-system">{$i18n.t('System')}</label
				>
				<NativeSelect
					bind:value={system}
					options={systems}
					placeholder={$i18n.t('All systems')}
					className="{inputClass} pr-8"
				/>
			</div>
			<div class="flex flex-col gap-1">
				<label class="text-[0.6875rem] text-gray-400" for="tests-slice">{$i18n.t('Slice')}</label>
				<input
					id="tests-slice"
					class={inputClass}
					type="text"
					bind:value={slice}
					placeholder="e.g. 0:50"
				/>
			</div>
		</div>

		<label class="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400 w-fit">
			<input type="checkbox" bind:checked={resume} class="rounded" />
			{$i18n.t('Resume previous run')}
		</label>
	{/if}

	<!-- Actions -->
	<div class="flex items-center gap-2">
		<button class={buttonClass} on:click={handleRun} disabled={starting || running || !suite}>
			{#if starting}
				<Spinner className="size-3" />
			{:else}
				{$i18n.t('Run')}
			{/if}
		</button>
		<button class={secondaryButtonClass} on:click={handleCancel} disabled={cancelling || !running}>
			{#if cancelling}
				<Spinner className="size-3" />
			{:else}
				{$i18n.t('Cancel')}
			{/if}
		</button>

		{#if running}
			<div class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 ml-2">
				<Spinner className="size-3" />
				{#if latest}
					{latest.i}/{latest.total || total} - {latest.passed} {$i18n.t('passed')}
				{:else}
					{$i18n.t('Starting...')}
				{/if}
			</div>
		{/if}
	</div>

	{#if streamError}
		<div
			class="text-xs text-red-700 dark:text-red-200 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
		>
			{streamError}
		</div>
	{/if}

	{#if suiteRunId}
		<div class="text-xs text-gray-500 dark:text-gray-400">
			{$i18n.t('Suite run')}: <span class="text-gray-900 dark:text-white">{suiteRunId}</span>
			· {$i18n.t('Total')}: {total} · {$i18n.t('Skipped')}: {skipped}
		</div>
	{/if}

	<!-- Item log -->
	<div class="flex flex-col gap-1">
		<div class="text-xs font-normal text-gray-700 dark:text-gray-300">{$i18n.t('Item Log')}</div>
		<div
			bind:this={logContainer}
			class="border border-gray-100 dark:border-gray-850 rounded-lg h-72 overflow-y-auto text-xs"
		>
			{#each items as item, idx (idx)}
				<div
					class="flex items-center gap-2 px-2.5 py-1 border-b border-gray-50 dark:border-gray-850/50"
				>
					<span class="text-gray-400 w-10 shrink-0">{item.i}</span>
					<span class="text-gray-700 dark:text-gray-300 truncate">{item.benchmark}</span>
					<span class="text-gray-400 truncate flex-1">{item.item_id}</span>
					<Badge type={outcomeBadgeType(item.outcome)} content={item.outcome} />
					{#if item.reason}
						<span class="text-gray-400 truncate max-w-[12rem]">{item.reason}</span>
					{/if}
				</div>
			{/each}
			{#if items.length === 0}
				<div class="px-2.5 py-4 text-center text-gray-400">{$i18n.t('No items yet')}</div>
			{/if}
		</div>
	</div>

	<!-- Final summary -->
	{#if done}
		<div
			class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs border border-gray-100 dark:border-gray-850 rounded-lg px-3 py-2.5"
		>
			<span class="text-sm font-medium text-gray-900 dark:text-white">{$i18n.t('Done')}</span>
			<span class="text-gray-500 dark:text-gray-400">
				{done.passed} / {done.attempted} {$i18n.t('passed')}
			</span>
			{#if done.cancelled}
				<Badge type="warning" content={$i18n.t('Cancelled')} />
			{/if}
		</div>
	{/if}
</div>
