<script lang="ts">
	import { getContext, onDestroy, onMount } from 'svelte';
	import DOMPurify from 'dompurify';
	import { marked } from 'marked';

	import {
		startTune,
		resumeTune,
		stopTune,
		getTuneStatus,
		getRecentSweeps,
		getTuneGrids,
		getServeProfiles,
		getTestOptions,
		streamTuneLog,
		parseBenchmarksEventStream
	} from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import NativeSelect from '$lib/components/common/NativeSelect.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Badge from '$lib/components/common/Badge.svelte';

	const i18n = getContext('i18n');

	const errDetail = (err: unknown, fallback: string): string => {
		if (typeof err === 'string') return err;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		return (err as any)?.detail ?? fallback;
	};

	// -----------------------------------------------------------------------------
	// Start-sweep form
	// -----------------------------------------------------------------------------

	let profile = '';
	let tier = 'smoke';
	let benchmark = '';
	let system = '';
	let grid = '';
	let budget = 'interactive';
	let candidates: number | null = null;
	let roundItems: number | null = null;
	let eta = 2;
	let seed = 0;
	let stageExplore = true;
	let stageRefine = true;

	let profileOptions: ({ label?: string; value: string } | string)[] = [
		{ value: '', label: $i18n.t('Default') }
	];
	let tierOptions: ({ label?: string; value: string } | string)[] = [
		'smoke',
		'standard',
		'full'
	];
	let benchmarkOptions: ({ label?: string; value: string } | string)[] = [
		{ value: '', label: $i18n.t('Default') }
	];
	let systemOptions: ({ label?: string; value: string } | string)[] = [
		{ value: '', label: $i18n.t('Default') }
	];
	let gridOptions: ({ label?: string; value: string } | string)[] = [
		{ value: '', label: $i18n.t('Default') }
	];

	const loadOptions = async () => {
		try {
			const [profilesRes, testOptionsRes, gridsRes] = await Promise.all([
				getServeProfiles(localStorage.token),
				getTestOptions(localStorage.token),
				getTuneGrids(localStorage.token)
			]);
			profileOptions = [
				{ value: '', label: $i18n.t('Default') },
				...(profilesRes?.profiles ?? [])
			];
			tierOptions = testOptionsRes?.tiers?.length ? testOptionsRes.tiers : ['smoke', 'standard', 'full'];
			benchmarkOptions = [
				{ value: '', label: $i18n.t('Default') },
				...(testOptionsRes?.benchmarks ?? [])
			];
			systemOptions = [
				{ value: '', label: $i18n.t('Default') },
				...(testOptionsRes?.systems ?? [])
			];
			gridOptions = [{ value: '', label: $i18n.t('Default') }, ...(gridsRes?.grids ?? [])];
		} catch (err) {
			// Field option lists are a convenience, not required to use the form -
			// leave the fields on their free-text-friendly defaults on failure.
			console.error('Failed to load tune field options:', err);
		}
	};

	let starting = false;
	let startError: string | null = null;

	let stopping = false;
	let showStopConfirm = false;

	let resuming = false;
	let resumeError: string | null = null;

	// The sweep id typed for resume/view, and which sweep the status panel
	// below is currently tracking. `null` means "the current/latest sweep",
	// matching getTuneStatus/streamTuneLog's own `sweepId ?? null` default.
	let sweepIdInput = '';
	let viewSweepId: string | null = null;

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let recentSweeps: any[] = [];
	let loadingRecent = false;

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	let status: any = null;
	let statusError: string | null = null;
	let streaming = false;
	let refreshing = false;
	let streamController: AbortController | null = null;

	const loadRecentSweeps = async () => {
		loadingRecent = true;
		try {
			const res = await getRecentSweeps(localStorage.token, 20);
			recentSweeps = res?.sweeps ?? [];
		} catch (e) {
			console.error('Failed to load recent sweeps:', e);
		} finally {
			loadingRecent = false;
		}
	};

	const stopStreaming = () => {
		if (streamController) {
			streamController.abort();
		}
		streamController = null;
		streaming = false;
	};

	// Drives the live status panel. `streamTuneLog` is a polling-based feed
	// that re-sends the full status object every ~2s (there's no subprocess
	// to tail here — the sweep runs in-process) and ends with `event: done`,
	// so each received payload simply replaces `status` wholesale.
	const startStreaming = async (sweepId: string | null) => {
		stopStreaming();
		statusError = null;

		const [res, controller] = await streamTuneLog(localStorage.token, sweepId).catch((err) => {
			statusError = errDetail(err, $i18n.t('Failed to start live status stream'));
			return [null, null] as [Response | null, AbortController | null];
		});

		if (!res) {
			return;
		}

		if (!res.ok || !res.body) {
			try {
				const body = await res.json();
				statusError = body?.detail ?? $i18n.t('Failed to start live status stream');
			} catch {
				statusError = $i18n.t('Failed to start live status stream');
			}
			return;
		}

		streamController = controller;
		streaming = true;

		try {
			for await (const evt of parseBenchmarksEventStream(res.body)) {
				if (evt.event === 'done') break;
				if (evt.data) status = evt.data;
			}
		} catch (err) {
			// An abort from stopStreaming()/component teardown is expected, not an error.
			if ((err as Error)?.name !== 'AbortError') {
				console.error('Tune log stream error:', err);
				statusError = errDetail(err, $i18n.t('Live status stream ended unexpectedly'));
			}
		} finally {
			streaming = false;
			streamController = null;
		}
	};

	const manualRefresh = async () => {
		refreshing = true;
		statusError = null;
		try {
			status = await getTuneStatus(localStorage.token, viewSweepId);
		} catch (err) {
			statusError = errDetail(err, $i18n.t('Failed to load sweep status'));
		} finally {
			refreshing = false;
		}
	};

	const selectSweep = (sweepId: string | null) => {
		viewSweepId = sweepId;
		sweepIdInput = sweepId ?? '';
		startStreaming(sweepId);
	};

	const handleStart = async () => {
		starting = true;
		startError = null;
		try {
			const stages = [...(stageExplore ? ['explore'] : []), ...(stageRefine ? ['refine'] : [])];

			await startTune(localStorage.token, {
				profile: profile.trim() || undefined,
				tier,
				benchmark: benchmark.trim() || undefined,
				system: system.trim() || undefined,
				grid: grid.trim() || undefined,
				budget,
				candidates: candidates ?? undefined,
				round_items: roundItems ?? undefined,
				eta,
				seed,
				stages: stages.length > 0 ? stages : undefined
			});

			await loadRecentSweeps();
			selectSweep(null);
		} catch (err) {
			startError = errDetail(err, $i18n.t('Failed to start sweep'));
		} finally {
			starting = false;
		}
	};

	const handleStopConfirmed = async () => {
		stopping = true;
		try {
			await stopTune(localStorage.token);
			stopStreaming();
			await manualRefresh();
			await loadRecentSweeps();
		} catch (err) {
			statusError = errDetail(err, $i18n.t('Failed to stop sweep'));
		} finally {
			stopping = false;
		}
	};

	const handleResume = async () => {
		resuming = true;
		resumeError = null;
		try {
			const sweepId = sweepIdInput.trim() || null;
			await resumeTune(localStorage.token, sweepId);
			await loadRecentSweeps();
			selectSweep(sweepId);
		} catch (err) {
			resumeError = errDetail(err, $i18n.t('Failed to resume sweep'));
		} finally {
			resuming = false;
		}
	};

	onMount(() => {
		loadRecentSweeps();
		loadOptions();
		// Watch whatever sweep is currently active/latest, if any. If none
		// exists yet, startStreaming sets statusError and the manual Refresh
		// button remains available.
		startStreaming(null);
	});

	onDestroy(() => {
		stopStreaming();
	});

	$: sortedCandidates = [...(status?.candidates ?? [])].sort(
		(a, b) => (b?.score ?? -Infinity) - (a?.score ?? -Infinity)
	);

	const candidateName = (c: Record<string, unknown>): string =>
		(c.label as string) || (c.config_id as string) || (c.candidate_sha as string) || $i18n.t('(unnamed)');

	const fmtNum = (n: unknown, digits = 3): string =>
		typeof n === 'number' ? n.toFixed(digits) : '—';
</script>

<div class="flex flex-col gap-4">
	<div>
		<h2 class="text-lg font-medium text-gray-900 dark:text-white">{$i18n.t('Tune')}</h2>
		<div class="text-xs text-gray-500 dark:text-gray-400">
			{$i18n.t('Run a configuration-search sweep to find better serving settings.')}
		</div>
	</div>

	<!-- Start-sweep form -->
	<form
		class="flex flex-col gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-850"
		on:submit|preventDefault={handleStart}
	>
		<div class="flex flex-wrap gap-3">
			<div class="flex flex-col gap-1">
				<label for="tune-profile" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Profile')}</label
				>
				<NativeSelect
					bind:value={profile}
					options={profileOptions}
					className="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-40"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-tier" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Tier')}</label
				>
				<NativeSelect
					bind:value={tier}
					options={tierOptions}
					className="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-32"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-benchmark" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Benchmark')}</label
				>
				<NativeSelect
					bind:value={benchmark}
					options={benchmarkOptions}
					className="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-40"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-system" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('System')}</label
				>
				<NativeSelect
					bind:value={system}
					options={systemOptions}
					className="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-40"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-grid" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Grid')}</label
				>
				<NativeSelect
					bind:value={grid}
					options={gridOptions}
					className="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-40"
				/>
			</div>
		</div>

		<div class="flex flex-wrap gap-3 items-end">
			<div class="flex flex-col gap-1">
				<label for="tune-budget" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Budget')}</label
				>
				<NativeSelect
					bind:value={budget}
					options={['interactive', 'overnight', 'multiday', 'trials']}
					className="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-36"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-candidates" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Candidates')}</label
				>
				<input
					id="tune-candidates"
					type="number"
					min="1"
					bind:value={candidates}
					placeholder={$i18n.t('Auto')}
					class="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-24"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-round-items" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Round items')}</label
				>
				<input
					id="tune-round-items"
					type="number"
					min="1"
					bind:value={roundItems}
					placeholder={$i18n.t('Auto')}
					class="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-24"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-eta" class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Eta')}</label>
				<input
					id="tune-eta"
					type="number"
					min="1"
					bind:value={eta}
					class="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-20"
				/>
			</div>

			<div class="flex flex-col gap-1">
				<label for="tune-seed" class="text-xs text-gray-500 dark:text-gray-400"
					>{$i18n.t('Seed')}</label
				>
				<input
					id="tune-seed"
					type="number"
					min="0"
					bind:value={seed}
					class="text-sm px-2.5 py-1.5 rounded-lg bg-white dark:bg-gray-900 outline-none w-20"
				/>
			</div>

			<div class="flex items-center gap-3 pb-1.5">
				<label class="flex items-center gap-1.5 text-sm text-gray-700 dark:text-gray-300">
					<input type="checkbox" bind:checked={stageExplore} class="size-4" />
					{$i18n.t('Explore')}
				</label>
				<label class="flex items-center gap-1.5 text-sm text-gray-700 dark:text-gray-300">
					<input type="checkbox" bind:checked={stageRefine} class="size-4" />
					{$i18n.t('Refine')}
				</label>
			</div>

			<button
				type="submit"
				disabled={starting || status?.running === true}
				class="px-3.5 py-1.5 text-sm font-medium rounded-lg bg-black text-white dark:bg-white dark:text-black disabled:opacity-50 flex items-center gap-2"
			>
				{#if starting}
					<Spinner className="size-3.5" />
				{/if}
				{$i18n.t('Start')}
			</button>

			<button
				type="button"
				disabled={stopping || status?.running !== true}
				on:click={() => (showStopConfirm = true)}
				class="px-3.5 py-1.5 text-sm font-medium rounded-lg bg-red-600 text-white disabled:opacity-50 flex items-center gap-2"
			>
				{#if stopping}
					<Spinner className="size-3.5" />
				{/if}
				{$i18n.t('Stop')}
			</button>
		</div>

		{#if startError}
			<div class="text-sm text-red-700 dark:text-red-200 bg-red-500/10 rounded-lg px-3 py-2">
				{startError}
			</div>
		{/if}
	</form>

	<ConfirmDialog
		bind:show={showStopConfirm}
		title={$i18n.t('Stop sweep')}
		message={$i18n.t(
			'Are you sure you want to stop the running sweep? It can be resumed later, but any in-progress round is lost.'
		)}
		confirmLabel={$i18n.t('Stop')}
		on:confirm={handleStopConfirmed}
	/>

	<!-- Resume / recent sweeps -->
	<div class="flex flex-wrap items-end gap-3">
		<div class="flex flex-col gap-1">
			<label for="tune-sweep-id" class="text-xs text-gray-500 dark:text-gray-400"
				>{$i18n.t('Sweep ID')}</label
			>
			<input
				id="tune-sweep-id"
				type="text"
				bind:value={sweepIdInput}
				placeholder={$i18n.t('Latest')}
				class="text-sm px-2.5 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-850 outline-none w-56"
			/>
		</div>

		<button
			type="button"
			disabled={resuming}
			on:click={handleResume}
			class="px-3 py-1.5 text-sm font-medium rounded-lg bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white disabled:opacity-50 flex items-center gap-2"
		>
			{#if resuming}
				<Spinner className="size-3.5" />
			{/if}
			{$i18n.t('Resume')}
		</button>

		<button
			type="button"
			on:click={() => selectSweep(sweepIdInput.trim() || null)}
			class="px-3 py-1.5 text-sm font-medium rounded-lg bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white"
		>
			{$i18n.t('View')}
		</button>

		<div class="flex flex-col gap-1 grow min-w-[16rem]">
			<label for="tune-recent" class="text-xs text-gray-500 dark:text-gray-400"
				>{$i18n.t('Recent sweeps')}</label
			>
			<select
				id="tune-recent"
				class="text-sm px-2.5 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-850 outline-none w-full"
				on:change={(e) => {
					const val = (e.target as HTMLSelectElement).value;
					if (val) selectSweep(val);
				}}
			>
				<option value="">
					{loadingRecent ? $i18n.t('Loading…') : $i18n.t('Select a sweep')}
				</option>
				{#each recentSweeps as sweep (sweep.sweep_id)}
					<option value={sweep.sweep_id}>
						{sweep.sweep_id} — {sweep.profile ?? '—'} / {sweep.tier ?? '—'} ({sweep.verdict})
					</option>
				{/each}
			</select>
		</div>

		{#if resumeError}
			<div class="text-sm text-red-700 dark:text-red-200 bg-red-500/10 rounded-lg px-3 py-2">
				{resumeError}
			</div>
		{/if}
	</div>

	<!-- Live status panel -->
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-medium text-gray-900 dark:text-white flex items-center gap-2">
			{$i18n.t('Status')}
			{#if streaming}
				<span class="flex items-center gap-1 text-xs font-normal text-green-600 dark:text-green-400">
					<span class="size-1.5 rounded-full bg-green-500 animate-pulse"></span>
					{$i18n.t('Live')}
				</span>
			{/if}
		</h3>
		<button
			type="button"
			disabled={streaming || refreshing}
			on:click={manualRefresh}
			class="px-2.5 py-1 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white disabled:opacity-50 flex items-center gap-1.5"
		>
			{#if refreshing}
				<Spinner className="size-3" />
			{/if}
			{$i18n.t('Refresh')}
		</button>
	</div>

	{#if statusError}
		<div class="text-sm text-red-700 dark:text-red-200 bg-red-500/10 rounded-lg px-3 py-2">
			{statusError}
		</div>
	{/if}

	{#if !status}
		<div class="text-sm text-gray-500 dark:text-gray-400 my-4">
			{$i18n.t('No sweep status to show yet.')}
		</div>
	{:else}
		<!-- Summary -->
		<div class="grid sm:grid-cols-2 md:grid-cols-4 gap-3 text-sm">
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Sweep ID')}</div>
				<div class="font-medium text-gray-900 dark:text-white truncate">{status.sweep_id ?? '—'}</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Verdict')}</div>
				<div class="font-medium text-gray-900 dark:text-white">
					{status.verdict ?? status.ended_reason ?? $i18n.t('running')}
				</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Profile / Tier')}</div>
				<div class="font-medium text-gray-900 dark:text-white truncate">
					{status.profile ?? '—'} / {status.tier ?? '—'}
				</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Budget mode')}</div>
				<div class="font-medium text-gray-900 dark:text-white">{status.budget_mode ?? '—'}</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Benchmark')}</div>
				<div class="font-medium text-gray-900 dark:text-white truncate">
					{status.benchmark ?? '—'}
				</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Item count')}</div>
				<div class="font-medium text-gray-900 dark:text-white">{status.item_count ?? '—'}</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Resumable')}</div>
				<div class="font-medium text-gray-900 dark:text-white">
					{status.resumable ? $i18n.t('Yes') : $i18n.t('No')}
				</div>
			</div>
			<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Cooling share')}</div>
				<div class="font-medium text-gray-900 dark:text-white">
					{fmtNum(status.time?.cooling_share)}
				</div>
			</div>
		</div>

		{#if status.verdict_reason}
			<div class="text-xs text-gray-500 dark:text-gray-400">
				{$i18n.t('Reason')}: {status.verdict_reason}
			</div>
		{/if}

		<!-- Rounds -->
		<div>
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
				{$i18n.t('Rounds')}
			</div>
			<div class="scrollbar-hidden relative whitespace-nowrap overflow-x-auto max-w-full">
				<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
					<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
						<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
							<th class="px-2.5 py-2">#</th>
							<th class="px-2.5 py-2">{$i18n.t('Stage')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Items')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Survivors')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Baseline gen tok/s')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Drift ratio')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Decision')}</th>
						</tr>
					</thead>
					<tbody>
						{#each status.rounds ?? [] as r (r.round)}
							<tr class="dark:border-gray-850 text-xs">
								<td class="px-3 py-1 text-gray-400">{r.round}</td>
								<td class="px-3 py-1">{r.stage ?? '—'}</td>
								<td class="px-3 py-1 text-right">{r.item_from}–{r.item_to}</td>
								<td class="px-3 py-1 text-right">{r.survivors ?? '—'}</td>
								<td class="px-3 py-1 text-right">{fmtNum(r.baseline_gen_tps, 1)}</td>
								<td class="px-3 py-1 text-right">{fmtNum(r.drift_ratio)}</td>
								<td class="px-3 py-1">{r.decision ?? '—'}</td>
							</tr>
						{/each}
						{#if (status.rounds ?? []).length === 0}
							<tr
								><td colspan="7" class="px-3 py-2 text-center text-gray-400"
									>{$i18n.t('No rounds yet')}</td
								></tr
							>
						{/if}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Candidates -->
		<div>
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
				{$i18n.t('Candidates')}
			</div>
			<div class="scrollbar-hidden relative whitespace-nowrap overflow-x-auto max-w-full">
				<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
					<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
						<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
							<th class="px-2.5 py-2">{$i18n.t('Candidate')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Score')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Score %')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Paired items')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Status')}</th>
						</tr>
					</thead>
					<tbody>
						{#each sortedCandidates as c (c.candidate_sha ?? candidateName(c))}
							<tr
								class="dark:border-gray-850 text-xs {c.status === 'winner'
									? 'bg-green-500/10'
									: c.is_baseline
										? 'bg-blue-500/5'
										: ''}"
							>
								<td class="px-3 py-1 font-normal text-gray-900 dark:text-white">
									{candidateName(c)}
									{#if c.status === 'winner'}
										<span class="text-green-600 dark:text-green-400">★ {$i18n.t('winner')}</span>
									{:else if c.is_baseline}
										<span class="text-blue-600 dark:text-blue-400">({$i18n.t('baseline')})</span>
									{/if}
								</td>
								<td class="px-3 py-1 text-right">{fmtNum(c.score)}</td>
								<td class="px-3 py-1 text-right">{fmtNum(c.score_pct, 1)}</td>
								<td class="px-3 py-1 text-right">{c.paired_items ?? '—'}</td>
								<td class="px-3 py-1">
									{c.status ?? '—'}
									{#if c.status_reason}
										<span class="text-gray-400"> — {c.status_reason}</span>
									{/if}
								</td>
							</tr>
						{/each}
						{#if sortedCandidates.length === 0}
							<tr
								><td colspan="5" class="px-3 py-2 text-center text-gray-400"
									>{$i18n.t('No candidates yet')}</td
								></tr
							>
						{/if}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Pauses -->
		<div>
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
				{$i18n.t('Pauses')} ({(status.pauses ?? []).length})
			</div>
			{#if (status.pauses ?? []).length > 0}
				<div class="scrollbar-hidden relative whitespace-nowrap overflow-x-auto max-w-full">
					<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
						<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
							<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
								<th class="px-2.5 py-2">{$i18n.t('Trigger')}</th>
								<th class="px-2.5 py-2 text-right">{$i18n.t('Round')}</th>
								<th class="px-2.5 py-2 text-right">{$i18n.t('Temp before→after')}</th>
								<th class="px-2.5 py-2 text-right">{$i18n.t('Power before→after')}</th>
								<th class="px-2.5 py-2 text-right">{$i18n.t('Drift ratio')}</th>
								<th class="px-2.5 py-2">{$i18n.t('Resolution')}</th>
							</tr>
						</thead>
						<tbody>
							{#each status.pauses as p, idx (idx)}
								<tr class="dark:border-gray-850 text-xs">
									<td class="px-3 py-1">{p.trigger_kind ?? '—'}</td>
									<td class="px-3 py-1 text-right">{p.round ?? '—'}</td>
									<td class="px-3 py-1 text-right"
										>{fmtNum(p.temp_before, 1)}→{fmtNum(p.temp_after, 1)}</td
									>
									<td class="px-3 py-1 text-right"
										>{fmtNum(p.power_before, 1)}→{fmtNum(p.power_after, 1)}</td
									>
									<td class="px-3 py-1 text-right">{fmtNum(p.drift_ratio)}</td>
									<td class="px-3 py-1">{p.resolution ?? '—'}</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{:else}
				<div class="text-xs text-gray-400">{$i18n.t('None')}</div>
			{/if}
		</div>

		<!-- Not-measured / blocked -->
		<div class="grid md:grid-cols-2 gap-3">
			<div>
				<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
					{$i18n.t('Not measured')} ({(status.not_measured ?? []).length})
				</div>
				{#if (status.not_measured ?? []).length > 0}
					<ul class="text-xs text-gray-600 dark:text-gray-400 list-disc pl-4 space-y-0.5">
						{#each status.not_measured as nm}
							<li>{nm.candidate_sha}: {nm.reason}</li>
						{/each}
					</ul>
				{:else}
					<div class="text-xs text-gray-400">{$i18n.t('None')}</div>
				{/if}
			</div>

			<div>
				<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
					{$i18n.t('Blocked')} ({(status.blocked ?? []).length})
				</div>
				{#if (status.blocked ?? []).length > 0}
					<ul class="text-xs text-gray-600 dark:text-gray-400 list-disc pl-4 space-y-0.5">
						{#each status.blocked as b}
							<li>{b}</li>
						{/each}
					</ul>
				{:else}
					<div class="text-xs text-gray-400">{$i18n.t('None')}</div>
				{/if}
			</div>
		</div>

		<!-- Design audit -->
		{#if status.design_audit}
			<div>
				<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
					{$i18n.t('Design audit')}
				</div>
				<div
					class="prose dark:prose-invert max-w-none text-sm bg-gray-50 dark:bg-gray-850 rounded-lg p-3"
				>
					{@html DOMPurify.sanitize(marked.parse(status.design_audit) as string)}
				</div>
			</div>
		{/if}

		<!-- Adoption -->
		{#if status.adoption}
			<div>
				<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
					{$i18n.t('Adoption')}
				</div>
				<div class="rounded-lg bg-gray-50 dark:bg-gray-850 p-3 flex flex-col gap-2">
					<div class="grid sm:grid-cols-2 gap-3 text-sm">
						<div>
							<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Candidate SHA')}</div>
							<div class="font-medium text-gray-900 dark:text-white truncate">
								{status.adoption.candidate_sha ?? '—'}
							</div>
						</div>
						<div>
							<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Config ID')}</div>
							<div class="font-medium text-gray-900 dark:text-white truncate">
								{status.adoption.config_id ?? '—'}
							</div>
						</div>
					</div>
					{#if status.adoption.flags}
						<code
							class="text-xs bg-gray-100 dark:bg-gray-800 rounded px-1.5 py-0.5 w-fit whitespace-pre-wrap break-all"
							>{status.adoption.flags}</code
						>
					{/if}
					{#if status.adoption.reduces_context}
						<Badge type="warning" content={$i18n.t('Reduces context')} />
					{/if}
					{#if status.adoption.note}
						<div class="text-xs text-gray-600 dark:text-gray-400">{status.adoption.note}</div>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Guard -->
		{#if status.guard}
			<div>
				<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
					{$i18n.t('Guard')}
				</div>
				<div class="grid sm:grid-cols-2 md:grid-cols-4 gap-3 text-sm">
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Paired items')}</div>
						<div class="font-medium text-gray-900 dark:text-white">{status.guard.n ?? '—'}</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Wins')}</div>
						<div class="font-medium text-gray-900 dark:text-white">{status.guard.b ?? '—'}</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Losses')}</div>
						<div class="font-medium text-gray-900 dark:text-white">{status.guard.c ?? '—'}</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">
							{$i18n.t('McNemar p-value')}
						</div>
						<div class="font-medium text-gray-900 dark:text-white">{fmtNum(status.guard.p, 4)}</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">
							{$i18n.t('Discordant pairs')}
						</div>
						<div class="font-medium text-gray-900 dark:text-white">
							{status.guard.discordant ?? '—'}
						</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">
							{$i18n.t('PSI (prior shift index)')}
						</div>
						<div class="font-medium text-gray-900 dark:text-white">{fmtNum(status.guard.psi)}</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">
							{$i18n.t('Winner passed')}
						</div>
						<div class="font-medium text-gray-900 dark:text-white">
							{status.guard.winner_passed ?? '—'}
						</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">
							{$i18n.t('Incumbent passed')}
						</div>
						<div class="font-medium text-gray-900 dark:text-white">
							{status.guard.incumbent_passed ?? '—'}
						</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">
							{$i18n.t('Min. detectable effect')}
						</div>
						<div class="font-medium text-gray-900 dark:text-white">{fmtNum(status.guard.mde)}</div>
					</div>
					<div class="p-2.5 rounded-lg bg-gray-50 dark:bg-gray-850">
						<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Items needed')}</div>
						<div class="font-medium text-gray-900 dark:text-white">
							{status.guard.needed ?? '—'}
						</div>
					</div>
					<div
						class="p-2.5 rounded-lg {status.guard.regressed
							? 'bg-red-500/10'
							: 'bg-gray-50 dark:bg-gray-850'}"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Regressed?')}</div>
						<div class="font-medium text-gray-900 dark:text-white">
							{status.guard.regressed ? $i18n.t('Yes') : $i18n.t('No')}
						</div>
					</div>
				</div>
			</div>
		{/if}
	{/if}
</div>
