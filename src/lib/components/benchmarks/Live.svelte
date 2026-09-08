<script lang="ts">
	import { onMount, onDestroy, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import { getLive } from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	const POLL_MS = 5000;

	let loading = true;
	let error: string | null = null;

	let run: Record<string, unknown> | null = null;
	let summary: Record<string, number | string> = {};
	let deltas: Record<string, number> = {};
	let requests = 0;
	let recentSamples: Array<{
		sample_id?: string | number;
		run_id?: string | number;
		at: number;
		temp_c?: number;
		util_pct?: number;
		mem_used_mib?: number;
		mem_total_mib?: number;
		power_w?: number;
		sm_mhz?: number;
		throttle?: boolean | string;
	}> = [];
	let warning: string | null = null;

	let intervalId: ReturnType<typeof setInterval> | null = null;

	const formatValue = (value: unknown): string => {
		if (value === null || value === undefined) return '-';
		if (typeof value === 'number') {
			return Number.isInteger(value) ? value.toString() : value.toFixed(2);
		}
		return String(value);
	};

	const formatTime = (at: number): string => {
		if (at === null || at === undefined) return '-';
		try {
			return new Date(at * 1000).toLocaleTimeString();
		} catch (e) {
			return String(at);
		}
	};

	const poll = async () => {
		try {
			const res = await getLive(localStorage.token);
			run = res?.run ?? null;
			summary = res?.summary ?? {};
			deltas = res?.deltas ?? {};
			requests = res?.requests ?? 0;
			recentSamples = res?.recent_samples ?? [];
			warning = res?.warning ?? null;
			error = null;
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to load live status');
		}
		loading = false;
	};

	onMount(() => {
		poll();
		intervalId = setInterval(poll, POLL_MS);
	});

	onDestroy(() => {
		if (intervalId) {
			clearInterval(intervalId);
			intervalId = null;
		}
	});
</script>

<div class="flex flex-col gap-4">
	<div class="flex items-center justify-between">
		<h2 class="text-sm font-medium text-gray-900 dark:text-white">{$i18n.t('Live')}</h2>
	</div>

	{#if error}
		<div
			class="text-xs text-red-700 dark:text-red-200 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
		>
			{typeof error === 'string' ? error : JSON.stringify(error)}
		</div>
	{/if}

	{#if loading}
		<div class="my-10 flex justify-center">
			<Spinner className="size-5" />
		</div>
	{:else if !run}
		<div
			class="text-sm text-gray-500 dark:text-gray-400 border border-gray-100 dark:border-gray-850 rounded-lg py-10 text-center"
		>
			{$i18n.t('Nothing is currently being recorded')}
		</div>
	{:else}
		{#if warning}
			<div
				class="text-xs text-yellow-800 dark:text-yellow-200 bg-yellow-500/10 border border-yellow-500/30 rounded-lg px-3 py-2 font-medium"
			>
				⚠ {warning}
			</div>
		{/if}

		<!-- Run identity -->
		<div
			class="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-gray-600 dark:text-gray-400 border border-gray-100 dark:border-gray-850 rounded-lg px-3 py-2"
		>
			<span
				><span class="text-gray-400 dark:text-gray-500">{$i18n.t('Model')}:</span>
				<span class="text-gray-900 dark:text-white font-normal">{run.model ?? '-'}</span></span
			>
			<span
				><span class="text-gray-400 dark:text-gray-500">{$i18n.t('Config')}:</span>
				<span class="text-gray-900 dark:text-white font-normal">{run.config_id ?? '-'}</span></span
			>
			<span
				><span class="text-gray-400 dark:text-gray-500">{$i18n.t('Port')}:</span>
				<span class="text-gray-900 dark:text-white font-normal">{run.port ?? '-'}</span></span
			>
			<span
				><span class="text-gray-400 dark:text-gray-500">{$i18n.t('Requests')}:</span>
				<span class="text-gray-900 dark:text-white font-normal">{requests}</span></span
			>
		</div>

		<!-- Summary stat grid -->
		<div>
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
				{$i18n.t('Summary')}
			</div>
			<div
				class="grid gap-2"
				style="grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));"
			>
				{#each Object.entries(summary) as [key, value]}
					<div
						class="flex flex-col gap-0.5 rounded-lg border border-gray-100 dark:border-gray-850 px-2.5 py-2"
					>
						<span class="text-[0.6875rem] text-gray-400 dark:text-gray-500 truncate">{key}</span>
						<span class="text-sm font-normal text-gray-900 dark:text-white">{formatValue(value)}</span
						>
					</div>
				{/each}
				{#if Object.keys(summary).length === 0}
					<div class="text-xs text-gray-400">{$i18n.t('No data')}</div>
				{/if}
			</div>
		</div>

		<!-- Deltas -->
		<div>
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
				{$i18n.t('Deltas')}
			</div>
			<div class="scrollbar-hidden relative whitespace-nowrap overflow-x-auto max-w-full">
				<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
					<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
						<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
							<th scope="col" class="px-2.5 py-2">{$i18n.t('Counter')}</th>
							<th scope="col" class="px-2.5 py-2 text-right">{$i18n.t('Value')}</th>
						</tr>
					</thead>
					<tbody>
						{#each Object.entries(deltas) as [name, value]}
							<tr class="dark:border-gray-850 text-xs">
								<td class="px-3 py-1 text-gray-900 dark:text-white">{name}</td>
								<td class="px-3 py-1 text-right">{formatValue(value)}</td>
							</tr>
						{/each}
						{#if Object.keys(deltas).length === 0}
							<tr
								><td colspan="2" class="px-3 py-2 text-center text-gray-400"
									>{$i18n.t('No data')}</td
								></tr
							>
						{/if}
					</tbody>
				</table>
			</div>
		</div>

		<!-- Recent samples -->
		<div>
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300 mb-1 px-0.5">
				{$i18n.t('Recent Samples')}
			</div>
			<div class="scrollbar-hidden relative whitespace-nowrap overflow-x-auto max-w-full">
				<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
					<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
						<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
							<th scope="col" class="px-2.5 py-2">{$i18n.t('Time')}</th>
							<th scope="col" class="px-2.5 py-2 text-right">{$i18n.t('Util %')}</th>
							<th scope="col" class="px-2.5 py-2 text-right">{$i18n.t('Mem')}</th>
							<th scope="col" class="px-2.5 py-2 text-right">{$i18n.t('Power (W)')}</th>
							<th scope="col" class="px-2.5 py-2 text-right">{$i18n.t('SM MHz')}</th>
							<th scope="col" class="px-2.5 py-2 text-right">{$i18n.t('Temp (°C)')}</th>
						</tr>
					</thead>
					<tbody>
						{#each recentSamples as sample (sample.sample_id ?? sample.at)}
							<tr class="dark:border-gray-850 text-xs">
								<td class="px-3 py-1">{formatTime(sample.at)}</td>
								<td class="px-3 py-1 text-right">{formatValue(sample.util_pct)}</td>
								<td class="px-3 py-1 text-right"
									>{formatValue(sample.mem_used_mib)} / {formatValue(sample.mem_total_mib)}</td
								>
								<td class="px-3 py-1 text-right">{formatValue(sample.power_w)}</td>
								<td class="px-3 py-1 text-right">{formatValue(sample.sm_mhz)}</td>
								<td class="px-3 py-1 text-right">{formatValue(sample.temp_c)}</td>
							</tr>
						{/each}
						{#if recentSamples.length === 0}
							<tr
								><td colspan="6" class="px-3 py-2 text-center text-gray-400"
									>{$i18n.t('No data')}</td
								></tr
							>
						{/if}
					</tbody>
				</table>
			</div>
		</div>
	{/if}
</div>
