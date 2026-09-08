<script lang="ts">
	import { onDestroy, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import type { i18n as i18nType } from 'i18next';
	import {
		getServeProfiles,
		getServeProfile,
		startServe,
		stopServe,
		checkServe,
		streamServe,
		parseBenchmarksEventStream,
		type ServeStartForm
	} from '$lib/apis/benchmarks';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import NativeSelect from '$lib/components/common/NativeSelect.svelte';

	const i18n = getContext<Writable<i18nType>>('i18n');

	const inputClass =
		'w-full h-8 rounded-lg border border-gray-100/50 bg-gray-50/40 px-2.5 text-xs text-gray-700 outline-hidden transition-colors placeholder:text-gray-300 focus:border-blue-400 dark:border-white/[0.04] dark:bg-white/[0.03] dark:text-gray-300 dark:placeholder:text-gray-700 dark:focus:border-blue-500';

	const buttonClass =
		'px-3.5 py-1.5 text-sm font-normal bg-black hover:bg-gray-900 text-white dark:bg-white dark:text-black dark:hover:bg-gray-100 transition rounded-full disabled:opacity-40 disabled:cursor-not-allowed';

	const secondaryButtonClass =
		'px-3.5 py-1.5 text-sm font-normal bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-gray-100 transition rounded-full disabled:opacity-40 disabled:cursor-not-allowed';

	// -------------------------------------------------------------------
	// Profile picker
	// -------------------------------------------------------------------
	let profiles: string[] = [];
	let selectedProfile = '';
	let profileDetails: Record<string, unknown> | null = null;
	let profilesLoading = false;
	let profileDetailsLoading = false;

	const loadProfiles = async () => {
		profilesLoading = true;
		error = null;
		try {
			const res = await getServeProfiles(localStorage.token);
			profiles = res?.profiles ?? [];
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to load profiles');
		}
		profilesLoading = false;
	};

	const loadProfileDetails = async (name: string) => {
		if (!name) {
			profileDetails = null;
			return;
		}
		profileDetailsLoading = true;
		try {
			profileDetails = await getServeProfile(localStorage.token, name);
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to load profile');
			profileDetails = null;
		}
		profileDetailsLoading = false;
	};

	$: loadProfileDetails(selectedProfile);

	// -------------------------------------------------------------------
	// Overrides / start form
	// -------------------------------------------------------------------
	let ngl: number | null = null;
	let ctx: number | null = null;
	let threads: number | null = null;
	let parallel: number | null = null;
	let ot = '';
	let reasoning = '';
	let spec = '';

	$: reasoningOptions = [
		{ value: '', label: $i18n.t('Default') },
		{ value: 'low', label: 'low' },
		{ value: 'medium', label: 'medium' },
		{ value: 'high', label: 'high' }
	];

	$: specOptions = [
		{ value: '', label: $i18n.t('Default') },
		{ value: 'on', label: 'on' },
		{ value: 'off', label: 'off' }
	];

	// -------------------------------------------------------------------
	// Start / Stop / Check
	// -------------------------------------------------------------------
	let starting = false;
	let stopping = false;
	let checking = false;
	let error: string | null = null;
	let checkResult: { port?: number; model?: string; profile?: string } | null = null;

	const handleStart = async () => {
		starting = true;
		error = null;
		checkResult = null;
		try {
			const form: ServeStartForm = {};
			if (selectedProfile) form.profile = selectedProfile;
			if (ngl !== null && ngl !== undefined && `${ngl}` !== '') form.ngl = ngl;
			if (ctx !== null && ctx !== undefined && `${ctx}` !== '') form.ctx = ctx;
			if (threads !== null && threads !== undefined && `${threads}` !== '') form.threads = threads;
			if (parallel !== null && parallel !== undefined && `${parallel}` !== '')
				form.parallel = parallel;
			if (ot.trim()) form.ot = ot.trim();
			if (reasoning) form.reasoning = reasoning;
			if (spec) form.spec = spec;

			await startServe(localStorage.token, form);
			startLogStream();
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to start');
		}
		starting = false;
	};

	const handleStop = async () => {
		stopping = true;
		error = null;
		try {
			await stopServe(localStorage.token);
			stopLogStream();
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to stop');
		}
		stopping = false;
	};

	const handleCheck = async () => {
		checking = true;
		error = null;
		try {
			checkResult = await checkServe(localStorage.token);
		} catch (err: any) {
			error = err?.detail ?? err ?? $i18n.t('Failed to check');
			checkResult = null;
		}
		checking = false;
	};

	// -------------------------------------------------------------------
	// Log stream
	// -------------------------------------------------------------------
	let logLines: string[] = [];
	let streaming = false;
	let abortController: AbortController | null = null;
	let logContainer: HTMLDivElement | null = null;

	const scrollToBottom = async () => {
		await new Promise((r) => setTimeout(r, 0));
		if (logContainer) {
			logContainer.scrollTop = logContainer.scrollHeight;
		}
	};

	const startLogStream = async () => {
		stopLogStream();
		logLines = [];
		streaming = true;

		const [res, controller] = await streamServe(localStorage.token);
		abortController = controller;

		if (!res || !res.body) {
			streaming = false;
			return;
		}

		try {
			for await (const { event, data } of parseBenchmarksEventStream<{ line: string }>(res.body)) {
				if (event === 'done') {
					break;
				}
				if (data?.line !== undefined) {
					logLines = [...logLines, data.line];
					scrollToBottom();
				}
			}
		} catch (err: any) {
			// Aborted or connection closed - not necessarily an error worth surfacing.
			console.error(err);
		}
		streaming = false;
	};

	const stopLogStream = () => {
		if (abortController) {
			abortController.abort();
			abortController = null;
		}
		streaming = false;
	};

	onDestroy(() => {
		stopLogStream();
	});

	loadProfiles();
</script>

<div class="flex flex-col gap-4">
	<div class="flex items-center justify-between">
		<h2 class="text-sm font-medium text-gray-900 dark:text-white">{$i18n.t('Serve')}</h2>
	</div>

	{#if error}
		<div
			class="text-xs text-red-700 dark:text-red-200 bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2"
		>
			{typeof error === 'string' ? error : JSON.stringify(error)}
		</div>
	{/if}

	<div class="grid md:grid-cols-2 gap-4">
		<!-- Profile picker -->
		<div class="flex flex-col gap-2">
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300">
				{$i18n.t('Profile')}
			</div>

			{#if profilesLoading}
				<div class="flex justify-center py-4">
					<Spinner className="size-4" />
				</div>
			{:else}
				<NativeSelect
					bind:value={selectedProfile}
					options={profiles}
					placeholder={$i18n.t('Select a profile')}
					className="{inputClass} pr-8"
				/>
			{/if}

			{#if profileDetailsLoading}
				<div class="flex justify-center py-2">
					<Spinner className="size-4" />
				</div>
			{:else if profileDetails}
				<div
					class="rounded-lg border border-gray-100 dark:border-gray-850 p-2.5 text-xs text-gray-600 dark:text-gray-400 max-h-64 overflow-y-auto"
				>
					{#each Object.entries(profileDetails) as [key, value]}
						<div class="flex justify-between gap-2 py-0.5">
							<span class="text-gray-400 dark:text-gray-500">{key}</span>
							<span class="text-gray-700 dark:text-gray-300 text-right break-all"
								>{typeof value === 'object' ? JSON.stringify(value) : value}</span
							>
						</div>
					{/each}
				</div>
			{/if}
		</div>

		<!-- Overrides -->
		<div class="flex flex-col gap-2">
			<div class="text-xs font-normal text-gray-700 dark:text-gray-300">
				{$i18n.t('Overrides')}
			</div>
			<div class="grid grid-cols-2 gap-2">
				<div class="flex flex-col gap-1">
					<label class="text-[0.6875rem] text-gray-400" for="serve-ngl">ngl</label>
					<input id="serve-ngl" class={inputClass} type="number" bind:value={ngl} />
				</div>
				<div class="flex flex-col gap-1">
					<label class="text-[0.6875rem] text-gray-400" for="serve-ctx">ctx</label>
					<input id="serve-ctx" class={inputClass} type="number" bind:value={ctx} />
				</div>
				<div class="flex flex-col gap-1">
					<label class="text-[0.6875rem] text-gray-400" for="serve-threads">threads</label>
					<input id="serve-threads" class={inputClass} type="number" bind:value={threads} />
				</div>
				<div class="flex flex-col gap-1">
					<label class="text-[0.6875rem] text-gray-400" for="serve-parallel">parallel</label>
					<input id="serve-parallel" class={inputClass} type="number" bind:value={parallel} />
				</div>
				<div class="flex flex-col gap-1 col-span-2">
					<label class="text-[0.6875rem] text-gray-400" for="serve-ot">ot</label>
					<input id="serve-ot" class={inputClass} type="text" bind:value={ot} placeholder="--ot" />
				</div>
				<div class="flex flex-col gap-1">
					<label class="text-[0.6875rem] text-gray-400" for="serve-reasoning"
						>{$i18n.t('Reasoning effort')}</label
					>
					<NativeSelect
						bind:value={reasoning}
						options={reasoningOptions}
						className="{inputClass} pr-8"
					/>
				</div>
				<div class="flex flex-col gap-1">
					<label class="text-[0.6875rem] text-gray-400" for="serve-spec"
						>{$i18n.t('Speculative decoding')}</label
					>
					<NativeSelect bind:value={spec} options={specOptions} className="{inputClass} pr-8" />
				</div>
			</div>
		</div>
	</div>

	<!-- Actions -->
	<div class="flex items-center gap-2">
		<button class={buttonClass} on:click={handleStart} disabled={starting}>
			{#if starting}
				<Spinner className="size-3" />
			{:else}
				{$i18n.t('Start')}
			{/if}
		</button>
		<button class={secondaryButtonClass} on:click={handleStop} disabled={stopping}>
			{#if stopping}
				<Spinner className="size-3" />
			{:else}
				{$i18n.t('Stop')}
			{/if}
		</button>
		<button class={secondaryButtonClass} on:click={handleCheck} disabled={checking}>
			{#if checking}
				<Spinner className="size-3" />
			{:else}
				{$i18n.t('Check')}
			{/if}
		</button>

		{#if checkResult}
			<div class="text-xs text-gray-500 dark:text-gray-400 ml-2">
				{$i18n.t('Port')}: {checkResult.port ?? '-'} · {$i18n.t('Model')}: {checkResult.model ??
					'-'} · {$i18n.t('Profile')}: {checkResult.profile ?? '-'}
			</div>
		{/if}
	</div>

	<!-- Log panel -->
	<div class="flex flex-col gap-1">
		<div class="flex items-center gap-2 text-xs font-normal text-gray-700 dark:text-gray-300">
			{$i18n.t('Log')}
			{#if streaming}
				<Spinner className="size-3" />
			{/if}
		</div>
		<div
			bind:this={logContainer}
			class="bg-gray-950 text-gray-100 rounded-lg p-2.5 text-xs font-mono h-72 overflow-y-auto whitespace-pre-wrap break-all"
		>
			{#each logLines as line}
				<div>{line}</div>
			{/each}
			{#if logLines.length === 0}
				<div class="text-gray-500">{$i18n.t('No log output yet')}</div>
			{/if}
		</div>
	</div>
</div>
