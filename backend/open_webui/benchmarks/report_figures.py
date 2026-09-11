"""benchmarks/report_figures.py - the matplotlib figures for report.py.

Ported from the `Figures` class and its supporting helpers in local-llm's
scripts/llama_report.py (outer repo). Split into its own module purely for
size -- report.py already carries the design audit, the paired tests, the
power analysis and the throughput/throttle sections, and adding four
matplotlib figure builders (plus their unicode fallbacks) to the same file
would make it unwieldy to review. The statistics stay in report.py; nothing
here computes a p-value, it only draws one that report.py already computed.

matplotlib is soft here, same as the original: every figure degrades to a
`unicode_bars()` text plot in a fenced markdown block when matplotlib is not
installed, so no analysis in report.py is ever silently hidden behind a
missing wheel. Checked once at import time (`HAVE_MPL`), not per figure.

Import direction: this module imports labelling and statistics helpers from
report.py (Block, Regimes, _iso, graded, timing, mcnemar_power,
detectable_effect, short_level, wrap_title, Figure). report.py imports this
module's `Figures` class back, but only lazily, inside `build_report()` --
never at module scope -- specifically so this file can import from report.py
at its own top level without the two forming a load-time cycle. See
report.py's module docstring for the fuller explanation.

Output change from the original: `Figures._save()` used to write a PNG
straight to a `Path` under the CLI's own `--out` directory
(`logs/report/<date>/`). This module has no filesystem to write to -- and
per the migration's brief, must not invent one -- so `_save()` renders to an
in-memory buffer and appends a `Figure(name, data, alt)` to `self.figures`
instead; report.py's ReportResult hands that list back whole, and it is the
router's job to decide where (or whether) each one gets written to disk.
"""

from __future__ import annotations

import io

from open_webui.benchmarks import stats as lstats
from open_webui.benchmarks.report import (
    Block,
    Figure,
    NONE_LEVEL,
    Regimes,
    _iso,
    detectable_effect,
    graded,
    mcnemar_power,
    short_level,
    timing,
    wrap_title,
)

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    HAVE_MPL = True
except Exception:  # pragma: no cover - exercised by running with it blocked
    HAVE_MPL = False


# Figure palette. Two hues only, and deliberately not green/red: pass-versus-
# fail in green and red is the one pairing a deuteranope cannot separate
# (measured dE 4.1, against a floor of 8, per the original). Blue carries
# every primary mark, orange marks the thing being flagged, and both are
# labelled in text as well, so no reading of a figure depends on seeing
# colour at all.
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
SURFACE = '#fcfcfb'
SERIES = '#2a78d6'  # primary mark
FLAG = '#eb6834'  # flagged: throttled, discordant, refused
FILL_OFF = '#e6e5df'  # the empty half of a binary cell
FILL_ON = '#cfd8e3'  # a filled cell in a row that carries no information

BLOCKS = ' ▏▎▍▌▋▊▉█'


def unicode_bars(labels: list[str], values: list[float | None], width: int = 36, unit: str = '') -> list[str]:
    """A bar chart in text, for when matplotlib is not installed.

    Deliberately a bar chart and not an attempt at a scatter or a line:
    eighth blocks give a bar about three times the resolution of a terminal
    cell, and a bar is the one form that survives being drawn in a
    monospace grid without lying about where a point sits.
    """
    real = [v for v in values if v is not None]
    top = max(real) if real else 0.0
    pad = max((len(l) for l in labels), default=0)
    lines = []
    for label, value in zip(labels, values):
        if value is None or top <= 0:
            lines.append(f'{label:<{pad}}  {"":<{width}}  -')
            continue
        eighths = int(round(value / top * width * 8))
        bar = '█' * (eighths // 8)
        if eighths % 8:
            bar += BLOCKS[eighths % 8]
        lines.append(f'{label:<{pad}}  {bar:<{width}}  {value:g}{unit}')
    return lines


def fenced(title: str, lines: list[str]) -> list[str]:
    return ['```text', title, ''] + lines + ['```', '']


class Figures:
    """PNG bytes when matplotlib is here, unicode in a fenced block when it is not.

    The fallback is not a courtesy. This report has to build inside a
    long-lived FastAPI process on a box where the matplotlib wheel failed
    to install, because the alternative -- a 500 where a document should be
    -- loses the analysis as well as the picture.
    """

    def __init__(self, enabled: bool = True):
        self.png = bool(enabled and HAVE_MPL)
        self.figures: list[Figure] = []
        if not enabled:
            self.reason = 'figures disabled by the caller'
        elif not HAVE_MPL:
            self.reason = 'matplotlib is not installed'
        else:
            self.reason = ''

    @property
    def written(self) -> list[str]:
        """Names of the figures actually rendered, for a log line -- kept for parity."""
        return [f.name for f in self.figures]

    # -- plumbing ---------------------------------------------------------
    MIN_WIDTH_IN = 7.5

    def _new(self, name: str, size=(8.0, 4.0)):
        # A floor on the width, because these figures size themselves to
        # their level count and the title does not shrink with them.
        size = (max(size[0], self.MIN_WIDTH_IN), size[1])
        fig, ax = plt.subplots(figsize=size, dpi=140)
        fig.patch.set_facecolor(SURFACE)
        ax.set_facecolor(SURFACE)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
            ax.spines[side].set_linewidth(1.0)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        return fig, ax

    def _save(self, fig, ax, name: str, title: str, out: list[str], alt: str):
        ax.set_title(wrap_title(title, fig.get_size_inches()[0]), color=INK, fontsize=11, loc='left', pad=12)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=SURFACE)
        plt.close(fig)
        self.figures.append(Figure(name=name, data=buf.getvalue(), alt=alt))
        out += [f'![{alt}]({name})', '', f'*{title}*', '']

    def _skip(self, out: list[str], title: str, lines: list[str]):
        out += fenced(f'{title}   [{self.reason}]', lines)

    # -- figure 1: the timeline -------------------------------------------
    def timeline(self, records: list[dict], regimes: Regimes, out: list[str]):
        by_run: dict[int, list[dict]] = {}
        for r in graded(records):
            if r.get('run_id') is not None and timing(r, 'predicted_per_second'):
                by_run.setdefault(r['run_id'], []).append(r)
        if not by_run:
            return
        run_id = max(by_run, key=lambda k: len(by_run[k]))
        series = sorted(by_run[run_id], key=lambda r: (r.get('at') or 0))
        values = [timing(r, 'predicted_per_second') for r in series]
        marks = [regimes.label(r) for r in series]
        title = f'Generation throughput over run {run_id}, requests in order (n = {len(series)})'

        if not self.png:
            labels = [f'{i:>3} {m[:18]:<18}' for i, m in enumerate(marks)]
            self._skip(out, title, unicode_bars(labels, values, unit=' t/s'))
            return

        fig, ax = self._new(f'fig1-timeline-run{run_id}.png', size=(9.5, 4.4))
        x = list(range(len(series)))
        # Shade the contiguous spans served under a throttle, before the
        # line, so the line stays the topmost mark. One legend entry rather
        # than one annotation per span: the spans are adjacent and every
        # label drawn at the same y overprints into an unreadable smear.
        shaded = False
        start_i = None
        for i, m in enumerate(marks + ['none']):
            throttled = m not in ('none', 'no telemetry') and 'GpuIdle' not in m
            if throttled and start_i is None:
                start_i = i
            elif not throttled and start_i is not None:
                ax.axvspan(start_i - 0.5, i - 0.5, color=FLAG, alpha=0.12, lw=0, label='served under a throttle' if not shaded else None)
                shaded = True
                start_i = None

        # The level boundaries are the confound this figure exists to show:
        # levels that ran one after another have anything that drifted with
        # time read as the level as far as the data can tell.
        prompts = [(r.get('system_name') or NONE_LEVEL) for r in series]
        for i in range(1, len(prompts)):
            if prompts[i] != prompts[i - 1]:
                ax.axvline(i - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)), zorder=1)

        ax.plot(
            x,
            values,
            color=SERIES,
            linewidth=2.0,
            marker='o',
            markersize=4.5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            zorder=3,
            label='generation t/s, one request',
        )
        ax.set_xlabel('request, in the order served  (dashed rules: system prompt changes)', color=INK_2, fontsize=9)
        ax.set_ylabel('generation t/s', color=INK_2, fontsize=9)
        ax.set_ylim(bottom=0, top=max(v for v in values if v) * 1.28)

        # The cliff, named with the moment it happened and the medians
        # either side of it. Section 6's refusals rest on this event, and a
        # reader who has to infer it from the shape of the line has to take
        # the prose on trust.
        cut = self._cliff(values)
        if cut is not None:
            before = lstats.percentile(values[:cut], 0.5)
            after = lstats.percentile(values[cut:], 0.5)
            at = _iso(series[cut].get('at'))[:19]
            ax.axvline(cut - 0.5, color=FLAG, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(
                f'{at}Z: median {before:.1f} -> {after:.1f} t/s,\nand never recovers',
                xy=(cut - 0.5, max(v for v in values if v)),
                xytext=(8, -4),
                textcoords='offset points',
                color=FLAG,
                fontsize=8.5,
                ha='left',
                va='top',
            )
        ax.legend(loc='upper right', frameon=False, fontsize=8, labelcolor=INK_2)
        self._save(
            fig,
            ax,
            f'fig1-timeline-run{run_id}.png',
            title,
            out,
            'generation throughput per request, with the throttle cliff and the system-prompt boundaries marked',
        )

    @staticmethod
    def _cliff(values: list[float]) -> int | None:
        """The index of the largest single step down in the series.

        Reported only when the step is a fall of more than half, which is
        the difference between a throttle event and the ordinary spread
        between two requests. Nothing downstream depends on this; it
        decides where to put one annotation.
        """
        best, at = 0.0, None
        for i in range(1, len(values)):
            prev, cur = values[i - 1], values[i]
            if not prev or not cur:
                continue
            drop = (prev - cur) / prev
            if drop > best:
                best, at = drop, i
        return at if best > 0.5 else None

    # -- figure 2: the paired matrix --------------------------------------
    def paired_matrix(self, b: Block, items, levels, matrix, discordant, out):
        title = f'Outcome by item and system prompt -- {b.benchmark} x {b.adapter}'
        if not discordant:
            # A grid of identical cells is a picture of nothing. The
            # sentence says what the figure would have said, in less space
            # and without inviting a reader to hunt the grid for the
            # difference.
            out += [
                f'*No figure: every one of the {len(items)} items returns '
                f'the same verdict under all {len(levels)} levels, so the '
                f'matrix has no variation to draw.*',
                '',
            ]
            return
        if not self.png:
            lines = ['item'.ljust(max(len(i) for i in items) + 2) + '  '.join(lv[:10].ljust(10) for lv in levels)]
            for i, item in enumerate(items):
                cells = '  '.join(('pass' if v else 'FAIL').ljust(10) for v in matrix[i])
                flag = '  <- varies' if i in discordant else ''
                lines.append(item.ljust(max(len(x) for x in items) + 2) + cells + flag)
            self._skip(out, title, lines)
            return
        name = f'fig2-matrix-{b.slug()}.png'
        fig, ax = self._new(name, size=(1.6 * len(levels) + 3.0, 0.42 * len(items) + 2.2))
        ax.grid(False)
        labels = [short_level(lv, levels) for lv in levels]
        for i, row in enumerate(matrix):
            varies = i in discordant
            for j, value in enumerate(row):
                # Emphasis follows information, not polarity. A constant
                # row says nothing, and the varying rows are the whole
                # effective sample, so the constant passes recede to a
                # quiet fill and the failures inside a varying row carry
                # the flag colour. Still labelled in words: nothing here
                # depends on seeing the difference between two hues.
                if value:
                    face, ink = (FILL_ON, INK_2) if not varies else (SERIES, SURFACE)
                else:
                    face, ink = (FILL_OFF, MUTED) if not varies else (FLAG, SURFACE)
                ax.add_patch(plt.Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, facecolor=face, linewidth=0))
                ax.text(j + 0.5, i + 0.5, 'pass' if value else 'fail', ha='center', va='center', fontsize=8, color=ink)
        for i in discordant:
            ax.add_patch(plt.Rectangle((-0.06, i + 0.03), 0.06, 0.94, facecolor=FLAG, linewidth=0))
        ax.set_xlim(-0.1, len(levels))
        ax.set_ylim(len(items), 0)
        ax.set_xticks([j + 0.5 for j in range(len(levels))])
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_yticks([i + 0.5 for i in range(len(items))])
        ax.set_yticklabels([f'{it}  {"*" if i in discordant else " "}' for i, it in enumerate(items)], fontsize=8)
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
        self._save(
            fig,
            ax,
            name,
            title
            + f' ({len(discordant)} of {len(items)} items vary; only those rows are drawn in full colour and ruled in orange, '
            'the rest are constant and carry no information)',
            out,
            'pass/fail per item under each system prompt, with the varying items emphasised',
        )

    # -- figure 3: discordance against the baseline -----------------------
    def level_discordance(self, b: Block, levels, matrix, bi, out):
        """What each level actually changed, item by item, against the baseline.

        This replaces a bar chart of marginal pass rates. That chart drew
        the comparison section 4 spends its effort refusing: bars are the
        picture of a between-groups test, and several of them differing by
        one item read as a ranking, when the columns are measurements of
        the *same* items and only the within-item changes carry any
        information. The marginal rates keep their table; the figure shows
        the pairing.
        """
        base = levels[bi]
        others = [j for j in range(len(levels)) if j != bi]
        lost, gained = {}, {}
        for j in others:
            lost[j] = [i for i, row in enumerate(matrix) if row[bi] == 1 and row[j] == 0]
            gained[j] = [i for i, row in enumerate(matrix) if row[bi] == 0 and row[j] == 1]
        n = len(matrix)
        if not any(lost[j] or gained[j] for j in others):
            # Same reasoning as the matrix above: an empty plot is a worse
            # way to say "nothing changed" than the sentence is.
            out += [
                f'*No figure: no item changes verdict between `{base}` and '
                f'any of the {len(others)} other level(s), so every McNemar '
                f'table here is empty and there is nothing to plot.*',
                '',
            ]
            return
        title = f'Items changed against baseline `{base}` -- {b.benchmark} x {b.adapter} (n = {n} per level)'
        if not self.png:
            rows = [f'{"level":<32} {"lost":>5} {"gained":>7} {"n disc":>7}']
            for j in others:
                rows.append(f'{levels[j][:32]:<32} {len(lost[j]):>5} {len(gained[j]):>7} {len(lost[j]) + len(gained[j]):>7}')
            rows += [
                '',
                'lost   = baseline passed, this level failed  (McNemar b)',
                'gained = baseline failed, this level passed  (McNemar c)',
                'n disc = what the exact p-value rests on; the concordant',
                '         items are not in the denominator.',
            ]
            self._skip(out, title, rows)
            return
        name = f'fig3-discordance-{b.slug()}.png'
        fig, ax = self._new(name, size=(8.0, 0.46 * len(others) + 1.9))
        ax.grid(True, axis='x', color=GRID, linewidth=0.8)
        ax.axvline(0, color=INK_2, linewidth=1.2, zorder=2)
        for row, j in enumerate(others):
            y = len(others) - 1 - row
            ndisc = len(lost[j]) + len(gained[j])
            if ndisc == 0:
                ax.annotate(
                    'no item changed verdict', xy=(0, y), xytext=(8, 0), textcoords='offset points', color=MUTED, fontsize=8.5, va='center'
                )
                continue
            for k, i in enumerate(lost[j]):
                ax.plot([-(k + 1)], [y], marker='o', markersize=9, color=FLAG, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            for k, i in enumerate(gained[j]):
                ax.plot([k + 1], [y], marker='o', markersize=9, color=SERIES, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            ax.annotate(
                f'n disc {ndisc}',
                xy=(max(len(gained[j]), 1), y),
                xytext=(14, 0),
                textcoords='offset points',
                color=INK_2,
                fontsize=8.5,
                va='center',
            )
        span = max(1, max((len(lost[j]) for j in others), default=1), max((len(gained[j]) for j in others), default=1))
        ax.set_xlim(-span - 0.8, span + 2.2)
        ax.set_ylim(-0.55, len(others) - 0.45)
        ax.set_yticks(list(range(len(others))))
        ax.set_yticklabels([short_level(levels[j], levels) for j in reversed(others)], fontsize=9)
        ticks = list(range(-span, span + 1))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(abs(t)) for t in ticks])
        ax.set_xlabel('items lost (left, orange) and gained (right, blue) against the baseline', color=INK_2, fontsize=9)
        for side in ('left',):
            ax.spines[side].set_visible(False)
        self._save(fig, ax, name, title, out, "items each system prompt lost and gained against the baseline, the pairing the McNemar test rests on")

    # -- figure 4: the MDE curve ------------------------------------------
    def mde_curve(self, psi: float, ci: tuple[float, float], now: int, out: list[str]):
        ns = [n for n in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512)]
        mde = [detectable_effect(n, psi) for n in ns]
        title = f'Smallest detectable difference at 80% power, alpha 0.05, discordance psi = {psi:.3f}'
        if not self.png:
            self._skip(out, title, unicode_bars([f'n = {n:>4}' for n in ns], [None if m is None else 100 * m for m in mde], unit=' pp'))
            return
        name = 'fig4-mde.png'
        fig, ax = self._new(name, size=(8.5, 4.4))
        xs = [n for n, m in zip(ns, mde) if m is not None]
        ys = [100 * m for m in mde if m is not None]
        ax.plot(xs, ys, color=SERIES, linewidth=2.0, marker='o', markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.set_xscale('log')
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        # A log axis keeps its own minor ticks and formats them, so they
        # were being drawn straight through the explicit labels above.
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel('items per level', color=INK_2, fontsize=9)
        ax.set_ylabel('detectable difference, percentage points', color=INK_2, fontsize=9)

        # Headroom above the ceiling, so the annotations below have
        # somewhere to sit that is not the title.
        top = 100 * psi * 1.35
        ax.set_ylim(0, top)

        # The ceiling. A paired difference cannot exceed the discordance
        # rate, since every unit of it has to come from an item that
        # changed, so this line is where section 5's `impossible` comes
        # from.
        ax.axhline(100 * psi, color=MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=1)
        ax.annotate(
            f'psi = {100 * psi:.1f} pp: no larger effect can exist',
            xy=(xs[-1] if xs else 1, 100 * psi),
            xytext=(0, 5),
            textcoords='offset points',
            color=MUTED,
            fontsize=8.5,
            ha='right',
        )

        # The marker is drawn whether or not an MDE exists at this n. It
        # used to be guarded on detectable_effect() being non-None, which
        # is exactly None when nothing is reachable at 80% power -- so the
        # figure dropped its own subject in the one case this section
        # exists to report.
        here = detectable_effect(now, psi)
        ax.axvline(now, color=FLAG, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
        if here is not None:
            note = f'this experiment:\n{now} items, {100 * here:.0f} pp'
        else:
            ceiling = mcnemar_power(now, psi, psi * 0.999)
            note = f'this experiment: {now} items.\nNo effect is reachable at\n80% power -- even {100 * psi:.1f} pp is found only {100 * ceiling:.0f}% of the time.'
        ax.annotate(note, xy=(now, top * 0.30), xytext=(10, 0), textcoords='offset points', color=FLAG, fontsize=9, ha='left', va='center')
        if xs and min(xs) > min(ns):
            # The curve starts where it starts for a reason; shade the dead
            # region and say why, rather than leaving a reader to wonder
            # how an axis of items per level begins where it does.
            ax.axvspan(min(ns), min(xs), color=MUTED, alpha=0.07, lw=0, zorder=0)
            ax.annotate(
                f'under {min(xs)} items nothing is\ndetectable at 80% power',
                xy=(min(ns), top * 0.06),
                xytext=(6, 0),
                textcoords='offset points',
                color=MUTED,
                fontsize=8.5,
                ha='left',
                va='bottom',
            )
        self._save(
            fig, ax, name, title, out, 'minimum detectable effect against items per level, with the psi ceiling and this experiment\'s position marked'
        )
