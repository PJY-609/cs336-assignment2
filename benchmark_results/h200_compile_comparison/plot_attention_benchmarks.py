"""Generate standalone plots from attention benchmark rows (no GPU work)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_plots(directory, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import numpy as np

    directory = Path(directory)
    plot_dir = directory / 'plots'
    plot_dir.mkdir(exist_ok=True)
    colors = {'naive': '#d95f02', 'flash_pytorch': '#7570b3', 'flash_triton': '#1b9e77', 'flash_pytorch_compiled': '#0072b2'}
    labels = {'naive': 'Dense PyTorch (compiled)', 'flash_pytorch': 'Tiled PyTorch', 'flash_triton': 'Triton FA2', 'flash_pytorch_compiled': 'Tiled PyTorch (compiled)'}

    def save(fig, name):
        for extension in ('png', 'pdf'):
            temporary = plot_dir / f'{name}.tmp.{extension}'
            fig.savefig(temporary, dpi=160, bbox_inches='tight')
            temporary.replace(plot_dir / f'{name}.{extension}')
        plt.close(fig)

    main_rows = [row for row in rows if row.get('experiment') == 'main']
    if main_rows:
        dims = sorted({row['embedding_dim'] for row in main_rows})
        dtypes = sorted({row['dtype'] for row in main_rows})
        all_n = sorted({row['sequence_length'] for row in main_rows})
        metrics = [('forward_ms', 'Forward latency', 'Latency (ms)', 'latency_forward'),
                   ('backward_ms', 'Backward latency', 'Latency (ms)', 'latency_backward'),
                   ('forward_backward_ms', 'Forward + backward latency', 'Latency (ms)', 'latency_forward_backward'),
                   ('forward_peak_gb', 'Forward peak memory', 'Allocated memory (GB)', 'memory_forward'),
                   ('forward_backward_peak_gb', 'Forward + backward peak memory', 'Allocated memory (GB)', 'memory_forward_backward')]
        present = [impl for impl in colors if any(r['implementation'] == impl for r in main_rows)]
        for field, title, unit, filename in metrics:
            fig, axes = plt.subplots(len(dtypes), len(dims), squeeze=False, sharey=True,
                                     figsize=(4.2 * len(dims), 3.6 * len(dtypes)))
            for i, dtype in enumerate(dtypes):
                for j, d in enumerate(dims):
                    ax = axes[i, j]
                    for index, impl in enumerate(present):
                        subset = {r['sequence_length']: r for r in main_rows
                                  if r['dtype'] == dtype and r['embedding_dim'] == d
                                  and r['implementation'] == impl}
                        ax.plot(all_n, [subset.get(n, {}).get(field, np.nan) for n in all_n],
                                marker=['o', 's', '^', 'D'][index], markersize=3,
                                color=colors[impl], label=labels[impl])
                        for n, row in subset.items():
                            if row.get('status') in ('oom', 'resource_limit', 'error', 'timeout'):
                                tag = {'oom': 'OOM', 'resource_limit': 'LIMIT', 'error': 'ERR', 'timeout': 'TIME'}[row['status']]
                                if field in row:
                                    tag += '*'
                                ax.text(n, .98 - index * .10, tag, color=colors[impl], fontsize=6,
                                        ha='center', va='top', transform=ax.get_xaxis_transform())
                    ax.set_xscale('log', base=2)
                    ax.set_xticks(all_n, [f'{n // 1024}K' if n >= 1024 else str(n) for n in all_n], rotation=45)
                    ax.set_yscale('log')
                    ax.set_title(f'{dtype}, d={d}')
                    ax.set_xlabel('Sequence length (tokens)')
                    if j == 0:
                        ax.set_ylabel(unit)
                    ax.grid(alpha=0.25)
            values = [r[field] for r in main_rows if r.get(field, 0) > 0 and np.isfinite(r[field])]
            if values:
                axes[0, 0].set_ylim(min(values) / 1.25, max(values) * 1.25)
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center', ncol=2 if len(present) > 3 else 3, fontsize=9)
            fig.suptitle(f'{title}: H200, batch=1, causal — logarithmic y-axis')
            fig.tight_layout(rect=(0, 0.09, 1, 0.94))
            save(fig, filename)

    # Square-tile PyTorch comparison is separate from the rectangular Triton grid.
    pytorch_tiles = [r for r in rows if r.get('experiment') == 'tile_sensitivity'
                     and r['implementation'] in ('flash_pytorch', 'flash_pytorch_compiled')]
    if pytorch_tiles:
        lengths = sorted({r['sequence_length'] for r in pytorch_tiles})
        dtypes = sorted({r['dtype'] for r in pytorch_tiles})
        sizes = sorted({r['q_tile'] for r in pytorch_tiles})
        for phase, title in (('forward', 'Forward'), ('backward', 'Backward'), ('forward_backward', 'Forward + backward')):
            field = phase + '_ms'
            fig, axes = plt.subplots(len(dtypes), len(lengths), squeeze=False, sharey=True,
                                     figsize=(4.2 * len(lengths), 3.8 * len(dtypes)))
            for i, dtype in enumerate(dtypes):
                for j, n in enumerate(lengths):
                    ax = axes[i, j]
                    for index, impl in enumerate(('flash_pytorch', 'flash_pytorch_compiled')):
                        subset = {r['q_tile']: r for r in pytorch_tiles
                                  if r['dtype'] == dtype and r['sequence_length'] == n and r['implementation'] == impl}
                        ax.plot(sizes, [subset.get(t, {}).get(field, np.nan) for t in sizes],
                                marker=['s', 'D'][index], color=colors[impl], label=labels[impl])
                        for t, row in subset.items():
                            if row.get('status') in ('oom', 'resource_limit', 'error', 'timeout'):
                                tag = {'oom': 'OOM', 'resource_limit': 'LIMIT', 'error': 'ERR', 'timeout': 'TIME'}[row['status']]
                                ax.text(t, .98 - index * .1, tag + ('*' if field in row else ''),
                                        color=colors[impl], ha='center', va='top', fontsize=7,
                                        transform=ax.get_xaxis_transform())
                    ax.set_xscale('log', base=2)
                    ax.set_xticks(sizes, sizes)
                    ax.set_yscale('log')
                    ax.set_xlabel('Square tile size (tokens)')
                    ax.set_ylabel('Latency (ms)')
                    ax.set_title(f'{dtype}, N={n:,}, d=64')
                    ax.grid(alpha=.25)
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center', ncol=2)
            fig.suptitle(f'PyTorch {title.lower()}: fixed tile sensitivity; H200, batch=1, causal')
            fig.tight_layout(rect=(0, .07, 1, .94))
            save(fig, 'tile_sensitivity_pytorch_' + phase)

    tiles = [row for row in rows if row.get('experiment') == 'tile_sensitivity' and row['implementation'] == 'flash_triton']
    if tiles:
        lengths = sorted({row['sequence_length'] for row in tiles})
        dtypes = sorted({row['dtype'] for row in tiles})
        sizes = sorted({row[key] for row in tiles for key in ('q_tile', 'k_tile')})
        for phase, title in (('forward', 'Forward'), ('backward', 'Backward'), ('forward_backward', 'Forward + backward')):
            field = phase + '_ms'
            fig, axes = plt.subplots(len(dtypes), len(lengths), squeeze=False,
                                     figsize=(4.2 * len(lengths), 3.8 * len(dtypes)))
            for i, dtype in enumerate(dtypes):
                for j, n in enumerate(lengths):
                    ax = axes[i, j]
                    subset = {(row['q_tile'], row['k_tile']): row for row in tiles
                              if row['dtype'] == dtype and row['sequence_length'] == n}
                    values = np.full((len(sizes), len(sizes)), np.nan)
                    for m, bm in enumerate(sizes):
                        for k, bn in enumerate(sizes):
                            row = subset.get((bm, bn), {})
                            if field in row:
                                values[m, k] = row[field]
                    finite = values[np.isfinite(values)]
                    norm = LogNorm(vmin=float(finite.min()), vmax=max(float(finite.max()), float(finite.min()) * 1.01)) if finite.size else None
                    cmap = plt.get_cmap('viridis_r').copy()
                    cmap.set_bad('#eeeeee')
                    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm)
                    if finite.size:
                        fig.colorbar(im, ax=ax, shrink=0.8, label='Latency (ms)')
                    for m, bm in enumerate(sizes):
                        for k, bn in enumerate(sizes):
                            row = subset.get((bm, bn), {})
                            if field in row:
                                label = f'{row[field]:.3g}'
                                color = 'white' if im.norm(row[field]) > 0.55 else 'black'
                            else:
                                label = {'oom': 'OOM', 'resource_limit': 'LIMIT'}.get(row.get('status'), 'ERR' if row else '—')
                                color = 'black'
                            ax.text(k, m, label, ha='center', va='center', fontsize=9, color=color)
                    ax.set_xticks(range(len(sizes)), sizes)
                    ax.set_yticks(range(len(sizes)), sizes)
                    ax.set_xlabel('BN (key tile)')
                    ax.set_ylabel('BM (query tile)')
                    ax.set_title(f'{dtype}, N={n}, d=64')
            fig.suptitle(f'Triton {title.lower()}: median ms; each panel uses its own logarithmic color scale')
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            save(fig, 'tile_sensitivity_' + phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted((args.directory / 'cases').glob('*.json'))]
    write_plots(args.directory, rows)
