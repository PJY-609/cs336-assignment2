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
    colors = {'naive': '#d95f02', 'flash_pytorch': '#7570b3', 'flash_triton': '#1b9e77'}
    labels = {'naive': 'Dense PyTorch (compiled)', 'flash_pytorch': 'Tiled PyTorch', 'flash_triton': 'Triton FA2'}

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
        for phase, title in (('forward', 'Forward'), ('forward_backward', 'Forward + backward')):
            field = phase + '_peak_gb'
            fig, axes = plt.subplots(len(dtypes), len(dims), squeeze=False,
                                     figsize=(4.2 * len(dims), 3.6 * len(dtypes)))
            for i, dtype in enumerate(dtypes):
                for j, d in enumerate(dims):
                    ax = axes[i, j]
                    for index, (impl, color) in enumerate(colors.items()):
                        subset = sorted((row for row in main_rows if row['dtype'] == dtype
                                         and row['embedding_dim'] == d and row['implementation'] == impl),
                                        key=lambda row: row['sequence_length'])
                        measured = [row for row in subset if field in row]
                        ax.plot([row['sequence_length'] for row in measured],
                                [row[field] for row in measured], 'o-', markersize=3,
                                color=color, label=labels[impl])
                        for row in subset:
                            if row.get('status') == 'oom' and field not in row:
                                # An OOM has no numeric memory measurement: annotate at
                                # the top of the axes rather than inventing a y value.
                                ax.text(row['sequence_length'], 0.98 - index * 0.11,
                                        f'OOM\n{labels[impl]}', color=color, fontsize=6,
                                        ha='center', va='top', transform=ax.get_xaxis_transform())
                    ax.set_xscale('log', base=2)
                    ax.set_xticks(all_n, [f'{n // 1024}K' if n >= 1024 else str(n) for n in all_n], rotation=45)
                    ax.set_ylim(bottom=0)
                    ax.set_title(f'{dtype}, d={d}')
                    ax.set_xlabel('Sequence length N')
                    ax.set_ylabel('Peak allocated GPU memory (GB)')
                    ax.grid(alpha=0.25)
            handles, legend_labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, legend_labels, loc='lower center', ncol=3, fontsize=9)
            fig.suptitle(f'{title}: single H200, batch=1, causal — allocated memory, including inputs')
            fig.tight_layout(rect=(0, 0.07, 1, 0.94))
            save(fig, 'memory_' + phase)

    tiles = [row for row in rows if row.get('experiment') == 'tile_sensitivity']
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
