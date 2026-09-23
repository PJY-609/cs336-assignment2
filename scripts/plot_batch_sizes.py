"""Regenerate the six batch-size figures from saved case records, without a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_plots(directory, rows, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    output = directory / 'plots'
    output.mkdir(exist_ok=True)
    styles = {'naive': ('#d95f02', 'o', 'Dense PyTorch (compiled)'),
              'flash_pytorch': ('#7570b3', 's', 'Tiled PyTorch'),
              'flash_triton': ('#1b9e77', '^', 'Triton FA2')}
    metrics = [('forward_ms', 'Forward latency (ms)'), ('backward_ms', 'Backward latency (ms)'),
               ('forward_backward_ms', 'Forward + backward latency (ms)'),
               ('tokens_per_second', 'Forward + backward throughput (tokens/s)'),
               ('forward_peak_gb', 'Forward peak allocated memory (GB)'),
               ('forward_backward_peak_gb', 'Forward + backward peak allocated memory (GB)')]
    batches = sorted(config['batch_sizes'])
    lengths, dtypes = config['sequence_lengths'], config['dtypes']
    lookup = {(r['implementation'], r['dtype'], r['sequence_length'], r['batch_size']): r for r in rows}
    for field, title in metrics:
        fig, axes = plt.subplots(len(dtypes), len(lengths), squeeze=False, sharey=True,
                                 figsize=(4.6 * len(lengths), 3.8 * len(dtypes)))
        for i, dtype in enumerate(dtypes):
            for j, length in enumerate(lengths):
                ax = axes[i, j]
                for index, impl in enumerate(config['implementations']):
                    color, marker, label = styles[impl]
                    subset = [lookup.get((impl, dtype, length, b), {}) for b in batches]
                    # NaNs preserve gaps rather than connecting across failed/pending cases.
                    ax.plot(batches, [r.get(field, np.nan) for r in subset], color=color, marker=marker, label=label)
                    for b, row in zip(batches, subset):
                        if row.get('status') in ('oom', 'resource_limit', 'error', 'timeout'):
                            text = {'oom': 'OOM', 'resource_limit': 'LIMIT', 'error': 'ERR', 'timeout': 'TIME'}.get(row['status'])
                            if field in row:
                                text += '*'
                            ax.text(b, .98 - .10 * index, text, color=color, fontsize=6,
                                    ha='center', va='top', transform=ax.get_xaxis_transform())
                ax.set_xscale('log', base=2)
                ax.set_xticks(batches, [str(b) for b in batches])
                if field.endswith('_ms') and any(r.get(field, 0) > 0 for r in rows):
                    ax.set_yscale('log')
                else:
                    ax.set_ylim(bottom=0)
                ax.set_title(f'{dtype}, N={length:,}, d={config["dimensions"][0]}')
                ax.set_xlabel('Batch size')
                if j == 0:
                    ax.set_ylabel(title)
                ax.grid(alpha=.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3)
        fig.suptitle(f'{title} — H200, causal, fixed baseline tiles\n* Measurement retained from a case that failed at a later stage')
        fig.tight_layout(rect=(0, .07, 1, .92))
        for extension in ('png', 'pdf'):
            temporary = output / f'{field}.tmp.{extension}'
            fig.savefig(temporary, dpi=160, bbox_inches='tight')
            temporary.replace(output / f'{field}.{extension}')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    config = json.loads((args.directory / 'config.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((args.directory / 'cases').glob('*.json'))]
    write_plots(args.directory, rows, config)
