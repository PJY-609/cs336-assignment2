"""Rebuild the all-reduce comparison table and plot from per-launch summaries."""
import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader((args.directory / 'summary.csv').open()))
    expected = {(size, ranks, repeat) for size in (10**6, 10**7, 10**8, 10**9) for ranks in (2, 4, 6) for repeat in (1, 2, 3)}
    actual = {(int(r['tensor_bytes']), int(r['world_size']), int(r['repeat'])) for r in rows}
    if len(rows) != 36 or actual != expected or any(r['status'] != 'ok' for r in rows):
        raise ValueError('Expected all 36 successful launches in the planned matrix')
    aggregate = []
    table = ['| Size per rank | GPUs | Repeat 1 median / IQR | Repeat 2 median / IQR | Repeat 3 median / IQR | Median of medians |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for size, label in [(10**6, '1 MB'), (10**7, '10 MB'), (10**8, '100 MB'), (10**9, '1 GB')]:
        for ranks in (2, 4, 6):
            group = sorted([r for r in rows if int(r['tensor_bytes']) == size and int(r['world_size']) == ranks], key=lambda r: int(r['repeat']))
            medians = [float(r['median_ms']) for r in group]
            center = statistics.median(medians)
            aggregate.append(dict(tensor_bytes=size, world_size=ranks, median_of_medians_ms=center,
                                  min_repeat_median_ms=min(medians), max_repeat_median_ms=max(medians)))
            values = ' | '.join(f"{float(r['median_ms']):.4f} / {float(r['iqr_ms']):.4f}" for r in group)
            table.append(f'| {label} | {ranks} | {values} | {center:.4f} |')
    (args.directory / 'comparison.md').write_text('All latency values are in milliseconds. Each repeat summarizes 50 iteration-wise rank maxima.\n\n' + '\n'.join(table) + '\n')
    with (args.directory / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    for ranks in (2, 4, 6):
        group = [r for r in aggregate if r['world_size'] == ranks]
        center = [r['median_of_medians_ms'] for r in group]
        ax.errorbar([r['tensor_bytes'] / 1e6 for r in group], center,
                    yerr=[[r['median_of_medians_ms'] - r['min_repeat_median_ms'] for r in group],
                          [r['max_repeat_median_ms'] - r['median_of_medians_ms'] for r in group]],
                    marker='o', capsize=4, label=f'{ranks} GPUs')
    ax.set(xscale='log', yscale='log', xlabel='FP32 tensor size per rank (decimal MB)',
           ylabel='Synchronized host-observed all-reduce latency (ms)',
           title='Single-node A100 SXM all-reduce\nMedian of three repeat medians; bars show repeat range')
    ax.set_xticks([1, 10, 100, 1000], labels=['1', '10', '100', '1000 (1 GB)'])
    ax.grid(True, which='both', alpha=.2)
    ax.legend()
    fig.savefig(args.directory / 'latency.png', dpi=180)
    fig.savefig(args.directory / 'latency.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
