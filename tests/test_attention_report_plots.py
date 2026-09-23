"""Report coverage and axis checks with synthetic data, never benchmark results."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import plot_attention_benchmarks
import plot_batch_sizes


def records(experiment):
    rows = []
    for index, impl in enumerate(('naive', 'flash_pytorch', 'flash_pytorch_compiled', 'flash_triton')):
        for n in (128, 2048):
            for b in (1, 16) if experiment == 'batch_sweep' else (1,):
                value = (index + 1) * n * b / 1000
                row = dict(implementation=impl, dtype='float32', sequence_length=n,
                           embedding_dim=64, batch_size=b, experiment=experiment, status='ok')
                for field in ('forward_ms', 'backward_ms', 'forward_backward_ms',
                              'forward_peak_gb', 'forward_backward_peak_gb', 'tokens_per_second'):
                    row[field] = value
                rows.append(row)
    return rows


def test_four_series_and_no_clipping_in_reports(tmp_path, monkeypatch):
    captured = []

    def inspect_figure(fig, path, **kwargs):
        if str(path).endswith('.png'):
            for ax in fig.axes:
                labels = {line.get_label() for line in ax.lines}
                assert 'Tiled PyTorch (compiled)' in labels
                low, high = ax.get_ylim()
                for line in ax.lines:
                    values = np.asarray(line.get_ydata())
                    values = values[np.isfinite(values)]
                    assert np.all((values >= low) & (values <= high))
            Path(path).touch()  # Save helper atomically renames this temporary file.
            captured.append(fig)
        else:
            Path(path).touch()

    monkeypatch.setattr(plt.Figure, 'savefig', inspect_figure)
    plot_attention_benchmarks.write_plots(tmp_path, records('main'))
    assert len(captured) == 5
    config = dict(batch_sizes=[1, 16], sequence_lengths=[128, 2048], dimensions=[64],
                  dtypes=['float32'], implementations=['naive', 'flash_pytorch', 'flash_pytorch_compiled', 'flash_triton'])
    plot_batch_sizes.write_plots(tmp_path, records('batch_sweep'), config)
    assert len(captured) == 11


def test_pytorch_tile_failures_are_gaps_not_triton_cells(tmp_path, monkeypatch):
    captured = []

    def inspect_figure(fig, path, **kwargs):
        Path(path).touch()
        if str(path).endswith('.png'):
            assert 'tile_sensitivity_pytorch' in str(path)
            for ax in fig.axes:
                compiled = next(line for line in ax.lines if line.get_label() == 'Tiled PyTorch (compiled)')
                assert np.isnan(compiled.get_ydata()[-1])
                assert 'TIME' in [text.get_text() for text in ax.texts]
            captured.append(fig)

    monkeypatch.setattr(plt.Figure, 'savefig', inspect_figure)
    rows = []
    for impl in ('flash_pytorch', 'flash_pytorch_compiled'):
        for tile in (256, 512):
            row = dict(implementation=impl, dtype='float32', sequence_length=1024,
                       embedding_dim=64, q_tile=tile, k_tile=tile, experiment='tile_sensitivity',
                       status='ok', forward_ms=1., backward_ms=2., forward_backward_ms=3.)
            if impl == 'flash_pytorch_compiled' and tile == 512:
                row = {k: v for k, v in row.items() if not k.endswith('_ms')}
                row['status'] = 'timeout'
            rows.append(row)
    plot_attention_benchmarks.write_plots(tmp_path, rows)
    assert len(captured) == 3
