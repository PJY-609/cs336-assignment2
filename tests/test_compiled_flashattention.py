"""Correctness checks for both execution modes; these are not benchmarks."""
import pytest
import torch

from cs336_systems.flashattention import CompiledFlashAttention, FlashAttention


@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_tiled_output_and_manual_backward(is_causal, dtype):
    torch.manual_seed(42)
    # Multiple batches, unequal lengths, partial tiles and strided inputs/dO.
    inputs = tuple(torch.randn(2, 8, n, dtype=dtype).transpose(-2, -1).requires_grad_()
                   for n in (19, 23, 23))
    reference_inputs = tuple(x.detach().float().requires_grad_() for x in inputs)
    q, k, v = reference_inputs
    scores = q @ k.transpose(-2, -1) / 8**0.5
    if is_causal:
        scores = scores.masked_fill(torch.arange(19)[:, None] < torch.arange(23)[None, :], float('-inf'))
    reference = scores.softmax(-1) @ v
    do = torch.randn(2, 8, 19, dtype=dtype).transpose(-2, -1)
    expected_grads = torch.autograd.grad(reference, reference_inputs, do.float())
    tolerance = .05 if dtype == torch.bfloat16 else 1e-5
    for implementation in (FlashAttention, CompiledFlashAttention):
        actual = implementation.apply(*inputs, is_causal, 16)
        gradients = torch.autograd.grad(actual, inputs, do, retain_graph=True)
        for value, expected in zip((actual, *gradients), (reference, *expected_grads)):
            torch.testing.assert_close(value.float(), expected, atol=tolerance, rtol=tolerance)
        # The benchmark times repeated backward on one retained forward graph.
        again = torch.autograd.grad(actual, inputs, do, retain_graph=True)
        for value, expected in zip(again, gradients):
            torch.testing.assert_close(value, expected)


def test_compiled_default_signature_and_tile_validation():
    qkv = tuple(torch.randn(1, 5, 8, requires_grad=True) for _ in range(3))
    output = CompiledFlashAttention.apply(*qkv)
    gradients = torch.autograd.grad(output.sum(), qkv)
    assert all(torch.isfinite(g).all() for g in gradients)
    for bad_tile in (0, -1, 1.5):
        with pytest.raises(ValueError, match='positive integer'):
            CompiledFlashAttention.apply(*qkv, True, bad_tile)
