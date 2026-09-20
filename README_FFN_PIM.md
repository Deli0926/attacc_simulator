# Sparsity-Aware FFN-PIM Extension for AttAcc

This branch extends the original AttAcc simulator to experiment with executing the LLaMA feed-forward network (FFN) during the generation phase on bank-level PIM.

The current implementation does **not** apply an actual sparsity-aware sorting or offloading policy. Instead, FFN PIM latency is modeled by removing a specified fraction of MAC commands from the generated FFN PIM trace according to the configured sparsity.

This provides a simple way to estimate FFN latency under different sparsity levels.

## Supported Configuration

- Supported models: `LLAMA2-7B`, `LLAMA2-13B`
- Supported PIM mode: `--pim bank`
- Currently unsupported: FFN traces for `--pim bg` and `--pim buffer`
- FFN sparsity is applied in:
  `ramulator2/trace_gen/gen_trace_attacc_bank_ff.py`

## Basic Usage

Run the simulator from the repository root:

```bash
python3 main.py \
  --system dgx-attacc \
  --gpu A100a \
  --ngpu 8 \
  --model LLAMA2-7B \
  --lin 2048 \
  --lout 128 \
  --batch 1 \
  --pim bank \
  --powerlimit
```

The default FFN sparsity is `0.95`.

This means that approximately 95% of FFN MAC commands are removed, leaving approximately 5% of the MAC commands in the generated trace.

## FFN Sparsity Option

FFN sparsity can be specified using `--ffn-sparsity`.

```bash
python3 main.py \
  --system dgx-attacc \
  --gpu A100a \
  --ngpu 8 \
  --model LLAMA2-7B \
  --lin 2048 \
  --lout 128 \
  --batch 1 \
  --pim bank \
  --powerlimit \
  --ffn-sparsity 0.80
```

The valid range is:

```text
0 <= sparsity < 1
```

Examples:

- `0.0`: No MAC commands are removed.
- `0.5`: Approximately 50% of MAC commands are removed.
- `0.95`: Approximately 95% of MAC commands are removed.

## Quick Test

For a quick functionality check, use a shorter input sequence and fewer output tokens.

```bash
python3 main.py \
  --system dgx-attacc \
  --gpu A100a \
  --ngpu 8 \
  --model LLAMA2-7B \
  --lin 16 \
  --lout 2 \
  --batch 1 \
  --pim bank \
  --powerlimit \
  --ffn-sparsity 0.95
```

## Output Files

### `output.csv`

The final simulator results are stored in `output.csv`.

All timing values are reported in milliseconds.

- `g_time (ms)`: Total generation time
- `g_matmul`: PIM attention matmul time
- `g_softmax`: PIM softmax time
- `g_ff_time`: PIM FFN time

To inspect FFN PIM latency, use `g_ff_time`.

### `ramulator.out`

Raw Ramulator results and cached simulation results are stored in `ramulator.out`.

Key columns include:

- `layer_type`: `MATMUL` or `FFN`
- `M`, `N`, `K`: Layer dimensions
- `pim_type`: PIM type
- `sparsity`: FFN sparsity; attention layers use `0`
- `cycle`: Ramulator cycle count
- `mac`, `softmax`, `mvgb`, `mvsb`, `wrgb`: PIM command counts

Raw Ramulator latency can be calculated as:

```text
time = cycle * 0.769 ns
```

For example, if the FFN `cycle` value is `29843`:

```text
29843 * 0.769 ns = 22945 ns = 0.0229 ms
```

The `g_ff_time` value in `output.csv` is the final simulator-level FFN latency after applying simulator scaling factors such as the number of decoder layers to the raw Ramulator latency.

## Notes

- `main.py` deletes the existing `output.csv` and creates a new one on each execution.
- `ramulator.out` caches results using the layer shape, PIM type, power constraint, and sparsity as cache keys.
- When the simulator is executed again with the same configuration, the cached value in `ramulator.out` is reused instead of rerunning Ramulator.
- When a different sparsity value is specified, the `sparsity` cache key changes and a new Ramulator trace is generated.
- `GPT-175B` is currently not mapped to the FFN PIM block. Supporting GPT FFN execution on PIM requires an additional GPT-specific FFN trace generator and model mapping.

## Relation to the Original AttAcc Simulator

This implementation is based on the original AttAcc simulator and keeps the original simulator structure intact while extending the bank-level PIM path to model FFN execution.

For the original project, installation instructions, and AttAcc simulator details, refer to the repository's main `README.md`.
