# ATTACC Minseo FFN-PIM 사용법

이 버전은 AttAcc simulator를 기반으로 generation 단계의 LLaMA FFN을 bank-level PIM에서 실행하도록 실험한 코드입니다. 현재 FFN은 실제 sparsity별 정렬/offload 정책이 아니라, FFN PIM trace의 MAC command를 지정한 sparsity만큼 제거해서 latency를 얻는 방식입니다.

## 지원 범위

- 지원 모델: `LLAMA2-7B`, `LLAMA2-13B`
- 지원 PIM 모드: `--pim bank`
- 현재 미지원: `--pim bg`, `--pim buffer`의 FFN trace
- FFN sparsity 적용 위치: `ramulator2/trace_gen/gen_trace_attacc_bank_ff.py`

## 기본 실행

```bash
cd /home/kms20201444/attacc_minseo

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

기본 FFN sparsity는 `0.95`입니다. 즉 FFN MAC command의 약 95%를 제거하고, 약 5%만 남기는 trace를 생성합니다.

## Sparsity 옵션

`--ffn-sparsity`로 FFN sparsity를 지정할 수 있습니다.

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

값 범위는 `0 <= sparsity < 1`입니다.

- `0.0`: MAC command를 제거하지 않음
- `0.5`: MAC command를 약 50% 제거
- `0.95`: MAC command를 약 95% 제거

## 짧은 테스트 실행

빠르게 동작만 확인하려면 sequence 길이와 output token 수를 작게 두면 됩니다.

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

## 결과 파일

### output.csv

최종 simulator 결과는 `output.csv`에 저장됩니다.

시간은 모두 ms 단위입니다.

- `g_time (ms)`: generation 전체 시간
- `g_matmul`: PIM attention matmul 시간
- `g_softmax`: PIM softmax 시간
- `g_ff_time`: PIM FFN 시간

FFN PIM latency를 볼 때는 `g_ff_time`을 확인하면 됩니다.

### ramulator.out

Ramulator raw 결과와 cache는 `ramulator.out`에 저장됩니다.

주요 Column:

- `layer_type`: `MATMUL` 또는 `FFN`
- `M`, `N`, `K`: layer shape
- `pim_type`: PIM type
- `sparsity`: FFN sparsity. attention은 `0`
- `cycle`: Ramulator cycle
- `mac`, `softmax`, `mvgb`, `mvsb`, `wrgb`: PIM command count

Ramulator raw latency는 다음처럼 계산할 수 있습니다.

```text
time_seconds = cycle * 0.769 ns
```

예를 들어 FFN의 `cycle`이 `29843`이면:

```text
29843 * 0.769 ns = 22945 ns = 0.0229 ms
```

`output.csv`의 `g_ff_time`은 이 raw latency에 decoder layer 수 등 simulator scaling을 적용한 최종 값입니다.

## 주의 사항

- `main.py`는 실행할 때 기존 `output.csv`를 삭제하고 새로 씁니다.
- `ramulator.out`은 layer shape, PIM type, power constraint, sparsity를 key로 cache됩니다.
- 같은 조건으로 다시 실행하면 Ramulator를 반복 실행하지 않고 `ramulator.out`의 값을 재사용합니다.
- 다른 sparsity를 주면 `sparsity` key가 달라져 새로 Ramulator trace를 생성합니다.
- `GPT-175B`는 현재 FFN PIM block으로 변환되지 않습니다. GPT FFN을 PIM에서 보려면 GPT용 FFN trace generator와 model mapping이 추가로 필요합니다.
