# Benchmarks

We check this library's [GEPA](https://github.com/gepa-ai/gepa) implementation against GEPA's own
benchmark suite. All six benchmarks are ported to Koog, and each one was measured over ten runs:
five with the agent's hand-written instructions, and five with the instructions GEPA produced.

The benchmarks are [PUPA](https://huggingface.co/datasets/Columbia-NLP/PUPA),
[AIME](https://huggingface.co/datasets/AI-MO/aimo-validation-aime),
[IFBench](https://github.com/allenai/IFBench),
[HotPotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa),
[HoVerQA](https://huggingface.co/datasets/hover-nlp/hover) and
[LiveBench Math](https://huggingface.co/datasets/livebench/math), each with the data splits and
rollout budgets GEPA reports. Every run used `gpt-4.1-mini` at temperature 1.0 as both the agent
model and GEPA's reflection model. Scores are on a 0-100 scale, higher is better, and the p-value
compares the five optimized runs against the five baseline runs.

| Benchmark | Baseline mean ± SD [95% CI] | Optimized mean ± SD [95% CI] | Improvement | p-value |
| --- | --- | --- | ---: | --- |
| PUPA | 81.19 ± 1.82 [78.94, 83.44] | 94.22 ± 0.54 [93.55, 94.89] | +13.03 | 3.36e-5 |
| AIME | 35.07 ± 2.77 [31.62, 38.51] | 43.60 ± 0.60 [42.86, 44.34] | +8.53 | 0.00184 |
| IFBench | 45.07 ± 0.60 [44.32, 45.81] | 48.16 ± 0.87 [47.08, 49.24] | +3.09 | 3.01e-4 |
| HotPotQA | 12.33 ± 1.31 [10.70, 13.96] | 65.00 ± 0.97 [63.79, 66.21] | +52.67 | 8.93e-12 |
| HoVerQA | 36.20 ± 1.26 [34.63, 37.77] | 41.40 ± 1.30 [39.79, 43.01] | +5.20 | 2.05e-4 |
| LiveBench Math | 51.90 ± 1.50 [50.04, 53.77] | 52.23 ± 2.39 [49.26, 55.20] | +0.33 | 0.804 |

Five of the six improve. LiveBench Math is the exception, where the optimized agent lands within
noise of the baseline.
