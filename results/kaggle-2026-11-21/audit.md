# HumSet audit

Rows: 154,104

| Split | Entries | Documents | Projects | Languages |
|---|---|---|---|---|
| train | 129,268 | 16,590 | 1 | en 76,466, es 27,810, fr 24,990, pt 2 |
| validation | 11,940 | 6,037 | 1 | en 7,345, fr 2,516, es 2,079 |
| test | 12,896 | 6,300 | 1 | en 7,900, fr 2,751, es 2,244, pt 1 |

## Cross-split overlap (share of rows in the later split)

| Pair | Same document | Same project | Same excerpt text |
|---|---|---|---|
| train->validation | 97.1% | 100.0% | 2.7% |
| train->test | 96.9% | 100.0% | 2.9% |
| validation->test | 72.0% | 100.0% | 0.4% |

## Labels

| Task | Classes | Labels per entry (train) | Entries with no label (train) |
|---|---|---|---|
| sectors | 11 | 1.086 | 20.4% |
| pillars_1d | 7 | 0.575 | 51.4% |
| pillars_2d | 6 | 0.97 | 25.2% |
| subpillars_1d | 36 | 0.551 | 57.0% |
| subpillars_2d | 19 | 0.997 | 32.0% |

Duplicate texts within splits: {'train': 4210, 'validation': 388, 'test': 448}; duplicate texts with different sector labels: 774.
