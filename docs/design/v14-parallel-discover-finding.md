# v14 parallel batch discovery

Date: 2026-05-21
Status: closed (small but real speedup)

## Question

The cross-shard chain study (v13) noted that chains traverse ~2
distinct shards on average and 95%+ of endpoints land on different
shards at realistic shard counts. This suggests parallel per-hop
queries could speed up walks. But INTRA-chain parallelism is
impossible -- each hop's start node IS the previous hop's answer,
so they must run sequentially.

The remaining opportunity is BATCH parallelism: when we have many
independent (start, target) probes to discover, run them concurrently.

## Approach

`rck/parallel_discover.py` wraps `discover_chains` in a
ThreadPoolExecutor. Each probe is independent; the shared KB is
read-only during discovery, and numpy releases the GIL inside the
HRR cleanup ops.

## Benchmark

80 transitive probes on the commonsense KB at auto-sized shards.

| workers | sequential | parallel | speedup |
|---------|------------|----------|---------|
| 2       | 779 ms     | 594 ms   | 1.31x   |
| 4       | 779 ms     | 573 ms   | 1.36x   |
| 8       | 779 ms     | 630 ms   | 1.24x   |

Hit count is identical across all configurations (78/80). The
speedup peaks around 4 workers, then degrades as scheduler
overhead overtakes the GIL-released chunks.

## When this matters

* Mass evaluation: running an analogy or chain benchmark against
  the KB.
* Maintenance: a cascade-induction pass that probes many
  (start, target) pairs.
* Initial KB warming: discovering all 2-hop transitive shortcuts
  in one batch instead of N sequential calls.

## When it doesn't

* Single ad-hoc query latency: unchanged.
* Single-threaded environments / WASM: ThreadPool is unavailable
  or doesn't get the GIL-release benefit.

## Open work

* Per-hop dispatch within a chain: would require an HRR substrate
  that supports concurrent reads across shards (likely fine since
  shard tensors are read-only mid-chain). Not yet measured.
* Worker count auto-tuning: currently a user parameter. Could be
  derived from os.cpu_count() with a cap.
