# CS336 Assignment 1 - Experiments

## Structure
```
experiments/
├── ts_lr_sweep/          # TinyStories LR sweep (completed)
├── ts_bs_sweep/          # TinyStories BS sweep (completed)
├── ts_ablation_rmsnorm/  # Ablation: no RMSNorm (completed)
├── ts_ablation_postnorm/ # Ablation: Post-norm (completed)
├── ts_ablation_nope/     # Ablation: no RoPE (completed)
├── ts_ablation_silu/     # Ablation: SiLU vs SWiGLU (completed)
├── owt_baseline/         # OWT baseline for 7.4 (running)
├── owt_qknorm/           # 7.5: QK-norm (not started)
└── owt_weight_tying/     # 7.5: weight tying (not started)
```

Each experiment has its own `train.sh`. Run from experiment dir: `sbatch train.sh`
Logs go to `../logs/`, checkpoints to `/scratch/s6398820/cs336/checkpoints/`.

## Results Summary

### LR Sweep (TS, bs=32, 40K steps)
| LR | Best Val Loss | WandB Run |
|----|--------------|-----------|
| **1e-3** | **1.5509** | d_model_512_layers_4_le-3_run2 |
| 5e-4 | 1.7419 | d_model_512_layers_4_5e-4 |
| 3e-4 | 1.7550 | d_model_512_layers_4_3e-4 |
| 1e-4 | 1.9980 | d_model_512_layers_4_1e-4 |

### BS Sweep (TS, tuned LR)
| BS | LR | Best Val Loss | WandB Run |
|----|----|--------------|-----------|
| **64** | **1.4e-3** | **1.3082** 🏆 | bs_sweep_bs64_lr1.4e-3 |
| 32 | 1e-3 | 1.3172 | bs_sweep_bs32_lr1e-3 |
| 256 | 2.8e-3 | 1.3294 | bs_sweep_bs256_lr2.8e-3_run2 |
| 128 | 2.0e-3 | 1.3499 | bs_sweep_bs128_lr2.0e-3 |
| 1 | 1.7e-4 | 1.3857 | bs_sweep_bs1_lr1.7e-4 |
| 512 | 4.0e-3 | failed | bs_sweep_bs512_lr4.0e-3 |

**Note:** bs=64 best_model.pt was NOT saved. Use step_17500.pt as approximation.

### Ablations (TS, vs SWiGLU baseline=1.3082)
| Experiment | Best Val Loss | Δ vs Baseline | WandB Run |
|-----------|--------------|---------------|-----------|
| SWiGLU (baseline) | 1.3082 | — | bs_sweep_bs64_lr1.4e-3 |
| Post-norm | 1.5107 | +0.203 | (from logs) |
| SiLU (replacing SWiGLU) | 1.5052 | +0.197 | ablation_silu_lr1e-3 |
| NoPE (no RoPE) | 1.5961 | +0.288 | ablation_nope_lr1e-3 |
| No RMSNorm | 1.8028 | +0.495 | ablation_no_rmsnorm_lr1e-4 |

**Conclusion:** SWiGLU > Post-norm ≈ SiLU > NoPE > No RMSNorm

### OWT Baseline (7.4) - IN PROGRESS
- Job: 27855573
- Config: bs=64, lr=1e-3, vocab_size=32000, 20480 steps (~335M tokens)
- Vocab fix applied: owt_vocab.json (ID→token) → owt_vocab_fixed.json (token→ID)

## 7.5 Leaderboard Plan
1. Wait for OWT baseline result
2. Add QK-norm (RMSNorm on Q/K vectors before attention)
3. Add weight tying (share embedding and LM head weights)
4. Optional: Muon optimizer

## Model Config
- d_model=512, d_ff=1344, num_layers=4, num_heads=16
- context_length=256, rope_theta=10000
- TS vocab=10000, OWT vocab=32000
- Total params (TS): ~22.8M (with embeddings)
- Total params (OWT): ~33.9M (with embeddings)

## Checkpoint Locations
- `/home5/s6398820/projects/cs336/assignment1/checkpoints/`
- `/scratch/s6398820/cs336/checkpoints/` (same content, symlinked)

## WandB
- Project: `zhanyu-university-of-groningen/cs336-assignment1`
- API key: `~/.netrc`
- Summary shows FINAL val_loss, not best. Check Charts for best.

## Known Issues
- `train.py` doesn't support `--eps` argument
- `encode_text.py` expects `owt_32k.json` (not `owt_vocab.json`)
- OWT vocab original format is inverted (ID→token, needs conversion)
- `best_model.pt` saving was only added to ablation scripts, not lr/bs sweep scripts
