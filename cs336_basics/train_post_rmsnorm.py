"""
Training script for Post-norm ablation.
Uses TransformerPostRMSNorm instead of standard pre-norm Transformer.
"""
import argparse
import numpy as np
import os
import torch

import wandb

from cs336_basics.ablation_post_rmsnorm import (
    TransformerPostRMSNorm,
    AdamW,
    save_checkpoint,
    load_checkpoint,
    get_batch,
    cross_entropy,
    lr_cosine_schedule,
    gradient_clipping
)

def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Post-norm Ablation Training")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--ckpt_save_path", type=str, default="./ckpt")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--total_steps", type=int, default=5000)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--lr_max", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--cosine_iter_cycle", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.95])
    parser.add_argument("--grad_clipping_norm", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print(f"=== Post-norm Ablation: lr={args.lr_max} ===")

    os.makedirs(args.ckpt_save_path, exist_ok=True)
    run_name = args.wandb_name if args.wandb_name else f"postnorm_lr_{args.lr_max}"
    wandb.init(project="cs336-assignment1", config=vars(args), name=run_name)

    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

    model = TransformerPostRMSNorm(
        d_model=args.d_model, num_layers=args.num_layers,
        num_heads=args.num_heads, d_ff=args.d_ff,
        vocab_size=args.vocab_size, eps=args.eps, weights={}
    ).to(args.device)

    if args.device == "cuda":
        torch.set_float32_matmul_precision("high")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}")

    optimizer = AdamW(params=model.parameters(), lr=args.lr_max,
                      betas=tuple(args.betas), weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,} (post-norm)")

    best_val_loss = float('inf')
    for step in range(args.total_steps):
        lr = lr_cosine_schedule(step, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_iter_cycle)
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x, args.context_length, args.rope_theta)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clipping_norm)
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: loss {loss.item():.4f}, lr {lr:.6f}")
        wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})

        if step > 0 and step % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, args.device)
                val_logits = model(x_val, args.context_length, args.rope_theta)
                val_loss = cross_entropy(val_logits, y_val)
                print(f"--- Step {step}: Val Loss {val_loss.item():.4f} ---")
                wandb.log({"val/loss": val_loss.item()})
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    save_checkpoint(model, optimizer, step, os.path.join(args.ckpt_save_path, "best_model.pt"))
                    print(f"  [Best] val_loss={best_val_loss:.4f}")
            model.train()

        if step > 0 and step % args.save_steps == 0:
            save_checkpoint(model, optimizer, step, os.path.join(args.ckpt_save_path, f"step_{step}.pt"))

    wandb.finish()
    print(f"Done! Best val_loss: {best_val_loss:.4f}")
