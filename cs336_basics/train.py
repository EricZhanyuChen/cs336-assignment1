import argparse
import numpy as np
import os
import torch

import wandb

from cs336_basics.transformer import(
    Transformer,
    AdamW,
    save_checkpoint,
    load_checkpoint,
    get_batch,
    cross_entropy,
    lr_cosine_schedule,
    gradient_clipping
)

from cs336_basics.tokenizer import Tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Transformer Training Loop")

    parser.add_argument("--device", type=str, 
    default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu",
    help="training device (detected automatically)")
    parser.add_argument("--train_data_path", type=str, required=True,
    help="tokenized training set path")
    parser.add_argument("--val_data_path", type=str, required=True,
    help="tokenized evalutation set path")
    parser.add_argument("--ckpt_save_path", type=str, default="./ckpt",
    help="saving path for checkpoints")
    parser.add_argument("--ckpt_load_path", type=str, default=None,
    help="checkpoints loading path")
    parser.add_argument("--wandb_name", type=str, default=None,
    help="custom name for wandb run")

    parser.add_argument("--vocab_size", type=int, default=10000,
    help="vocab size")
    parser.add_argument("--d_model", type=int, default=512,
    help="dimension of transformer")
    parser.add_argument("--d_ff", type=int, default=1344,
    help="feedforward layer dimension")
    parser.add_argument("--num_layers", type=int, default=4,
    help="number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=16,
    help="number of transformer heads")
    parser.add_argument("--context_length", type=int, default=256,
    help="max context length")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
    help="theta value for rope")

    parser.add_argument("--batch_size", type=int, default=32,
    help="batch size")
    parser.add_argument("--total_steps", type=int, default=5000,
    help="total training steps")
    parser.add_argument("--val_every", type=int, default=100,
    help="evaluation steps")
    parser.add_argument("--save_steps", type=int, default=500,
    help="save steps")

    parser.add_argument("--lr_max", type=float, default=3e-4,
    help="max learning rate for learning rate scheduling")
    parser.add_argument("--lr_min", type=float, default=1e-5,
    help="max learning rate for learning rate scheduling")
    parser.add_argument("--warmup_iters", type=int, default=500,
    help="warmup steps for cosine learning rate scheduling")
    parser.add_argument("--cosine_iter_cycle", type=int, default=5000,
    help="cosine learning rate scheduling steps, the same as total training steps")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
    help="weight decay for AdamW")
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.95],
    help="beta values for AdamW")
    parser.add_argument("--grad_clipping_norm", type=float, default=1.0,
    help="max L2 norm for gradient clipping")

    return parser.parse_args()
    
def init_logger_and_dir(args):
    os.makedirs(args.ckpt_save_path, exist_ok=True)
    run_name = args.wandb_name if args.wandb_name else f"d_model_{args.d_model}_layers_{args.num_layers}"
    wandb.init(
        project="cs336-assignment1",
        config=vars(args),
        name=run_name
    )

def load_tokenized_datasets(args):
    train_data = np.memmap(
        args.train_data_path,
        dtype=np.uint16,
        mode="r"
    )

    val_data = np.memmap(
        args.val_data_path,
        dtype=np.uint16,
        mode="r"
    )
    print(f"Dataset loaded successfully")
    print(f"  Total token in training set：{len(train_data):,}")
    print(f"  Total token in dev set：{len(val_data):,}")
    return train_data, val_data

def init_model_optimizer(args):
    model = Transformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        weights={},
        vocab_size=args.vocab_size
    ).to(args.device)

    if args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    elif args.device == "cpu":
        model = torch.compile(model)
    elif args.device == "cuda":
        torch.set_float32_matmul_precision("high")
        try:
            model = torch.compile(model)
            print("🚀 torch.compile enabled successfully!")
        except Exception as e:
            print(f"⚠️ torch.compile failed, falling back to eager mode: {e}")
    
    optimizer = AdamW(
        params=model.parameters(),
        lr=args.lr_max,
        betas=tuple(args.betas),
        weight_decay=args.weight_decay
    )

    start_step = 0
    if args.ckpt_load_path is not None:
        start_step = load_checkpoint(
            src=args.ckpt_load_path,
            model=model,
            optimizer=optimizer
        )
        print(f"Resume training from step {start_step}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model and optimizer initialized")
    print(f"Total model parameters: {total_params:,}")
    print(f"training device: {args.device}")

    return model, optimizer, start_step


if __name__ == "__main__":
    args = parse_args()
    print("Argument parsed sucessfully!")
    print(f"  Current device：{args.device}")
    print(f"  AdamW beta：{args.betas}")
    
    init_logger_and_dir(args)
    print(f"✅ log initialized，Checkpoint path：{args.ckpt_save_path}")
    
    train_data, val_data = load_tokenized_datasets(args)

    model, optimizer, start_step = init_model_optimizer(args)

    print(f"Model initialized! Starting step：{start_step}")

    best_val_loss = float('inf')
    best_val_step = 0
    for _ in range(start_step, args.total_steps):
        lr = lr_cosine_schedule(_, args.lr_max, args.lr_min, args.warmup_iters, args.cosine_iter_cycle)

        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.zero_grad()
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

        logits = model(x, args.context_length, args.rope_theta)
        loss = cross_entropy(logits, y)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clipping_norm)
        optimizer.step()
        
        if _ % 50 == 0: 
             print(f"Step {_}: loss {loss.item():.4f}, lr {lr:.6f}")
        wandb.log({
            "train/loss": loss.item(),
            "train/perplexity": np.exp(loss.item()),
            "train/lr": lr,
            "step": _
        })
        
        if _ > 0 and _ % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, args.batch_size, args.context_length, args.device)
                val_logits = model(x_val, args.context_length, args.rope_theta)
                val_loss = cross_entropy(val_logits, y_val)
                print(f"--- Step {_}: Validation Loss {val_loss.item():.4f} ---")
                wandb.log({"val/loss": val_loss.item(), "val/perplexity": np.exp(val_loss.item())})

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_val_step = _
                    best_ckpt_path = os.path.join(args.ckpt_save_path, "best_model.pt")
                    save_checkpoint(model, optimizer, _, best_ckpt_path)
                    print(f" [Best model saved] val_loss={best_val_loss:.4f} at step {_}")
            model.train()

        if _ > 0 and _ % args.save_steps == 0:
            cpkt_file = os.path.join(args.ckpt_save_path, f"step_{_}.pt")
            save_checkpoint(model, optimizer, _, cpkt_file)

 





