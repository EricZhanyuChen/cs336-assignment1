import argparse
import torch

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import(
    Transformer,
    AdamW,
    load_checkpoint,
    softmax
)

def parse_args():
    parser = argparse.ArgumentParser(description="CS336 Generating Text")

    parser.add_argument("--ckpt_load_path", type=str, default=None,
                        help="checkpoints loading path")
    parser.add_argument("--device", type=str, 
    default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--d_model", type=int, default=512, 
                        help="dimension of transformer, default is 512")
    parser.add_argument("--d_ff", type=int, default=1344,
                        help="dimension of feedforward layer, default is 1344")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="number of transformer layers, default is 4")
    parser.add_argument("--num_heads", type=int, default=16,
                        help="number of heads, default is 16")
    parser.add_argument("--vocab_size", type=int, default=10000,
                        help="vocab size, default is 10000")
    parser.add_argument("--context_length", type=int, default=256,
                        help="max context length")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="theta value for rope")
    
    parser.add_argument("--prompt", type=str, required=True, help="prompt")
    parser.add_argument("--max_tokens", type=int, required=True, default=2048,
                        help="max generated tokens, default is 2048")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature when generating tokens")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="top p value")
    
    parser.add_argument("--vocab_path", type=str, 
                        default="data/owt_vocab.json", 
                        help="Path to vocab.json")
    parser.add_argument("--merge_path", type=str, 
                        default="data/owt_merge.txt",
                        help="Path to merges.txt")
    return parser.parse_args()

def init_model_optimizer(args):
    model = Transformer(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        vocab_size=args.vocab_size,
        weights={}
    ).to(args.device)

    if args.device == "mps":
        model = torch.compile(model, backend="aot_eager")
    elif args.device == "cpu":
        model = torch.compile(model)
    elif args.device == "cuda":
        torch.set_float32_matmul_precision("high")
        model = torch.compile(model)

    optimizer = AdamW(model.parameters())

    last_step = load_checkpoint(args.ckpt_load_path, model, optimizer)

    model.eval()
    
    print(f"Model initialized. Loaded checkpoint from step {last_step}")

    return model


def generate(model, tokenizer, prompt, max_tokens, temperature, top_p, device, context_length, rope_theta):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    eos_token_id = tokenizer.token_str_to_int.get("<|endoftext|>", None)
    
    for _ in range(max_tokens):
        curr_input = input_ids[:, -context_length:]
        with torch.no_grad():
            logits = model(curr_input, context_length, rope_theta)

        last_logit = logits[0, -1, :]
        
        if temperature == 0:
            next_token_id = torch.argmax(last_logit).item()
        else:
            logits_with_temperature = last_logit / temperature

            sorted_logits, sorted_indices = torch.sort(logits_with_temperature, descending=True)
            sorted_probs = softmax(sorted_logits, dim=-1)

            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            last_logit[indices_to_remove] = float("-inf")
            probs = softmax(last_logit, dim=-1)

            next_token_id = torch.multinomial(probs, num_samples=1).item()
        if next_token_id == eos_token_id:
            break

        tokens.append(next_token_id)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)
        
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    args = parse_args()

    model = init_model_optimizer(args)
    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merge_path,
        special_tokens=["<|endoftext|>"]
    )
    decoded_text = generate(model, tokenizer, args.prompt, args.max_tokens, args.temperature, args.top_p, args.device, args.context_length, args.rope_theta)
    
    print(f"\nGenerated Output: \n{decoded_text}")
