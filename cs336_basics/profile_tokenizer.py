import sys
sys.path.append("cs336_basics")  # 确保能导入你的 tokenizer 模块
from cs336_basics.tokenizer import BPE_tokenizer  # 导入你的 BPE 训练函数

def main():
    # 配置测试参数（用小数据集快速 profiling，避免耗时过久）
    input_path = "tests/fixtures/corpus.en"  # 作业提供的测试数据集（小而快）
    vocab_size = 500  # 小词汇量，缩短训练时间
    special_tokens = ["<<|endoftext|>"]
    
    # 调用 BPE 训练函数（profiling 会监控此过程）
    vocab, merges = BPE_tokenizer(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    print(f"Profiling completed. Vocab size: {len(vocab)}")

if __name__ == "__main__":
    main()