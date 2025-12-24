#!/usr/bin/env python3
"""
vLLM性能基准测试脚本
测试不同配置下的推理性能
"""

import argparse
import time
import json
from typing import List, Dict
import numpy as np

def run_benchmark(
    model_path: str,
    num_prompts: int = 100,
    prompt_length: int = 128,
    output_length: int = 256,
    batch_sizes: List[int] = None
):
    """运行性能基准测试"""

    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]

    try:
        from vllm import LLM, SamplingParams

        print("=" * 60)
        print("vLLM性能基准测试")
        print("=" * 60)
        print(f"\n配置:")
        print(f"  模型: {model_path}")
        print(f"  测试样本数: {num_prompts}")
        print(f"  输入长度: ~{prompt_length} tokens")
        print(f"  输出长度: {output_length} tokens")
        print(f"  批次大小: {batch_sizes}")
        print()

        # 加载模型（Deepseek V3使用FP8）
        print("加载模型...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=8,
            dtype="auto",  # 自动检测FP8
            kv_cache_dtype="fp8",  # FP8 KV缓存
            max_model_len=8192,
            gpu_memory_utilization=0.95,
            trust_remote_code=True
        )
        print("✓ 模型加载完成\n")

        # 生成测试prompts
        print("生成测试prompts...")
        test_prompt = "人工智能技术 " * (prompt_length // 6)  # 大约每个词6个字符
        prompts = [f"问题{i}: {test_prompt}" for i in range(num_prompts)]
        print(f"✓ 生成 {len(prompts)} 个测试样本\n")

        # 采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=output_length
        )

        # 运行基准测试
        results = []

        print("=" * 60)
        print("开始基准测试...")
        print("=" * 60)

        for batch_size in batch_sizes:
            print(f"\n测试批次大小: {batch_size}")
            print("-" * 60)

            # 分批处理
            batch_prompts = prompts[:batch_size]

            # 预热
            print("预热中...")
            _ = llm.generate(batch_prompts[:min(2, batch_size)], sampling_params)

            # 正式测试
            print("测试中...")
            start_time = time.time()
            outputs = llm.generate(batch_prompts, sampling_params)
            duration = time.time() - start_time

            # 统计
            total_input_tokens = sum(len(p.split()) * 1.3 for p in batch_prompts)  # 粗略估计
            total_output_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
            total_tokens = total_input_tokens + total_output_tokens

            throughput = total_output_tokens / duration
            latency = duration / batch_size

            result = {
                "batch_size": batch_size,
                "duration": duration,
                "total_input_tokens": int(total_input_tokens),
                "total_output_tokens": total_output_tokens,
                "total_tokens": int(total_tokens),
                "throughput": throughput,
                "latency": latency,
                "tokens_per_second": total_tokens / duration
            }
            results.append(result)

            # 打印结果
            print(f"  耗时: {duration:.2f} 秒")
            print(f"  输入tokens: {int(total_input_tokens)}")
            print(f"  输出tokens: {total_output_tokens}")
            print(f"  总tokens: {int(total_tokens)}")
            print(f"  吞吐量: {throughput:.2f} tokens/秒")
            print(f"  平均延迟: {latency:.2f} 秒/请求")
            print(f"  总tokens/秒: {total_tokens/duration:.2f}")

        # 汇总结果
        print("\n" + "=" * 60)
        print("基准测试汇总")
        print("=" * 60)
        print(f"\n{'批次大小':<10} {'吞吐量(tok/s)':<15} {'延迟(s)':<12} {'总tok/s':<12}")
        print("-" * 60)
        for r in results:
            print(f"{r['batch_size']:<10} {r['throughput']:<15.2f} {r['latency']:<12.2f} {r['tokens_per_second']:<12.2f}")

        # 最佳配置
        best_throughput = max(results, key=lambda x: x['throughput'])
        print(f"\n最佳吞吐量配置: 批次大小={best_throughput['batch_size']}, 吞吐量={best_throughput['throughput']:.2f} tokens/秒")

        # 保存结果
        output_file = f"benchmark_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'model': model_path,
                    'num_prompts': num_prompts,
                    'prompt_length': prompt_length,
                    'output_length': output_length,
                    'tensor_parallel_size': 8
                },
                'results': results
            }, f, indent=2)

        print(f"\n结果已保存到: {output_file}")
        print("=" * 60)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="vLLM性能基准测试")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/models/deepseek-v3",
        help="模型路径"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="测试样本数"
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=128,
        help="输入prompt长度（tokens）"
    )
    parser.add_argument(
        "--output-length",
        type=int,
        default=256,
        help="输出长度（tokens）"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8,16,32",
        help="测试的批次大小，逗号分隔"
    )

    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(',')]

    run_benchmark(
        model_path=args.model_path,
        num_prompts=args.num_prompts,
        prompt_length=args.prompt_length,
        output_length=args.output_length,
        batch_sizes=batch_sizes
    )

if __name__ == "__main__":
    main()
