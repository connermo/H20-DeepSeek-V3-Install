#!/usr/bin/env python3
"""
vLLM Deepseek V3 推理测试脚本
支持两种模式：
1. 直接推理模式（使用LLM类）
2. API模式（调用OpenAI兼容API）
"""

import argparse
import time
from typing import List

def test_direct_inference(model_path: str, prompts: List[str]):
    """直接推理模式测试"""
    print("=" * 60)
    print("直接推理模式测试")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams

        print(f"\n加载模型: {model_path}")
        print("这可能需要几分钟时间...\n")

        # 初始化模型（Deepseek V3使用FP8）
        llm = LLM(
            model=model_path,
            tensor_parallel_size=8,
            dtype="auto",  # 自动检测FP8
            kv_cache_dtype="fp8",  # FP8 KV缓存
            max_model_len=8192,
            gpu_memory_utilization=0.95,
            trust_remote_code=True
        )

        print("✓ 模型加载成功\n")

        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )

        # 推理
        print("开始推理...\n")
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        duration = time.time() - start_time

        # 打印结果
        for i, output in enumerate(outputs):
            print(f"{'=' * 60}")
            print(f"Prompt {i+1}:")
            print(f"{'-' * 60}")
            print(output.prompt)
            print(f"\n{'Generated:'}")
            print(f"{'-' * 60}")
            print(output.outputs[0].text)
            print()

        # 统计信息
        total_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        print(f"{'=' * 60}")
        print(f"性能统计:")
        print(f"  总耗时: {duration:.2f} 秒")
        print(f"  总token数: {total_tokens}")
        print(f"  吞吐量: {total_tokens/duration:.2f} tokens/秒")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def test_api_inference(base_url: str, prompts: List[str]):
    """API模式测试"""
    print("=" * 60)
    print("API模式测试")
    print("=" * 60)

    try:
        from openai import OpenAI

        print(f"\n连接到API服务器: {base_url}\n")

        # 初始化客户端
        client = OpenAI(
            api_key="EMPTY",
            base_url=base_url
        )

        # 测试每个prompt
        for i, prompt in enumerate(prompts):
            print(f"{'=' * 60}")
            print(f"Prompt {i+1}:")
            print(f"{'-' * 60}")
            print(prompt)
            print(f"\n{'Generated:'}")
            print(f"{'-' * 60}")

            start_time = time.time()

            # 调用API
            response = client.chat.completions.create(
                model="deepseek-v3",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=512
            )

            duration = time.time() - start_time

            # 打印结果
            generated_text = response.choices[0].message.content
            print(generated_text)
            print(f"\n耗时: {duration:.2f} 秒")
            print()

        print(f"{'=' * 60}")
        print("API测试完成")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"错误: {e}")
        print("\n提示: 请确保vLLM API服务器正在运行")
        print("启动命令: ./start_inference.sh")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="vLLM Deepseek V3 推理测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "api"],
        default="direct",
        help="测试模式: direct (直接推理) 或 api (API调用)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/models/deepseek-v3",
        help="模型路径 (direct模式使用)"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="API服务器地址 (api模式使用)"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="自定义测试prompt"
    )

    args = parser.parse_args()

    # 测试prompts
    if args.custom_prompt:
        prompts = [args.custom_prompt]
    else:
        prompts = [
            "请解释什么是大语言模型，并说明其主要应用场景。",
            "用Python写一个快速排序算法，并添加详细注释。",
            "介绍一下分布式训练中的张量并行技术。"
        ]

    # 运行测试
    if args.mode == "direct":
        test_direct_inference(args.model_path, prompts)
    else:
        test_api_inference(args.api_url, prompts)

if __name__ == "__main__":
    main()
