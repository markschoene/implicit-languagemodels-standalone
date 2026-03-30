import argparse
import os

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import TransformersModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.utils.imports import is_accelerate_available

import implicit_llm  # to register the huggingface models

if is_accelerate_available():
    from datetime import timedelta

    from accelerate import Accelerator, InitProcessGroupKwargs

    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=3000))])
else:
    accelerator = None


def main():
    parser = argparse.ArgumentParser(description="Run LightEval benchmark.")
    parser.add_argument("--model_name", type=str, required=True, help="Path or name of the model.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b", help="Tokenizer name or path.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="lighteval|lambada:openai|0|0,leaderboard|hellaswag|0|0,lighteval|piqa|0|0",
        help="Tasks to evaluate, comma-separated.",
    )

    args = parser.parse_args()

    evaluation_tracker = EvaluationTracker(
        output_dir="./results",
        save_details=True,
        push_to_hub=False,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        max_samples=args.max_samples,
    )

    model_config = TransformersModelConfig(
        model_name=args.model_name,
        tokenizer=args.tokenizer,
        batch_size=args.batch_size,
    )

    pipeline = Pipeline(
        tasks=args.tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    )

    pipeline.evaluate()
    pipeline.show_results()


if __name__ == "__main__":
    main()
