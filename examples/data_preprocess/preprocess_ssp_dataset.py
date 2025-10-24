# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import tempfile

import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <search> query </search> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)

def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question", "")

    # Build prompt structure
    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    # Extract ground truth from reward_model or fallback to golden_answers
    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    # Process data source
    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))

    # Build tools kwargs structure
    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    # Build complete extra_info structure
    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    splits = []
    if args.train:
        splits.append("train")
    if args.test:
        splits.append("test")
    if not splits:
        logger.warning("No --train or --test specified, nothing to process.")
        return

    # Download and process files using temporary directory
    with tempfile.TemporaryDirectory():
        for split in splits:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            try:
                # Download Parquet file from HuggingFace
                # logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
                # local_parquet_filepath = hf_hub_download(
                #     repo_id=args.hf_repo_id,
                #     filename=parquet_filename,
                #     repo_type="dataset",
                #     local_dir=tmp_download_dir,
                #     local_dir_use_symlinks=False,
                # )

                if split == "train":
                    local_parquet_filepath = args.train
                elif split == "test":
                    local_parquet_filepath = args.test
                else:
                    logger.error(f"Unsupported split: {split}")
                    continue

                # Load and process Parquet file
                df_raw = pd.read_parquet(local_parquet_filepath)
                logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

                def apply_process_row(row, split_name=split):
                    return process_single_row(row, current_split_name=split_name, row_index=row.name)

                df_processed = df_raw.apply(apply_process_row, axis=1)

                # Save processed DataFrame
                output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                df_processed.to_parquet(output_file_path, index=False)
                logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
                processed_files.append(output_file_path)

            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument("--train", default="")
    parser.add_argument("--test", default="")
    parser.add_argument(
        "--local_dir",
        default="/",
        help="Local directory to save the processed Parquet files.",
    )

    args = parser.parse_args()

    # System and user content configuration
    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()
