#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Truncate a JSONL file based on answer consistency.

Each line in the input file has two fields: "question" and "answer".
Identical questions always appear in contiguous blocks.

For every question block, we keep reading answers until the same
answer appears `threshold` times in a row.  As soon as this happens,
we write the line where the threshold is reached to the output file
and ignore the rest of the block.  If the threshold is never reached,
the last line of the block is written instead.

The output file therefore contains at most one line per distinct question.
"""

import argparse
import json

def process_file(in_path: str, out_path: str, threshold: int = 3) -> None:
    """
    Read the input JSONL file line-by-line and write the truncated
    lines to `out_path`.
    """
    # Prepare output file handle
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        current_q        = None   # The question we are currently processing
        last_answer      = None   # The most recent answer seen
        repeat_count     = 0      # How many times `last_answer` has appeared consecutively
        block_saved      = False  # Whether we've already written a line for this question
        last_line_cache  = None   # Fallback: the last line of the current block

        for raw_line in fin:
            data = json.loads(raw_line)

            # If we encounter a new question, flush the previous block if needed
            if current_q is not None and data["question"] != current_q:
                if not block_saved and last_line_cache is not None:
                    fout.write(json.dumps(last_line_cache, ensure_ascii=False) + "\n")
                # Reset state for the new block
                current_q       = data["question"]
                last_answer     = data["answer"]
                repeat_count    = 1
                block_saved     = False
                last_line_cache = data
                continue

            # First line of the file OR still inside the same block
            if current_q is None:
                current_q = data["question"]

            # Update repetition counters
            if data["answer"] == last_answer:
                repeat_count += 1
            else:
                last_answer  = data["answer"]
                repeat_count = 1

            last_line_cache = data  # Always keep the latest line in case we need it

            # If we hit the threshold and haven't saved yet, save now
            if not block_saved and repeat_count >= threshold:
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                block_saved = True  # Ignore the remainder of this block

        # End-of-file: handle the very last block
        if current_q is not None and not block_saved and last_line_cache is not None:
            fout.write(json.dumps(last_line_cache, ensure_ascii=False) + "\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="Truncate JSONL by answer repetition.")
    parser.add_argument("--input",  "-i", required=True, help="Path to the input JSONL file")
    parser.add_argument("--output", "-o", required=True, help="Path for the output JSONL file")
    parser.add_argument("--threshold", "-t", type=int, default=3,
                        help="Number of consecutive identical answers needed to truncate (default: 3)")
    args = parser.parse_args()

    process_file(args.input, args.output, args.threshold)

if __name__ == "__main__":
    main()

