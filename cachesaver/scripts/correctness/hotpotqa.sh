#!/bin/bash

# Define benchmarks
benchmarks=("hotpotqa")

# Define methods

# methods=("reflexion_react")
methods=("reflect_prev_k")
# methods=("reflect_summary")
# Define models
provider="openai"
model="gpt-4.1-nano"

# Define number of retrials
split="mini"
trials=10

# Delete caches if they exist
for method in "${methods[@]}"; do
    FOLDER="caches/correctness/hotpotqa/${method}"
    if [ -d "$FOLDER" ]; then
        rm -rf "$FOLDER"
    fi
done

for benchmark in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
            echo "Running $benchmark with $method"
            python3 "scripts/correctness/${benchmark}.py" \
                --provider "$provider" \
                --model "$model" \
                --batch_size 25 \
                --trials $trials \
                --timeout 2.0 \
                --temperature 0.7 \
                --max_completion_tokens 300 \
                --top_p 1.0 \
                --dataset_path "datasets/dataset_${benchmark}.csv.gz" \
                --split "$split" \
                --method "$method" \
                --conf_path "scripts/correctness/${benchmark}.yaml" \
                --correctness 0 \
                --value_cache
    done
done