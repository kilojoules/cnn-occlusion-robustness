#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- 🚀 Starting Full Experiment Pipeline ---"

# --- Configuration ---
CONFIG_FILE="configs/eval/matrix.yaml"
RESULTS_DIR="results"
MODELS_DIR="$RESULTS_DIR/models"
EVAL_DIR="$RESULTS_DIR/eval"

# --- Read experiment grid by piping the config into yq (CORRECTED SYNTAX) ---
TRAIN_CONDITIONS=$(cat "$CONFIG_FILE" | yq -r '.training_conditions[]')
TEST_CONDITIONS=$(cat "$CONFIG_FILE" | yq -r '.test_conditions[]')
TEST_DATA_DIR=$(cat "$CONFIG_FILE" | yq -r '.test_data_dir')

echo "Found $(echo "$TRAIN_CONDITIONS" | wc -w | tr -d ' ') training conditions."
echo "Found $(echo "$TEST_CONDITIONS" | wc -w | tr -d ' ') test conditions."

# --- 1. Training Phase (CORRECTED) ---
echo "--- 🧑‍🍳 Phase 1: Training all models using config: $CONFIG_FILE ---"
for train_effect in $TRAIN_CONDITIONS; do
    echo "Training model for effect: $train_effect"
    MODEL_PATH="$MODELS_DIR/${train_effect}_model.pth"

    if [ -f "$MODEL_PATH" ]; then
        echo "Model $MODEL_PATH already exists. Skipping training."
    else
        # This is the NEW, correct way to call train.py
        python src/cnn_occlusion_robustness/train.py \
            --config "$CONFIG_FILE" \
            --train-effect "$train_effect" \
            --save-path "$MODEL_PATH"
    fi
done

# --- 2. Evaluation Phase ---
# This part can remain the same, but we use the TEST_DATA_DIR from the config.
echo "--- 🧐 Phase 2: Evaluating all models against all test conditions ---"
for train_effect in $TRAIN_CONDITIONS; do
    for test_effect in $TEST_CONDITIONS; do
        echo "Evaluating model '$train_effect' on test condition '$test_effect'"
        MODEL_PATH="$MODELS_DIR/${train_effect}_model.pth"
        OUTPUT_PATH="$EVAL_DIR/${train_effect}_on_${test_effect}.json"

        if [ -f "$OUTPUT_PATH" ]; then
            echo "Evaluation result $OUTPUT_PATH already exists. Skipping."
        else
            python src/cnn_occlusion_robustness/eval.py \
                --config "$CONFIG_FILE" \
                --model-path "$MODEL_PATH" \
                --data-dir "$TEST_DATA_DIR" \
                --test-effect "$test_effect" \
                --output-path "$OUTPUT_PATH"
        fi
    done
done

# --- 3. Build Evaluation Matrix ---
echo "--- 📊 Phase 3: Building the evaluation matrix ---"
python src/cnn_occlusion_robustness/build_eval_matrix.py \
    --eval-dir "$EVAL_DIR" \
    --output-dir "$RESULTS_DIR"

# --- 4. Run Final Analysis ---
echo "--- 📈 Phase 4: Generating final analysis report and figures ---"
python src/cnn_occlusion_robustness/analysis/advanced_analysis_report.py \
    --results-dir "$RESULTS_DIR" \
    --output-dir "analysis_output"

echo "--- ✅ Pipeline Complete! Check analysis_output/ for results. ---"
