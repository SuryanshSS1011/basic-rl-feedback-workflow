#!/bin/bash
# SecureCodeRL: Complete Enhanced Training Pipeline
# Usage:
#   ./run_both_experiments.sh              # Full run (500 episodes, ~7-9 hours)
#   ./run_both_experiments.sh --quick      # Quick test (100 episodes, ~1.5 hours)
#   ./run_both_experiments.sh --medium     # Medium run (200 episodes, ~3 hours)

set -e

# ==================== PARSE ARGUMENTS ====================
MODE="full"
if [ "$1" == "--quick" ] || [ "$1" == "-q" ]; then
    MODE="quick"
    echo "🚀 QUICK TEST MODE: 100 episodes (~1.5 hours)"
elif [ "$1" == "--medium" ] || [ "$1" == "-m" ]; then
    MODE="medium"
    echo "📊 MEDIUM MODE: 200 episodes (~3 hours) - Good for submission"
fi

# ==================== CONFIGURATION ====================
if [ "$MODE" == "quick" ]; then
    EPISODES=100
    NUM_EVAL_SAMPLES=20
elif [ "$MODE" == "medium" ]; then
    EPISODES=200
    NUM_EVAL_SAMPLES=50
else
    EPISODES=500
    NUM_EVAL_SAMPLES=100
fi

BATCH_SIZE=2
LEARNING_RATE=1e-6

SFT_CHECKPOINT="./checkpoints/sft_stdin/best"
PPO_CHECKPOINT="./checkpoints/ppo/ppo/best"
PROMPTS_FILE="./data/prompts/ppo_prompts_with_tests.json"

OUTPUT_CONTINUE="./checkpoints/ppo_enhanced_continue"
OUTPUT_FRESH="./checkpoints/ppo_enhanced_fresh"
RESULTS_DIR="./results"

# ==================== SETUP ====================
mkdir -p "$RESULTS_DIR/baseline" "$RESULTS_DIR/final_comparison"
LOG_FILE="$RESULTS_DIR/experiment_log.txt"

log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "=========================================="
log "SecureCodeRL Pipeline Started"
if [ "$MODE" == "quick" ]; then
    log "MODE: Quick Test (100 episodes, ~1.5 hours)"
elif [ "$MODE" == "medium" ]; then
    log "MODE: Medium (200 episodes, ~3 hours)"
else
    log "MODE: Full Run (500 episodes, ~7-9 hours)"
fi
log "Episodes: $EPISODES | Batch: $BATCH_SIZE"
log "=========================================="

# ==================== STEP 0: PREPARE DATA ====================
log "[0/5] Preparing data..."
if [ -f "$PROMPTS_FILE" ]; then
    log "  Data exists, skipping"
else
    python prepare_ppo_data.py --output "$PROMPTS_FILE"
fi

# ==================== STEP 1: BASELINE EVAL ====================
log "[1/5] Baseline evaluation..."
python evaluate_dual_metrics.py \
    --checkpoints "$SFT_CHECKPOINT" "$PPO_CHECKPOINT" \
    --prompts_file "$PROMPTS_FILE" \
    --num_samples 50 \
    --output_dir "$RESULTS_DIR/baseline"
log "  Done -> $RESULTS_DIR/baseline/"

# ==================== STEP 2: TRAIN OPTION A ====================
log "[2/5] Training Option A (continue from PPO)..."
python train_ppo.py \
    --sft_checkpoint "$SFT_CHECKPOINT" \
    --ppo_checkpoint "$PPO_CHECKPOINT" \
    --resume \
    --prompts_file "$PROMPTS_FILE" \
    --use_bandit \
    --episodes $EPISODES \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir "$OUTPUT_CONTINUE"
log "  Done -> $OUTPUT_CONTINUE"

# ==================== CUDA CACHE CLEANUP ====================
log "Clearing CUDA cache before Option B..."
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"
sleep 5

# ==================== STEP 3: TRAIN OPTION B ====================
log "[3/5] Training Option B (fresh from SFT)..."
python train_ppo.py \
    --sft_checkpoint "$SFT_CHECKPOINT" \
    --prompts_file "$PROMPTS_FILE" \
    --use_bandit \
    --episodes $EPISODES \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --output_dir "$OUTPUT_FRESH"
log "  Done -> $OUTPUT_FRESH"

# ==================== CUDA CACHE CLEANUP ====================
log "Clearing CUDA cache before final evaluation..."
python -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"
sleep 5

# ==================== STEP 4: FINAL EVAL ====================
log "[4/5] Final comparative evaluation..."
python evaluate_dual_metrics.py \
    --checkpoints "$SFT_CHECKPOINT" "$PPO_CHECKPOINT" "$OUTPUT_CONTINUE/ppo/best" "$OUTPUT_FRESH/ppo/best" \
    --prompts_file "$PROMPTS_FILE" \
    --num_samples $NUM_EVAL_SAMPLES \
    --output_dir "$RESULTS_DIR/final_comparison"
log "  Done -> $RESULTS_DIR/final_comparison/"

# ==================== STEP 5: SUMMARY ====================
log "[5/5] Generating summary..."
cat > "$RESULTS_DIR/EXPERIMENT_SUMMARY.md" << EOF
# SecureCodeRL Experiment Summary

## Configuration
- Episodes: $EPISODES
- Batch size: $BATCH_SIZE
- Learning rate: $LEARNING_RATE
- Eval samples: $NUM_EVAL_SAMPLES

## Models
1. **SFT** - Baseline (before RL)
2. **PPO-simple** - Simple metrics (syntax + regex)
3. **PPO-continue** - Enhanced metrics (from PPO)
4. **PPO-fresh** - Enhanced metrics (from SFT)

## Results
- Baseline: \`$RESULTS_DIR/baseline/\`
- Final: \`$RESULTS_DIR/final_comparison/\`

## Completed
$(date)
EOF

# ==================== DONE ====================
log "=========================================="
log "ALL EXPERIMENTS COMPLETE!"
log "=========================================="
log "Results: $RESULTS_DIR/"
log "Models:"
log "  - Continue: $OUTPUT_CONTINUE/ppo/best"
log "  - Fresh: $OUTPUT_FRESH/ppo/best"
