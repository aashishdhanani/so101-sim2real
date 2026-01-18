# Training Structure Explanation

## Overview

Your training setup uses **PPO (Proximal Policy Optimization)** with the following structure:

## Key Concepts

### **Iterations** vs **Epochs** vs **Mini-batches**

1. **Iteration** = One complete cycle (collect + learn):
   - **Each iteration** = Collect experience → Learn from that experience
   - Your config: `max_iterations = 1500` means **1500 iterations total**
   - **Each iteration is a full collect and learn cycle**

2. **Epoch** (within each iteration):
   - After collecting experience, you train on the **same collected data** multiple times
   - Your config: `num_learning_epochs = 5` means **5 epochs per iteration**
   - Each epoch processes the same 24 steps, but shuffled differently

3. **Mini-batch** (within each epoch):
   - Each epoch divides the collected data into smaller chunks
   - Your config: `num_mini_batches = 4` means **4 mini-batches per epoch**
   - Each mini-batch triggers one network update

### Training Flow

```
For each iteration (1 to 1500):
  ┌─────────────────────────────────────┐
  │ 1. COLLECT PHASE                    │
  │    - Run policy in env for 24 steps │
  │    - Store: states, actions,        │
  │      rewards, values, log_probs     │
  └─────────────────────────────────────┘
           ↓
  ┌─────────────────────────────────────┐
  │ 2. LEARN PHASE (5 epochs)           │
  │    For epoch 1 to 5:                 │
  │      - Shuffle the 24 collected steps│
  │      - Split into 4 mini-batches     │
  │        (6 steps per mini-batch)      │
  │      - For each mini-batch:          │
  │        • Compute advantages         │
  │        • Update actor (policy)      │
  │        • Update critic (value)      │
  └─────────────────────────────────────┘
```

### Data Flow Per Iteration

```
24 steps collected (1 iteration)
  ↓
Shuffled and split into 4 mini-batches (6 steps each)
  ↓
Epoch 1: Update on all 4 mini-batches (using same 24 steps)
Epoch 2: Update on all 4 mini-batches (using same 24 steps, shuffled again)
Epoch 3: Update on all 4 mini-batches (using same 24 steps, shuffled again)
Epoch 4: Update on all 4 mini-batches (using same 24 steps, shuffled again)
Epoch 5: Update on all 4 mini-batches (using same 24 steps, shuffled again)
  ↓
Total: 5 epochs × 4 mini-batches = 20 network updates per iteration
```

## Your Current Configuration

- **Max iterations**: 1500
- **Steps per env per iteration**: 24
- **Learning epochs per iteration**: 5
- **Mini-batches per epoch**: 4
- **Episode length**: 20 seconds
- **Number of environments**: 1 (for simplicity)

## What This Means

**Per Iteration:**
- **Collect**: 24 environment steps
- **Learn**: 5 epochs × 4 mini-batches = **20 network updates**
- **Total**: Each iteration = 24 steps collected + 20 network updates

**Total Training:**
- **Total environment steps**: 1 env × 24 steps × 1500 iterations = **36,000 steps**
- **Total network updates**: 1500 iterations × 5 epochs × 4 mini-batches = **30,000 updates**
- **Time per iteration**: Depends on simulation speed, but roughly:
  - Collecting 24 steps: ~few seconds
  - Training (5 epochs × 4 batches): ~few seconds
  - Total: ~5-10 seconds per iteration (varies)
- **Total training time**: 1500 iterations × ~5-10s = **~2-4 hours** (rough estimate)

**Key Point**: Each iteration is a complete cycle - you collect fresh experience, then learn from it multiple times (5 epochs) before moving to the next iteration.

## What to Expect When Running

### Visual (if not headless):
- **1 robot arm** in the scene
- Robot will start with **random/jittery movements** (exploration)
- Gradually, movements should become **more purposeful**
- Robot should learn to:
  1. **Reach** toward the object
  2. **Grasp** the object (close gripper)
  3. **Lift** the object

### Console Output:
- Training metrics printed each iteration
- Episode rewards (should increase over time)
- Loss values (should decrease)
- Checkpoints saved every 50 iterations

### WandB Dashboard:
- **Episode rewards**: Should trend upward
- **Policy loss**: Should decrease (become more negative initially, then stabilize)
- **Value loss**: Should decrease
- **Episode length**: Should increase as robot learns to stay alive longer
- **Success rate**: If you track it, should increase

## Important Metrics to Watch

1. **Mean Episode Reward**: Should increase over iterations
2. **Episode Length**: Should increase (robot staying alive longer)
3. **Policy Loss**: Negative values are good (maximizing reward)
4. **Value Loss**: Should decrease (critic learning better predictions)
5. **Explained Variance**: Should increase (critic explaining rewards better)

## Early Training Behavior

- **First 100-200 iterations**: 
  - Random movements
  - Low rewards
  - High losses
  - Episodes end quickly (timeout or failure)

- **200-500 iterations**:
  - Some purposeful movements
  - Rewards starting to increase
  - Robot might reach object occasionally

- **500-1000 iterations**:
  - More consistent reaching
  - Occasional successful grasps
  - Rewards steadily increasing

- **1000-1500 iterations**:
  - Should see successful lift attempts
  - More stable policy
  - Rewards plateauing or slowly increasing

## Troubleshooting

If training isn't working:
1. **Rewards not increasing**: Check reward weights, might need tuning
2. **Robot not moving**: Check action space, joint limits
3. **Crashes**: Check for NaN values, reduce learning rate
4. **Too slow**: Reduce num_envs or num_steps_per_env for testing

## Next Steps After Training

1. **Evaluate**: Test the trained policy
2. **Analyze**: Check WandB plots to see what worked
3. **Tune**: Adjust hyperparameters based on results
4. **Scale up**: Increase num_envs for faster training

