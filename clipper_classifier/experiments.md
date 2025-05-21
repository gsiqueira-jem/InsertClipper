# CLIP Fine-tuning Experiments for Open-Set Classification

## Dataset Structure
- Training data: `./dataset/train/clip/`
- Validation data: `./dataset/val/clip/`
- Test data: `./dataset/test/clip/`

### Class Mapping
- Door (folders 1-6) → class 0
- Window (folders 7-10) → class 1
- Furniture (folders 11-27) → class 2
- Unknown (folders 28-30) → class 3

## Base Configuration
```python
# Model
- CLIP backbone: ViT-B/32
- Frozen CLIP parameters
- Trainable classifier head

# Training
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Number of workers: 4

# Data Augmentation
- Resize: 224x224
- Normalization: CLIP default values
```

## Experiments

### v0.1 - Base Model 
- Description: Basic setup with frozen CLIP and trainable classifier
- Configuration: Base Configuration

### v0.2 - Base Model + Early Stop
- Description: Basic setup + Early Stop
- Configuration: Base Configuration but no Early Stop
```python
Early stopping patience: 5 epochs
Save checkpoint every: 10 epochs
Minimum improvement threshold: 0.5%
```

## Metrics to Track
1. Training Loss
2. Training Accuracy
3. Validation Accuracy
4. Time per epoch
5. Best validation accuracy
6. Number of epochs until convergence

## Results Template
```
Experiment: [Name]
Date: [YYYY-MM-DD]
Duration: [HH:MM:SS]

Best Validation Accuracy: XX.XX%
Epochs to Convergence: XX
Final Training Loss: X.XXXX
Final Training Accuracy: XX.XX%

Notes:
- [Any observations]
- [Issues encountered]
- [Potential improvements]
```

## Next Steps
1. Run base model experiment
2. Analyze results and identify bottlenecks
3. Implement variations based on initial findings
4. Compare results across experiments
5. Select best performing configuration
6. Fine-tune selected configuration 