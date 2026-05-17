## Experiment Report: Zero-Order Fine-Tuning of ResNet18

The goal of this experiment was to improve the zero-order fine-tuning baseline for a pretrained ResNet18 under a limited training budget. The baseline result was approximately **1.22% top-1 validation accuracy**, while the final submitted configuration reached **1.50% top-1 validation accuracy**.

The final configuration used:

```text
batch_size = 64
n_batches = 128
learning_rate = 3e-2
epsilon = 1e-4
perturbation = Rademacher / SPSA-style
momentum = 0.9
augmentation = none
head initialization = Xavier
```

The tuned layers were:

```text
fc.weight
fc.bias
layer4.1.bn2.weight
layer4.1.bn2.bias
```

The final validation result was:

```json
{
  "val_accuracy_top1_imagenet_head": 0.0037,
  "val_accuracy_top1_init_head": 0.0122,
  "val_accuracy_top1_finetuned": 0.015,
  "n_batches": 128,
  "batch_size": 64
}
```

Compared with the baseline value of **0.0122**, the final result improved the score to **0.0150** (relative improvement of roughly **23%**).


## Main Ideas Behind the Experiments

The main difficulty of zero-order optimization is that the gradient estimate is extremely noisy. Because the number of allowed optimization steps is limited, the solution had to focus on reducing noise and making each step informative.

The first idea was to replace the naive per-parameter finite-difference logic with a more SPSA-like simultaneous perturbation strategy. Instead of perturbing each parameter tensor independently and spending separate loss evaluations for each one, all selected parameters are perturbed together using random Rademacher directions. This keeps the update close to the classic two-point central-difference baseline, but makes each zero-order step more efficient.

The second idea was to add momentum to the zero-order update. Momentum smoothes the sequence of estimated directions and makes the optimizer less sensitive to random perturbation noise

The third idea was to tune a small but more expressive subset of parameters. Tuning only the classifier head is safe, but may be too restrictive. The final method tunes the classifier head together with the affine parameters of the last BatchNorm layer:

```text
layer4.1.bn2.weight
layer4.1.bn2.bias
```

This acts like a lightweight adaptation mechanism but keeps the number of optimized parameters relatively small

The fourth idea was to use class-balanced training batches

Finally, BatchNorm layers are forced into evaluation mode during the optimizer step.


## Experimental Findings

observations:

```text
baseline_64_128_s42                         -> 1.22%
crop_color_bn_spsa_balanced_mmnt_xav_64_128 -> 1.48%
eps3e4_bn_spsa_balanced_mmnt_xav_64_128     -> 1.48%
eps1e4_bn_spsa_balanced_mmnt_xav_64_128     -> 1.50%
lr1e2_bn_spsa_balanced_mmnt_xav_64_128      -> 1.34%
lr3e3_bn_spsa_balanced_mmnt_xav_64_128      -> 1.22%
lr1e3_bn_spsa_balanced_mmnt_xav_64_128      -> 1.20%
```


Interestingly, adding stronger augmentations did not improve the result. The best final version used no augmentation. This is also consistent with the nature of zero-order optimization: augmentations increase the variance of the loss estimate. 

---

## Final Method Summary

```text
1. SPSA-style simultaneous perturbation
2. Rademacher random directions
3. Momentum over noisy zero-order estimates
4. Normalized parameter updates
5. Class-balanced training batches
6. Tuning classifier head + final BatchNorm affine parameters
7. BatchNorm evaluation mode during zero-order loss estimation
8. No augmentation to keep the loss signal stable
```
