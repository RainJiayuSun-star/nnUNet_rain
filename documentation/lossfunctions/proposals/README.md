# Loss-function integration

## Current decision: binary whole-metastasis training

The approved implementation supersedes the earlier three-class/core-and-edema CC
design. Train one foreground mask containing enhancing and non-enhancing/necrotic
metastasis, excluding edema. The network has background and whole-metastasis outputs.

- `nnUNetTrainerDiceCEBaseline`: global Dice+CE, no additive Dice smoothing.
- `nnUNetTrainerCCDiceCE`: binary component-wise Dice+CE.
- `nnUNetTrainerGlobalCCDiceCE`: global plus component-wise Dice+CE, weighted 1:1.
- All use native `nnUNetTrainer` optimization and augmentation, not configurable
  YAML: SGD, LR 0.01, weight decay 3e-5, momentum 0.99, Nesterov, polynomial decay,
  and 1,000 epochs.
- Apply the complete objective at every active deep-supervision scale, recomputing
  component geometry; preserve normalized weights and the DDP safeguard.
- Preserve Dataset003. Convert combined labels 1→1 and {0,2}→0 into a distinct
  dataset ID, audit provenance where same-space original annotations are available,
  and plan/preprocess afresh. Transfer splits only after case/patient checks.
- GlobalCC is the primary candidate. Compare all three using the same data, folds,
  plans, and budgets. Use one fixed lesion-matching protocol for recall, precision,
  false positives per scan, count error, and segmentation overlap.

See [implementation, formulas, converter, verification, and one-GPU runbook](01_cc_dice_ce.md).
CPU checks do not establish CUDA/DDP acceptance. Full server conversion and long
training runs are separate execution steps, gated on CUDA acceptance and a real
training/validation smoke test with measured VRAM.

## Historical contour and Zhao proposals

The following research notes are preserved, **not approved implementations**.
Their three-class/core-and-edema examples predate the binary decision above;
they are not the CC contract and must be re-scoped before any implementation.
Neither the configurable optimizer nor three-class outputs should be inferred
from these notes for the current binary experiment.

## 5. Modified Contour Dice proposal

### Verified sources and limitation

[Paper transcription](../../../../../../lossFunctions/ModifiedContourDice/Modifiedcontourdice.md) and [PDF](../../../../../../lossFunctions/ModifiedContourDice/Modifiedcontourdice.pdf), PDF pages 6–7, Eqs. 3–5 and Sections 3.1–3.2, were visually checked. The modified formula scores the two contour-to-band intersections divided by the sum of contour sizes, with smoothing. Do not substitute Eq. 2's band-to-band Dice, which is a different formulation.

The text describes thresholded predictions, erosion, dilation, and XOR, computed slice-wise. Literal boolean thresholding in ordinary autograd disconnects the contour term from network probabilities. The provided text does not establish a usable gradient implementation. The proposal must address this explicitly; it cannot claim trainability merely because the final ratio resembles Dice.

### Recommended adaptation to specify

Use softmax foreground probabilities separately for core and edema, preserving the global three-class Dice + CE. Propose a differentiable 3D morphological surrogate: local minimum/maximum pooling, soft contour `P - erode(P)`, soft band `dilate(P) - erode(P)`, and product-based intersections in the Eq. 4 structure. Specify the exact kernels: the paper's band figure uses different erosion and dilation kernels, so one generic radius must not be silently described as identical to that figure.

Mark soft morphology, 3D processing, multiclass reduction, highest-resolution-only auxiliary supervision, and retaining CE as project adaptations. The published combination is Dice plus contour loss, with contour weight 0.5 in its experiments; 0.5 is a candidate starting point, not validated for our adapted objective. Include a Dice-plus-contour ablation if reproducing the published composition matters.

Define foreground-class weights and reduction order, smoothing, padding at crop faces, and treatment of empty/full masks. A constant probability field can have no morphological contour, so the global term remains necessary. State whether kernels are measured in voxels or physical distance; do not report a voxel kernel as a millimeter tolerance. Keep hard contour evaluation independent of the soft training surrogate.

### Acceptance criteria

Nonzero finite auxiliary gradients on deliberately misplaced soft boundaries; numerical gradient checks away from pooling ties; correct binary-limit geometry; explicit empty-target and all-foreground behavior; crop-edge padding tests; core/edema channel separation; and behavior on thin, touching, shifted, and single-voxel structures. Include a test showing that the literal hard-threshold path does not supply the intended training gradient.

## 6. Modified Zhao proposal

### Verified sources and missing definitions

[Paper transcription](../../../../../../lossFunctions/modifiedZhao/SpinalDiseaseImageSegmentationTechIntegratingU-ResNetandShape-AwareAttention.md) and [PDF](../../../../../../lossFunctions/modifiedZhao/SpinalDiseaseImageSegmentationTechIntegratingU-ResNetandShape-AwareAttention.pdf), PDF pages 7–8, Eqs. 11–16, were visually checked. The specified terms are weighted binary cross-entropy, distance-weighted absolute probability error, an occupancy-dependent regional coefficient, and absolute volume error with coefficient 0.1. These loss equations do not define a new Dice score, despite references to Dice elsewhere in the paper's prose.

The loss section does not fully specify the density-weight function `omega(s)`, the construction/direction/normalization of distance weight `d(s)`, or a reproducible rule for boundary coefficient `beta`. It also does not fully specify spatial reductions and how volume units are reconciled. Do not invent these details and attribute them to the paper. The earlier spatial attention factor `beta(s)` is not automatically the loss coefficient `beta`.

### Recommended proposal structure

First transcribe Eqs. 11–16 with a source-versus-assumption ledger. Then define a named Zhao-inspired adaptation for core and edema, keeping the existing architecture. Architecture and shape-attention changes are outside this loss-integration proposal.

Resolve the following mathematically before marking that proposal ready to implement:

- Whether the main experiment retains Dice + CE and adds Zhao-inspired boundary/volume terms, or replaces the objective with all Zhao terms. Recommend the additive variant first, with a separate full-composition experiment to avoid silently double-counting CE.
- For the full composition, how one-versus-rest binary cross-entropy is computed from three-class softmax probabilities, how density weights are bounded, and how classes/voxels/samples are averaged.
- A nonnegative boundary weight with an explicit distance transform and units. Raw distance to the boundary is larger far from the boundary, so it cannot be described as emphasizing near-boundary voxels without an additional transformation. A decaying distance weight is a possible project choice, not a supplied formula.
- The value/range of the regional modulation parameter and an explicit bounded rule or fixed setting for `beta`. Do not assume `beta = 1 - alpha` from the printed equations.
- A volume term with explicit units and normalization. Recommend patch occupancy error as an initial scale-stable adaptation; document that it differs from the paper's unnormalized absolute volume/count error, and that coefficient 0.1 cannot automatically be transferred after rescaling.
- Handling absent classes, full masks, patches cut through lesions, and augmentation. Target geometry and target volume must refer to the same patch as the prediction; a cropped patch cannot be compared with a full-scan target volume.

### Acceptance criteria

Hand-computable regional/boundary/volume examples; bounded weights; finite empty-target behavior; gradients in the expected direction; demonstrated behavior under patch-size and voxel-spacing changes; and independent logging of each term to expose domination by the volume penalty. This proposal needs more mathematical specification than the other two before implementation.
