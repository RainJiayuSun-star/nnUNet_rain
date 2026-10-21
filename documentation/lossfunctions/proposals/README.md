# Plan for loss-function integration proposals

Status: the two-label CC-DiceCE integration is implemented and CPU-tested; CUDA acceptance remains pending. Contour and Zhao remain proposals. See [CC implementation and usage](01_cc_dice_ce.md).

Prepared on 2026-10-20 against the local nnUNet_rain checkout, the supplied paper transcriptions and PDFs, and the local Learning-to-Look-Closer repository. This file is the checked plan for writing the individual proposals listed below. Suggested names, configuration keys, and defaults are proposals, not existing APIs.

## Recorded decision: two-foreground-label CC integration

The user selected CC supervision for both tumor core (label 1) and edema (label 2), rather than the reference implementation's single foreground label. Background remains label 0: this is a three-output, mutually exclusive segmentation task, not a two-output network. This decision supersedes the earlier core-only CC and finest-resolution-only CC recommendations. The user subsequently approved implementation; the implemented formulas, safeguards, commands, and test results are recorded in [01_cc_dice_ce.md](01_cc_dice_ce.md).

- Integrate only the two requested trainer variants, `nnUNetTrainerCCDiceCE` and `nnUNetTrainerGlobalCCDiceCE`; Blob trainers and the proposed cascade are outside this integration's scope. Document these trainers as project multiclass adaptations, not unchanged binary reproductions.
- Compute core-versus-rest and edema-versus-rest CC Dice+CE separately, including separate GT component/Voronoi maps. Do not merge core and edema into one foreground mask.
- Use the identical component term in both trainers: `L_CC = 0.5 * (L_CC_core + L_CC_edema)`. Each class term averages components within a sample, then samples; absent-class behavior must be explicitly tested rather than silently dropping that class.
- At every active deep-supervision scale, use `L_CC` for CCDiceCE and `L_CC + L_global_3class` for GlobalCCDiceCE. Thus the complete objectives are `sum_s w_s L_CC(z_s, y_s)` and `sum_s w_s [L_CC(z_s, y_s) + L_global_3class(z_s, y_s)]`. Start with equal global/component coefficients of 1; retain identical normalized scale weights and recompute geometry at each scale.
- Global CE supervises background, core, and edema; global Dice excludes background and supervises core and edema. Keep the existing YAML-based training settings, with matched architecture, optimizer, sampling, splits, and training budget between the two trainers. The added global term is the only loss-design difference in this comparison.
- This choice provides supervision for both foreground classes; it does not establish that edema CC supervision improves tumor detection/counting. Edema components are not assumed to correspond one-to-one with tumors. Neither class-wise CC nor the added global loss explicitly enforces core–edema attachment, exact lesion counts, or precise boundaries. Lesion definition and detection/counting evaluation remain separate specifications.

## 1. Correct the task definition first

The intended task has two image inputs, FLAIR and T1post, and three mutually exclusive classes:

- 0: background.
- 1: tumor core, formed by combining original labels 1, 3, and 4 where present.
- 2: edema, retained from original label 2.

For 3D training, input images have shape `[B, 2, D, H, W]`, network logits have shape `[B, 3, D, H, W]`, and the integer target has shape `[B, 1, D, H, W]`. Use three-class softmax internally and argmax for the final label map. Deep supervision produces corresponding lists of logits and targets at different resolutions. There is no separate enhancing-tumor output after label combination. Whole tumor can be evaluated as the union `{1, 2}` without adding an output channel.

Evidence checked:

- [Label-combining script](../../../../../image_preprocessing/labeling/fastLabelCombine.py) maps original 1/3/4 to 1 and original 2 to 2. Its current top-level invocation processes only five subjects; its standard-segmentation fallback writes a different `tumorcore.seg` filename. Script availability does not establish complete cohort conversion.
- [Current Dataset001 formatter](../../../nnUnet_dataset/nnUnet_raw/Dataset001_UCSFbrainmets/reformat.java) now selects combined labels and emits `background: 0, tumor_core: 1, edema: 2`.
- [Dataset003 formatter](../../../nnUnet_dataset/nnUnet_raw/Dataset003_UCSFbrainmets_222/reformat.java) emits the same schema for the 222-patient cohort. Its [README](../../../nnUnet_dataset/nnUnet_raw/Dataset003_UCSFbrainmets_222/README.md) documents patient selection and source requirements.
- Two local source cases, UCSF 100101A and 100102A, were checked by reading the NIfTI voxel arrays. Both have combined values exactly `{0, 1, 2}` and match the original-label remapping voxel for voxel. This is a spot check, not a whole-cohort geometry or integrity audit.
- The saved [Dataset001 preprocessed metadata](../../../nnUnet_dataset/nnUNet_preprocessed/Dataset001_UCSFbrainmets/dataset.json) still describes the older overlapping WT/TC/ET targets and 461 training cases. It is not evidence for the intended new training schema.

The shared proposal must require verification of the actual training dataset JSON, all label values, image/label geometry, patient-level splits, and plan provenance before experiments. Select the intended dataset/cohort explicitly. Reprocess the combined-label dataset into a separate, coherent dataset version rather than mixing its labels with old preprocessed data or checkpoints. Do not infer that the deployed server has already been converted.

The new baseline is class-based Dice + CE. The root YAML's old `region_weights: [1, 1, 2]` describes WT/TC/ET and must not be repurposed as background/core/edema weights. The current trainer only applies that wrapper when `has_regions` is true; document an explicit class-mode configuration with region weights unset. Any new per-class auxiliary weights need a separately named, validated setting.

## 2. Proposed documents and writing order

1. `00_shared_contract_and_baseline.md`: data contract, baseline, trainer/configuration design, deep supervision, reduction rules, reproducibility, and common acceptance criteria.
2. [01_cc_dice_ce.md](01_cc_dice_ce.md): implemented CC objective, reference-code audit, exact one-versus-rest adapters for core and edema, matched CC-only/global-plus-CC trainers, component geometry, dependencies, and parity tests.
3. `02_modified_contour_dice.md`: published contour definitions, differentiability issue, explicitly proposed soft 3D adaptation, and boundary-specific checks.
4. `03_modified_zhao.md`: published regional/boundary/volume terms, missing definitions, and a fully specified proposed adaptation once the choices below are settled.
5. `04_validation_and_experiments.md`: synthetic checks, gradient and GPU checks, controlled ablations, full-volume metrics, and criteria for advancing to cross-validation.

Every individual proposal should contain: motivation; source and equation locations; exact published formula; differences between the paper and available code; proposed project formula; tensor shapes; reduction and empty-target behavior; intended code/configuration changes; validation criteria; experiment design; and unresolved choices. Label statements as source-defined, observed in code, or proposed. Do not describe a proposed adaptation as a faithful reproduction.

## 3. Shared integration design to specify

- Extend the existing [configurable trainer](../../../nnunetv2/training/nnUNetTrainer/variants/nnUNetTrainerMetsConfigurable.py), with loss implementations under `nnunetv2/training/loss/`. Keep loss mathematics in modules, not inside the trainer. Prefer a small loss-construction helper if selection logic becomes unwieldy.
- For CC, expose the two requested trainer names with the matched two-foreground-label objectives recorded above; the earlier suggested `cc_dice_ce_core` selector is superseded. Suggested selectors for the other proposals remain `modified_contour_dice_ce` and `modified_zhao`. Reject unknown selectors and invalid parameters rather than silently selecting baseline Dice + CE. Validate the three-class, non-region contract at initialization.
- Keep architecture, optimizer, augmentation, sampling, splits, and training budget fixed for the first loss comparisons. The existing AdamW setup is one possible project baseline; changing optimizer or adopting the paper's training recipe must be a separately identified experiment.
- For contour and Zhao proposals, retain baseline global Dice + CE at all active deep-supervision scales and initially apply the auxiliary term only at the highest-resolution output: `sum_s w_s L_global(z_s, y_s) + lambda L_aux(z_0, y_0)`, with lambda independent of `w_0`. CC is explicitly different: the selected complete CC-only or global-plus-CC objective runs at every active scale, following the recorded decision above.
- Preserve the existing lowest-scale DDP weight safeguard. Build auxiliary geometry from the target after spatial augmentation at its actual loss resolution. Do not interpolate component IDs, and do not reuse full-volume distances after a nonrigid or scaling transform without recomputing them.
- Specify float32 loss accumulation under AMP, device placement, and each reduction order. For instance terms, average components within each sample, then samples; do not unintentionally give patients with more lesions greater weight. A DDP test must cover unequal component counts and the intended sample weighting across ranks.
- Keep component labeling and distance transforms outside autograd. Prediction probabilities must stay connected to logits. CuPy/variable-component code should stay outside whole-loss compilation; test compiled numerical submodules separately.
- Initially reject ignore-label datasets for the new losses unless masking semantics are explicitly implemented and tested. Geometry must never treat ignored voxels as real background by accident.
- Save a resolved configuration, dataset/plan identity, code revision, loss parameters, and relevant library versions with each experiment. Inference must not depend on whichever YAML happens to be in the working directory.
- Log the global and auxiliary contributions separately, with names and signs that distinguish training loss from evaluation scores. A lower compound loss is not itself evidence of better Dice.

## 4. CC-DiceCE proposal

### Verified sources

- [Paper transcription](../../../../../../lossFunctions/ccDice/CCLoss_LearningtoLookCloser.md) and [PDF](../../../../../../lossFunctions/ccDice/CCLoss_LearningtoLookCloser.pdf), PDF pages 2–3, Eqs. 1–3 and Section 3.1.
- Local [instance losses](../../../../../../lossFunctions/ccDice/Learning-to-Look-Closer/nnunetv2/training/loss/instance_losses.py), [geometry helpers](../../../../../../lossFunctions/ccDice/Learning-to-Look-Closer/nnunetv2/utilities/connected_components.py), and [trainer variants](../../../../../../lossFunctions/ccDice/Learning-to-Look-Closer/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainerInstanceLosses.py).
- Reference checkout commit: `e5013bdb560eee544a94e1f068cb5cb4d003a51c`; checkout was clean when inspected. Preserve attribution and the supplied Apache-2.0 license notices for adapted code.

The paper partitions the lattice into nearest-GT-component Voronoi regions using 26-connected foreground components, then averages a base objective across components. It combines instance DiceCE and global DiceCE in a 1:1 ratio. This is not a loss on the difference between predicted and true component counts.

The reference implementation explicitly requires CUDA, two output channels, int16 target labels, and no region or ignore-label training. Its geometry uses CuPy and voxel-coordinate Euclidean distances. `nnUNetTrainerGlobalCCDiceCE` includes both global and component terms; `nnUNetTrainerCCDiceCE` alone is instance-only despite its similar name. Preserve this distinction in the proposal.

### Recommended adaptation

Use the recorded two-foreground-label adaptation: compute CC supervision independently for core and edema and average the two class terms equally. Use this same component term in both requested trainers, adding three-class global Dice + CE only in GlobalCCDiceCE. Do not use the union `{1,2}` for CC geometry: an edema bridge can merge spatially separate cores. Separate components remain geometric proxies, not guaranteed biological lesion identities; edema components need not correspond one-to-one with metastases.

For three-class logits `z`, construct core binary logits `[logsumexp(z_background, z_edema), z_core]` with target `y == 1`, and edema binary logits `[logsumexp(z_background, z_core), z_edema]` with target `y == 2`. Each binary softmax foreground probability is exactly that class's original three-class probability. Dropping the other foreground channel and renormalizing would change the probability and is incorrect. These are differentiable views of the same three-output network, not extra heads or stored-label conversions.

Write the component Dice and component-normalized CE equations explicitly. Use `L_CC = 0.5 * (L_CC_core + L_CC_edema)` and, for GlobalCCDiceCE, start with global and combined CC coefficients both equal to 1 as reference-informed candidates, not established optima. Both trainers are adaptations of the binary reference; apply their complete objectives at every active deep-supervision scale.

The reference uses no additive Dice smoothing, with numerical denominator clamps; its global Dice returns negative Dice while its component Dice returns `1 - Dice`. These offsets affect reported loss values even when gradients agree. Record them. Keep a matched no-additive-smoothing baseline for paper-style comparisons, separately from the project's default `smooth=1e-5` baseline.

Specify absent-class handling independently for core and edema before the EDT: the reference numerical fallback for `dice_ce` is whole-patch binary CE plus a constant 1. Test both the geometry guard and fallback gradients, including classes disappearing at coarse scales. A patch lacking a class must still penalize false-positive probability for that class. Handle full-foreground masks, ties, one-voxel components, and crop-truncated lesions explicitly.

Build a small CPU reference for correctness; use the CUDA/CuPy path for production after parity checks. Do not import the entire reference nnU-Net fork, mutate global compiler settings as a side effect, or depend on two installed packages with the same `nnunetv2` namespace. Check tensor dtype conversion, current CUDA device/stream, and required CuPy version. Preserve voxel-distance geometry for the reference-parity variant; spacing-aware Voronoi regions are a separately named modification.

### Acceptance criteria

Binary adapter probability/gradient identity; explicit-loop versus vectorized values and gradients; reference CUDA parity; correct 26-connectivity and tie handling; CE normalization per Voronoi region; finite empty-mask gradients; small-versus-large lesion weighting; retained false-positive penalties; and feasible time/VRAM overhead on representative 3D patches.

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

## 7. Validation and experiment plan

Write validation before launching training:

1. Verify the selected cohort and three-class data contract, including all label files and patient-disjoint splits. The two local mask spot checks are supporting evidence only.
2. Establish a matched Dice + CE baseline. Add the existing Focal Tversky + CE as an optional comparator with the same training setup.
3. Validate each new loss on small synthetic 3D tensors: perfect prediction, missing lesion, distant false positive, empty foreground, full foreground, two unequal lesions, touching components, displaced boundary, and class swap. Compare numerical values and isolated auxiliary gradients, not just the total backward pass.
4. Run CPU-reference checks where applicable, followed by CUDA, AMP, DDP, and compile-on/off checks. Do not claim the CUDA reference is validated by CPU-only tests. For CC, exercise distinct GPU ranks/devices and variable component counts.
5. Smoke-test real patches and a small overfit case, checking finite losses, all active decoder branches, time per step, and peak allocated GPU memory. Freeze acceptance budgets before a long run.
6. Compare baseline versus CC, contour, and Zhao variants separately using the same fold/seed, architecture, patch size, effective batch size, augmentation, optimizer, epochs/updates, and checkpoint selection rule. If memory forces smaller batches, provide a matched baseline or label the comparison confounded. Do not combine all auxiliary terms in the first experiment.
7. Advance promising candidates to fixed patient-level cross-validation. Keep a held-out test set untouched during coefficient selection. Report paired per-case results, fold dispersion, and small-lesion recall alongside voxel overlap.

Evaluation must run on reconstructed full-volume predictions in the reference geometry, using identical postprocessing across experiments. Report core Dice, edema Dice, derived whole-tumor Dice, and their explicitly defined aggregation. Add core CC-Dice, one-to-one lesion precision/recall/F1, false positives per scan, and recall by lesion volume in mm3. Matching must specify connectivity and overlap criteria; maximum-cardinality one-to-one matching among overlapping pairs is a concrete candidate. Define split/merge and empty-GT conventions. Component counts may be reported as diagnostics but are not sufficient detection metrics.

For boundary comparisons, define surface extraction and a physical tolerance for surface/contour Dice, and optionally HD95/ASSD in mm. These are additional evaluation implementations: the current evaluator's ordinary Dice/IoU output does not establish them. Keep a paper-style voxel-band contour metric distinct from a physical surface metric. Specify both-empty and one-empty outcomes, denominators, NaN handling, and inclusion in aggregate summaries before comparing methods.

## 8. Double-check conclusions

- The combined 0/1/2 label convention is supported by current code and two inspected source masks; the older WT/TC/ET metadata must not determine the new design.
- The supplied CC code is useful reference material but cannot be used unchanged on three-class outputs. Its global-plus-instance trainer is the relevant paper-composition reference.
- The contour method requires a documented gradient strategy; a soft 3D implementation is an adaptation.
- Zhao's printed loss is not a Dice variant and is underspecified for a direct port. Its missing choices must remain visible until explicitly defined.
- The prior referenced conversation was consulted for provenance; it confirms transcription work, not the correctness of a future implementation. The equation pages were checked directly here.
- The subsequent CC integration includes passing CPU loss/gradient and trainer/inference tests; see its document for the exact results. No server preprocessing, CUDA verification, model training, or full-cohort audit has been performed. Contour/Zhao implementation and the broader evaluation work remain future proposals.
