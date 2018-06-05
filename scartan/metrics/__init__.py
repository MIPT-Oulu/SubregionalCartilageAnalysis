from torch import nn
from .volume import (confusion_matrix,
                     dice_score, dice_score_from_cm,
                     jaccard_score, jaccard_score_from_cm,
                     precision_score, precision_from_cm,
                     recall_score, recall_from_cm,
                     sensitivity_score, sensitivity_from_cm,
                     specificity_score, specificity_from_cm,
                     volumetric_similarity, vol_sim_from_cm,
                     volume_error, volume_total)
from .surface import (avg_surf_dist,
                      avg_symm_surf_dist,
                      rms_symm_surf_dist,
                      robust_hausdorff_dist,
                      surf_dice_at_tol,
                      surf_overlap_at_tol)
from .local_thickness import local_thickness
from .distance_transf import distance_transform


dict_metrics = {
    'confusion_matrix': confusion_matrix,
    'dice_score': dice_score,
    'jaccard_score': jaccard_score,
    'precision_score': precision_score,
    'recall_score': recall_score,
    'sensitivity_score': sensitivity_score,
    'specificity_score': specificity_score,
    'volumetric_similarity': volumetric_similarity,
    'volume_error': volume_error,
    'volume_total': volume_total,

    'bce_loss': nn.BCELoss(),

    'avg_surf_dist': avg_surf_dist,
    'avg_symm_surf_dist': avg_symm_surf_dist,
    'rms_symm_surf_dist': rms_symm_surf_dist,
    'robust_hausdorff_dist': robust_hausdorff_dist,
    'surf_dice_at_tol': surf_dice_at_tol,
    'surf_overlap_at_tol': surf_overlap_at_tol,

    'local_thickness': local_thickness,
    'distance_transform': distance_transform,
}
