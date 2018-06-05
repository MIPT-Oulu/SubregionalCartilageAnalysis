#!/bin/bash

EXPERIM_ID=20190720_0001

# ---- Prepare datasets
(cd ../scartan/datasets &&
 python prepare_dataset_oai_imo.py \
    /home/egor/MedData/OAI_iMorphics_scans \
    /home/egor/MedData/OAI_iMorphics_annotations \
    /home/egor/Workspace/proj_scartan/data/91_OAI_iMorphics_full_meta \
    --margin 20 \
)

(cd ../scartan/datasets &&
 python prepare_dataset_oai_custom.py \
    /home/egor/MedData/OAI_project_22 \
    /home/egor/Workspace/proj_scartan/data/61_OAI_project_22_full_meta \
    --margin 20 \
)
# ----

# ---- Build multi-atlases
(cd ../scartan &&
 python build_multiatlas.py \
    /home/egor/Workspace/proj_scartan/data/atlas_OAI_iMorphics_Chondr75n \
)

(cd ../scartan &&
 python build_multiatlas.py \
    /home/egor/Workspace/proj_scartan/data/atlas_OAI_iMorphics_Biomediq \
)
# ----

# ---- Train segmentation model
(cd ../scartan/ &&
 python train.py \
    --path_data_root ../../data \
    --path_experiment_root \../../results/${EXPERIM_ID} \
    --model_segm vgg19bn_unet --pretrained \
    --input_channels 1 --output_channels 5 --center_depth 1 \
    --lr_segm 0.0001 --batch_size 16 --epoch_num 30 --fold_num 5 --fold_idx -1 \
    --mixup_alpha 0.7 \
    --num_workers 12 \
)
# ----

# ---- Make testing set predictions
(cd ../scartan/ &&
 python evaluate.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --model_segm vgg19bn_unet --center_depth 1 \
    --restore_weights --output_channels 5 \
    --dataset oai_imo --subset test --mask_mode all_unitibial_unimeniscus \
    --batch_size 40 --fold_num 5 --fold_idx -1 \
    --predict_folds --merge_predictions \
    --num_workers 12 \
)

(cd ../scartan/ &&
 python evaluate.py \
    --path_data_root ../../data \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --model_segm vgg19bn_unet --center_depth 1 \
    --restore_weights --output_channels 5 \
    --dataset oai_prj_22 --subset all --mask_mode all_unitibial_unimeniscus \
    --batch_size 40 --fold_num 5 --fold_idx -1 \
    --predict_folds --merge_predictions \
    --num_workers 12 \
)
# ----

# ---- Make sub-regional division
(cd ../scartan/ &&
 python register_remap.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --path_atlas_root ../../data/atlas_OAI_iMorphics_Chondr75n \
    --atlas_suffix chondr75 \
    --path_elastix_root /home/egor/Software/elastix-5.0.0-linux \
    --path_config_elastix ./registration/config_lk8_elastic.txt \
    --dataset oai_imo \
    --num_workers 4 \
    --num_threads_elastix 6 \
)

(cd ../scartan/ &&
 python register_remap.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --path_atlas_root ../../data/atlas_OAI_iMorphics_Biomediq \
    --atlas_suffix biomediq \
    --path_elastix_root /home/egor/Software/elastix-5.0.0-linux \
    --path_config_elastix ./registration/config_lk8_elastic.txt \
    --dataset oai_prj_22 \
    --no_prep \
    --num_workers 4 \
    --num_threads_elastix 6 \
)

(cd ../scartan/ &&
 python register_remap.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --path_atlas_root ../../data/atlas_OAI_iMorphics_Chondr75n \
    --atlas_suffix chondr75n \
    --path_elastix_root /home/egor/Software/elastix-5.0.0-linux \
    --path_config_elastix ./registration/config_lk8_elastic.txt \
    --dataset oai_prj_22 \
    --no_prep \
    --num_workers 4 \
    --num_threads_elastix 6 \
)
# ----

# ---- Quantify predictions
(cd ../scartan/ &&
 python quantify_predictions.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --dirname_pred mask_foldavg_imo \
    --dirname_true mask_prep_imo \
    --dataset oai_imo \
    --atlas imo \
    --num_workers 12 \
    --ignore_cache \
)

(cd ../scartan/ &&
 python quantify_predictions.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --dirname_pred mask_foldavg_chondr75n \
    --dirname_true mask_prep_chondr75n \
    --dataset oai_imo \
    --atlas chondr75n \
    --num_workers 12 \
    --ignore_cache \
)

(cd ../scartan/ &&
 python quantify_predictions.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --dirname_pred mask_foldavg_biomediq \
    --dataset oai_prj_22 \
    --atlas biomediq \
    --num_workers 12 \
    --ignore_cache \
)

(cd ../scartan/ &&
 python quantify_predictions.py \
    --path_experiment_root ../../results/${EXPERIM_ID} \
    --dirname_pred mask_foldavg_chondr75n \
    --dataset oai_prj_22 \
    --atlas chondr75n \
    --num_workers 12 \
    --ignore_cache \
)
# ----

# ---- Analyze predictions
(cd ../notebooks/ &&
  jupyter notebook Analyze_assessments.ipynb
)
# ----

# ---- Create figures, other artefacts
(cd ../notebooks/ &&
  jupyter notebook Artefacts_and_rev.ipynb
)
# ----
