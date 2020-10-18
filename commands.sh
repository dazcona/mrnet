#!/bin/bash

## PRE-TRAINED

# ACL

# python 'src/train_base.py' \
#     --prefix_name 'base' \
#     -t 'acl' \
#     -p 'axial' \
#     --epochs 100 \
#     --augment_prob 0.75

# python 'src/train_base.py' \
#     --prefix_name 'base' \
#     -t 'acl' \
#     -p 'coronal' \
#     --epochs 100 \
#     --augment_prob 0.90

# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'acl' \
#     -p 'sagittal' \
#     --epochs 100 \
#     --augment_prob 0.85

# Menicus

# Best model at "model_test_meniscus_axial_train_auc_0.9776_val_auc_0.8815_train_loss_0.2697_val_loss_0.6016_epoch_16_arch_pretrained-resnet18_cut_vertical_augment_albumentations-group_augment-probability_0.4.pth" 
# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'meniscus' \
#     -p 'axial' \
#     --epochs 200 \
#     --augment_prob 0.40

# Best model at "model_test_meniscus_coronal_train_auc_0.9861_val_auc_0.8521_train_loss_0.2198_val_loss_0.7246_epoch_19_arch_pretrained-resnet18_cut_vertical_augment_albumentations-group_augment-probability_0.4.pth"
# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'meniscus' \
#     -p 'coronal' \
#     --epochs 200 \
#     --augment_prob 0.40

# Best model at "model_test_meniscus_sagittal_train_auc_0.9164_val_auc_0.7805_train_loss_0.5253_val_loss_0.7107_epoch_14_arch_pretrained-resnet18_cut_vertical_augment_albumentations-group_augment-probability_0.9.pth"
# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'meniscus' \
#     -p 'sagittal' \
#     --epochs 100 \
#     --augment_prob 0.90

# Abnormal

# Best model at "model_test_abnormal_axial_train_auc_0.9154_val_auc_0.9486_train_loss_0.1356_val_loss_0.1243_epoch_12_arch_pretrained-resnet18_cut_vertical_augment_albumentations-group_augment-probability_0.55.pth"
# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'abnormal' \
#     -p 'axial' \
#     --epochs 100 \
#     --augment_prob 0.55

# Best model at "model_test_abnormal_coronal_train_auc_0.9824_val_auc_0.9128_train_loss_0.0685_val_loss_0.2088_epoch_12_arch_pretrained-resnet18_cut_vertical_augment_albumentations-group_augment-probability_0.05.pth"
# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'abnormal' \
#     -p 'coronal' \
#     --epochs 100 \
#     --augment_prob 0.05

# Best model at "model_test_abnormal_sagittal_train_auc_0.9227_val_auc_0.9503_train_loss_0.1359_val_loss_0.1396_epoch_15_arch_pretrained-resnet18_cut_vertical_augment_albumentations-group_augment-probability_0.35.pth"
# python 'src/train_baseline.py' \
#     --prefix_name 'base' \
#     -t 'abnormal' \
#     -p 'sagittal' \
#     --epochs 100 \
#     --augment_prob 0.35

# MODEL='pretrained-resnet18'
# EPOCHS=20
# CUT='vertical'
# AUGMENT='albumentations-group'
# AUGMENT_PROB=0.5
# EXPERIMENT='experiment'

# for TASK in acl meniscus abnormal
# do

#     for PLANE in axial coronal sagittal
#     do

# 	    echo "Running training for '$TASK' & '$PLANE'"

#         for AUGMENT_PROB in `seq 0.00 0.05 1`
#         do

#         python src/train_baseline.py \
#             --prefix_name "$EXPERIMENT" \
#             -t "$TASK" \
#             -p "$PLANE" \
#             --epochs "$EPOCHS" \
#             --model "$MODEL" \
#             --cut "$CUT" \
#             --augment "$AUGMENT" \
#             --augment_prob "$AUGMENT_PROB"

#         done

#     done

# done

## SLICES

# MODEL='resnet18'
# EPOCHS=20
# CUT='vertical'
# AUGMENT='albumentations-group'
# AUGMENT_PROB=0.5
# EXPERIMENT='experiment'

# for TASK in acl meniscus abnormal
# do

#     for PLANE in axial coronal sagittal
#     do

# 	    echo "Running training for '$TASK' & '$PLANE'"

#         for AUGMENT_PROB in `seq 0.00 0.05 1`
#         do

#         python src/train_slices.py \
#             --prefix_name "$EXPERIMENT" \
#             -t "$TASK" \
#             -p "$PLANE" \
#             --epochs "$EPOCHS" \
#             --model "$MODEL_SLICES" \
#             --cut "$CUT" \
#             --augment "$AUGMENT" \
#             --augment_prob "$AUGMENT_PROB"

#         done

#     done

# done

