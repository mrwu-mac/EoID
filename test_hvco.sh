python3 main.py \
        --pretrained output/logs_hvco_vit32_hard_0.1/checkpoint_best.pth \
        --dataset_file hvco \
        --hoi_path data \
        --num_obj_classes 80 \
        --num_verb_classes 123 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --eval \
        --use_nms_filter \
        --clip 8 \
        --clip_backbone RN50x16 \
        # --use_matching \