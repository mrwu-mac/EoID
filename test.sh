python3 main.py \
        --pretrained ckpt/EoID.pth \
        --dataset_file hico_ua_st_v1 \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --eval \
        --use_nms_filter \
        --model eoid \
        --clip_backbone RN50x16 \
        # --use_matching \