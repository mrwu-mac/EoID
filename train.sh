python3 -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        main.py \
        --pretrained params/detr-r50-pre-2stage-q64.pth \
        --output_dir output/EoID \
        --dataset_file hico_ua_st_v1 \
        --hoi_path data/hico_20160224_det \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --backbone resnet50 \
        --num_queries 64 \
        --dec_layers_hopd 3 \
        --dec_layers_interaction 3 \
        --epochs 90 \
        --lr_drop 60 \
        --use_nms_filter \
        --batch_size 4 \
        --clip_backbone RN50x16 \
        --model eoid \
        --inter_score \
        --vdetach \
#        --uc_type uc0 \


