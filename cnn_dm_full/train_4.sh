OUTPUT='cnn_dm_full_4'
CUDA_VISIBLE_DEVICES=1 python main.py \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --lr_coverage 0.15 \
       --batch_size 20 \
       --max_dec_steps 40 \
       --reload_path=log/${OUTPUT}/model/model_6000_1575698674 \
       --train_data_path=../finished_files_wlabels_wner_wcoref_chains/chunked/train_* \
       --eval_data_path=../finished_files_wlabels_wner_wcoref_chains/val.bin \
       --vocab_path=../finished_files_wlabels_wner_wcoref_chains/vocab \
       --use_summ_loss \


#       --reload_path=log/${OUTPUT}/model/model_6000_1575698674 \
# seed 4
