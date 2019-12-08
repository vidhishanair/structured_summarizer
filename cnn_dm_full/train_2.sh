OUTPUT='cnn_dm_full_2'
CUDA_VISIBLE_DEVICES=0 python main.py \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --lr_coverage 0.15 \
       --batch_size 20 \
       --max_dec_steps 20 \
       --reload_path=log/${OUTPUT}/model/model_27000_1575717825 \
       --train_data_path=../finished_files_wlabels_wner_wcoref_chains/chunked/train_* \
       --eval_data_path=../finished_files_wlabels_wner_wcoref_chains/val.bin \
       --vocab_path=../finished_files_wlabels_wner_wcoref_chains/vocab \
       --use_summ_loss \


#       --reload_path=log/${OUTPUT}/model/model_6000_1575696434 \
# model_27000_1575717825
# seed 2