OUTPUT='undir_wcoref_dec_sent_att'
#MODEL='model_12000_1575717509'

CUDA_VISIBLE_DEVICES=3 python main.py \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --train_data_path=/home/ubuntu/finished_files_wlabels_wner_wcoref_chains/chunked/train_* \
       --eval_data_path=/home/ubuntu/finished_files_wlabels_wner_wcoref_chains/val.bin \
       --vocab_path=/home/ubuntu/finished_files_wlabels_wner_wcoref_chains/vocab \
       --lr 0.15 \
       --batch_size 20 \
       --max_dec_steps 20 \
       --use_summ_loss \
       --heuristic_chains \
       --use_coref \
       --use_weighted_annotations \
       --use_undirected_weighted_graphs \
       --use_coref_att_encoder \
       --sent_attention_at_dec \
       --clear_old_checkpoints \