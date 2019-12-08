#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

OUTPUT='bbc_test_full_2'
# OUTPUT='test_loading_summ2'
#RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'

#--reload_path=log/${OUTPUT}/model/model_156000_1574266495 \
CUDA_VISIBLE_DEVICES=3 python main.py \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --lr_coverage 0.15 \
       --batch_size 20 \
       --max_dec_steps 20 \
       --train_data_path=../artidoro-cnn-dailymail/finished_files_wlabels_wner_wcoref_chains/chunked/train_* \
       --eval_data_path=../artidoro-cnn-dailymail/finished_files_wlabels_wner_wcoref_chains/val.bin \
       --vocab_path=../artidoro-cnn-dailymail/finished_files_wlabels_wner_wcoref_chains/vocab \
       --use_summ_loss \
       # --heuristic_chains \
       # --use_coref \
       # --is_coverage \

       # Weighted coref
       # --use_gold_annotations_for_decode
       # --use_weighted_annotations     # otherwise it's binary
       # --all_sent_head_at_decode # or child

# Learning rate
# 0.75    ----    0.01
#       --reload_path=log/${OUTPUT}/model/model_300000_1575711197 \

# To reload
# log/FOLDERNAME/model/modelNAME
#       --reload_path=log/sent_level_tag_coverage/model/model_315000_1557756822 \


#       --use_summ_loss \
#       --link_id_typed \
       

#python main.py \
#       --is_coverage \
#       --pointer_gen \
#       --save_path=${OUTPUT} \
#       --reload_path=log/${OUTPUT}/model/model_30000_1572141813\
#       --sent_scores \
#       --test_sent_matrix \
#       --lr_coverage 0.75 \
#       --batch_size 60 \
#       --max_dec_steps 50 \
#       --train_data_path \
#       --eval_data_path \
#       --heuristic_chains \
#       --link_id_typed \
