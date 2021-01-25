#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

OUTPUT='pretrain_heuristic_ner_chains'
#OUTPUT='test_ner'
RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'
#--reload_pretrained_clf_path=${RELOAD_CLF_PATH} \

CUDA_VISIBLE_DEVICES=0,1 python main.py \
       --is_coverage \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --reload_path=log/${OUTPUT}/model/model_65000_1572335399 \
       --sent_scores \
       --test_sent_matrix \
       --lr_coverage 0.0001 \
       --batch_size 100 \
       --max_dec_steps 20 \
       --train_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_wlabels_wnerchains/chunked/train_* \
       --eval_data_path=/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_wlabels_wnerchains/val.bin \
       --heuristic_chains \
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
