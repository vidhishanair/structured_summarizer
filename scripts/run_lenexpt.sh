#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

OUTPUT='test_len_change'
#OUTPUT='test_ner'
RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'

CUDA_VISIBLE_DEVICES=0,1 python main.py \
       --is_coverage \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --reload_pretrained_clf_path=${RELOAD_CLF_PATH} \
       --sent_scores \
       --test_sent_matrix \
       --lr 0.75 \
       --batch_size 40 \
       --max_dec_steps 100 \
       --use_summ_loss \
       --test_len 

#python main.py \
#       --is_coverage \
#       --pointer_gen \
#       --save_path=${OUTPUT} \
#       --reload_path=log/${OUTPUT}/model/model_120000_1572379081 \
#       --sent_scores \
#       --test_sent_matrix \
#       --lr_coverage 0.05 \
#       --batch_size 20 \
#       --max_dec_steps 150 \
#       --use_summ_loss \
