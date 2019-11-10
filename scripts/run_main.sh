#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

OUTPUT='test'
#OUTPUT='test_ner'
RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
       --is_coverage \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --reload_pretrained_clf_path=${RELOAD_CLF_PATH} \
       --sent_scores \
       --lr 0.5 \
       --batch_size 60 \
       --max_dec_steps 20 \
       --use_summ_loss

#python main.py \
#       --is_coverage \
#       --pointer_gen \
#       --save_path=${OUTPUT} \
#       --reload_path=log/${OUTPUT}/model/model_215000_1572857694 \
#       --sent_scores \
#       --lr_coverage 0.01 \
#       --batch_size 20 \
#       --max_dec_steps 150 \
#       --use_summ_loss \
