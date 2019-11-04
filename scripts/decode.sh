#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

OUTPUT='dataparallel_speed_changes_emnlp_repro'
RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'
MODEL='model_210000_1572799314'
python decode.py \
       --is_coverage \
       --pointer_gen \
       --save_path=${OUTPUT} \
       --reload_path=log/${OUTPUT}/model/${MODEL} \
       --sent_scores \

cd ../pointer_summarizer/

source activate pointgen

python pyrouge_eval.py ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_ref/ ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_dec_dir/ > ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt

cat ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt

source deactivate

source activate py3.7tor1.1

cd ../structured_summarizer/
