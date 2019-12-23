#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

#OUTPUT='test_loading_summ2'
OUTPUT='bbc_test_low_resource_model_15000_1575518068_first_convergence'
# RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'
MODEL='model_15000_1575518068'
CUDA_VISIBLE_DEVICES=3 python decode.py \
       --is_coverage \
       --pointer_gen \
       --decode_data_path=../data/finished_files_wlabels_wner_wcoref_chains_reduced_1/test.bin \
       --vocab_path=../data/finished_files_wlabels_wner_wcoref_chains_reduced_1/vocab \
       --save_path=${OUTPUT} \
       --reload_path=log/${OUTPUT}/model/${MODEL} \
       --predict_summaries \
       --max_dec_steps 70 \
       --beam_size 3 \
       # max_dec_steps 100/120, beam_size 3
       # --heuristic_chains \
       # --sm_ner_model \

cd ../pointer_summarizer/

# source activate pointgen

python pyrouge_eval.py ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_ref/ ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_dec_dir/ > ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt

cat ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt
cat ../structured_summarizer/log/${OUTPUT}/decode_${MODEL}/stats.txt

# conda deactivate

# source activate py3.7tor1.1

cd ../structured_summarizer/
