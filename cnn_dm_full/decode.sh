#Namespace(L1_structure_penalty=False, autoencode=False, is_coverage=True, pointer_gen=True, reload_path=None, reload_pretrained_clf_path='log/token_sent_level_tag/model/model_455000_1555707099', save_path='sent_level_tag_coverage', sent_scores=True, sep_sent_features=False, token_scores=False)

#OUTPUT='test_loading_summ2'
OUTPUT='cnn_dm_full_2'
# RELOAD_CLF_PATH='log/token_sent_level_tag/model/model_455000_1555707099'
MODEL='model_76000_1575767768'
CUDA_VISIBLE_DEVICES=2 python decode.py \
       --bu_coverage_penalty \
       --pointer_gen \
       --decode_data_path=../artidoro-cnn-dailymail/finished_files_wlabels_wner_wcoref_chains/test.bin \
       --vocab_path=../artidoro-cnn-dailymail/finished_files_wlabels_wner_wcoref_chains/vocab \
       --save_path=${OUTPUT} \
       --reload_path=log/${OUTPUT}/model/${MODEL} \
       --predict_summaries \
       --max_dec_steps 70 \
       --beam_size 3 \


cd ../artidoro-pointer_summarizer/

python pyrouge_eval.py ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_ref/ ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_dec_dir/ > ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt

cat ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt
cat ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/stats.txt

cd ../artidoro-structured_summarizer/


# max_dec_steps 100/120, beam_size 3
# --heuristic_chains \
# --sm_ner_model \
#python pyrouge_eval.py ../artidoro-structured_summarizer/log/bbc_test_low_resource/decode_model_21000_1575521262/rouge_ref/ ../artidoro-structured_summarizer/log/bbc_test_low_resource/decode_model_21000_1575521262/rouge_dec_dir/ > ../artidoro-structured_summarizer/log/bbc_test_low_resource/decode_model_21000_1575521262/rouge_results.txt