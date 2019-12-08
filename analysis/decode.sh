OUTPUT='analysis'
MODEL='model_174000_1575579625'
CUDA_VISIBLE_DEVICES=2 python decode.py \
       --bu_coverage_penalty \
       --pointer_gen \
       --decode_data_path=../finished_files_wlabels_wner_wcoref_chains/test.bin \
       --vocab_path=../finished_files_wlabels_wner_wcoref_chains/vocab \
       --save_path=${OUTPUT} \
       --reload_path=/home/ubuntu/${MODEL} \
       --predict_summaries \
       --max_dec_steps 70 \
       --beam_size 3 \
       --heuristic_chains \
       --use_ner \


cd ../artidoro-pointer_summarizer/

python pyrouge_eval.py ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_ref/ ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_dec_dir/ > ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt

cat ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/rouge_results.txt
cat ../artidoro-structured_summarizer/log/${OUTPUT}/decode_${MODEL}/stats.txt

cd ../artidoro-structured_summarizer/