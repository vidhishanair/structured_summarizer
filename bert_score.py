import os
from tqdm import tqdm

def preprocess_results(path_results, dest_file_path):
    files = os.listdir(path_results)
    with open(dest_file_path, 'w') as dest_file:
        for res_file_path in tqdm(files):
            with open(os.path.join(path_results, res_file_path)) as res_file:
                line = res_file.read().replace('\n', '')
                dest_file.write(line + '\n')

if __name__ == '__main__':
    # reference_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/reference"
    # baseline_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/baseline"
    # pointgen_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/pointer-gen"
    # pointgen_cov_dir = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/pointer-gen-cov"

    # reference_dest = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/reference.txt"
    # baseline_dest = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/baseline.txt"
    # pointgen_dest = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/pointer-gen.txt"
    # pointgen_cov_dest = "/home/artidoro/data/Pointer Generator Network Test Output/test_output/pointer-gen-cov.txt"

    # preprocess_results(reference_dir, reference_dest)
    # preprocess_results(baseline_dir, baseline_dest)
    # preprocess_results(pointgen_dir, pointgen_dest)
    # preprocess_results(pointgen_cov_dir, pointgen_cov_dest)


    reference_dir = "/home/ubuntu/artidoro-structured_summarizer/log/analysis/decode_model_174000_1575579625/rouge_ref"
    reference_dest = "/home/ubuntu/artidoro-structured_summarizer/log/analysis/decode_model_174000_1575579625/ref_file.txt"

    struct_sum_dir = "/home/ubuntu/artidoro-structured_summarizer/log/analysis/decode_model_174000_1575579625/rouge_dec_dir"
    struct_sum_dest = "/home/ubuntu/artidoro-structured_summarizer/log/analysis/decode_model_174000_1575579625/rouge_dec.txt"

    preprocess_results(reference_dir, reference_dest)
    preprocess_results(struct_sum_dir, struct_sum_dest)