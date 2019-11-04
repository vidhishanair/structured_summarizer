#Content of this file is copied from https://github.com/atulkum/pointer_summarizer/blob/master/

import os

#root_dir = os.path.expanduser("~")
root_dir = os.path.expanduser("/remote/bones/user/public/vbalacha/structured_summarizer")
#train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
# train_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files/chunked/train_*")
# eval_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files/val.bin")
# decode_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files/test.bin")
# vocab_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files/vocab")

train_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/chunked/train_*")
#train_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/cnn-dailymail/finished_files_wlabels_wnerchains/chunked/train_*")
eval_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/val.bin")
decode_data_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/test.bin")
vocab_path = os.path.join(root_dir, "/remote/bones/user/public/vbalacha/datasets/cnndailymail/finished_files_wlabels_p3/vocab")

embeddings_file = os.path.join(root_dir, 'glove/glove.6B.300d.txt')

log_root = os.path.join(root_dir, "log")

# Hyperparameters
hidden_dim= 256
sem_dim_size = 150

emb_dim= 128
batch_size=60
max_enc_steps=400
#max_dec_steps=100
max_dec_steps=50
beam_size=6
min_dec_steps=35
vocab_size=50000

#lr=0.35
lr=0.5
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

# pointer_gen = True
# is_coverage = False
# autoencode = False
# concat_rep = True

cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000
eval_interval = 5000

use_gpu=True

#lr_coverage=0.15
lr_coverage=0.75

use_maxpool_init_ctx = True
