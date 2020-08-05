CUDA_VISIBLE_DEVICES=0 nohup python ./seq2seq_tf2/bin/main.py \
                        --batch_size 64 \
                        --vocab_size 50000 \
                        --enc_units 256 \
                        --dec_units 256 \
                        --attn_units 256 \
                        --mode 'test' > ./log/train.log &


CUDA_VISIBLE_DEVICES=0 python ./seq2seq_tf2/bin/main.py \
                        --batch_size 64 \
                        --vocab_size 50000 \
                        --enc_units 256 \
                        --dec_units 256 \
                        --attn_units 256 \
                        --mode 'test'
