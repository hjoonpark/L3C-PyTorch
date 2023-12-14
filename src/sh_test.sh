clear
rm -rf "../data/models_test/0306_0001 cr oi"
rm -rf "test_sampled/0306_0001"
python test.py "/nobackup/joon/1_Projects/L3C-PyTorch/data/models/" 0306_0001 "/nobackup/joon/1_Projects/L3C-PyTorch/data/val_oi" \
    --names "L3C" --recursive=auto --sample "test_sampled"