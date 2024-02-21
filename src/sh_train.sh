clear;
# rm -rf log_dir
rm -rf output_plots
mkdir output_plots
rm nohup.out

python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir --restore "/nobackup/joon/1_Projects/L3C-PyTorch/data/models/0306_0001 cr oi" --restore_restart
# python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir 
# nohup python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir --restore "/nobackup/joon/1_Projects/L3C-PyTorch/data/models/0306_0001 cr oi" --restore_restart > nohup.out &




# nohup python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir > nohup.out &