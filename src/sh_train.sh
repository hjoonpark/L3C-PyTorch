clear;
rm -rf log_dir
rm -rf output_plots
mkdir output_plots
rm nohup.out
# python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir
nohup python train.py configs/ms/cr.cf configs/dl/oi.cf log_dir > nohup.out &