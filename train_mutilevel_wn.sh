python train_gta2cityscapes_multi_wn.py --snapshot-dir ./snapshots/GTA2Cityscapes_multi_wn \
                                     --lambda-seg 0.1 \
                                     --lambda-adv-target1 0.0002 --lambda-adv-target2 0.001



 python train_gta2cityscapes_multi_syn_noD2.py --snapshot-dir /data2/zhangjunyi/snapshots/snashots_syn2/syn_nod2_2_1  \
                 --lambda-seg 0.1                    \                  
                 --lambda-adv-target1 0.0002 --lambda-av-target2 0.001                     \                 
                 --beta 0.95