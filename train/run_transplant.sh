python transplant.py --model /workspace/volume/guojun/Train/ObjDetection/outputs/best.pth --num_classes 4 --quantized_mode 1 --batch_size 1 --core_version MLU270 --core_number 16 --offline_model /workspace/volume/guojun/Train/ObjDetection/offline/best-mlu270.cambricon --device mlu
# python transplant.py --model ./runs/best.pth --num_classes 1 --quantized_mode 1 --batch_size 1 --core_version MLU220 --core_number 4 --offline_model ./offline/best-mlu220.cambricon --device mlu
# python transplant.py --model ./runs/best.pth --num_classes 1 --quantized_mode 1 --batch_size 1 --core_version MLU270 --core_number 16 --offline_model ./offline/best-mlu270.cambricon --device gpu