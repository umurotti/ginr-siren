# 2023_vision_practical
1. Data Preprocessing: set mode to 'siren_sdf' or 'sdf' to generate the data. Mode 'siren_sdf' is used for sdf representation, together with loss_formulation in siren paper. Mode 'sdf' is used for sdf or sdf representation, together with L1_loss or BCE_loss.
   
   
```python
   python utils/mesh_to_sdf_test.py --mode=siren_sdf --input_folder=$YOUR_INPUT_FOLDER --output_folder=$YOUR_OUTPUT_FOLDER

```

2. Fill in the data_path in the corresponding config files and start training.
```python
   # for single shape overfitting 
   python main_stage_inr.py -m=./config/shapenet_meta_sdf_overfit.yaml -t=$EXP_NAME

   # for meta_learning
   python main_stage_inr.py -m=./config/shapenet_meta_sdf.yaml -t=$EXP_NAME
```  
