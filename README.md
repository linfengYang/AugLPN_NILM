# **
In this repository are available codes for implementation of our study.

# Requirements:
The version of python should preferably be greater than 3.7
our environment(for reference only):
    tensorflow==2.3.0
    keras==2.4.0
    scikit-learn==1.1.2

# Reference(Acknowledgement):
1. https://github.com/MingjunZhong/NeuralNetNilm
2. https://github.com/MingjunZhong/transferNILM/
3. C. Zhang, M. Zhong, Z. Wang, N. Goddard, and C. Sutton. Sequence-to-point learning with neural networks
for non-intrusive load monitoring. In Proceedings for Thirty-Second AAAI Conference on Artificial Intelligence.
AAAI Press, 2018.


# Get the paper results quickly
Some already well-trained models ('*.h5' files) are in the folder directory '/models' 
Change the file path (refer to the parameter 'param_file')  in the AugPANNILM_test.py, and you will get the results soon.
    for example: param_file = args.trained_model_dir + '/UK_DALE'+ '/AugPANP_' + args.appliance_name + '_pointnet_model'
note:AugPAN and AugPANP (AugPAN with attention mechanism) are our proposed models

# --------------------***Reproduce  our results***-----start--------------------
# 1. Prepare training and test dataset for REDD and UK_DALE
1. REDD and UK_DALE datasets are available in (http://redd.csail.mit.edu/) and (https://data.ukedc.rl.ac.uk/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2015/UK-DALE-disaggregated).
2. Put the raw data into the folder directory dataset_preprocess, and named low_freq and UK_DALE respectively.
3. Run redd_processing.py and uk_dale_processing.py to get the prepared dataset for training and test.
   (note that the preprocessing of UK_DALE dataset needs another step :put the preprocessed data in "dataset_preprocess/created_data/UK_DALE/" )
The structure of folder directory is as follows:
   dataset_processing/
        created_data/
            REDD/
            UK_DALE/
        low_freq/
            house_1/
            house_2/
            ...
        UK_DALE/
            house_2/
        redd_processing.py
        ukdale_processing.py

# 2. Start the training
You can run AugPANNILM_train.py to verify the results in our paper after you have preprocessed all the dataset.
The best results('*.h5' files) will be stored in the file directory '/models'
# 3. Start the test
Change the file path (refer to the parameter 'param_file')  in the AugPANNILM_test.py, and you will get the results soon.
    for example: param_file = args.trained_model_dir + '/AugPANP_' + args.appliance_name + '_pointnet_model'
# --------------------***Reproduce  our results***-----end--------------------

Contact e-mail:
2113301058@st.gxu.edu.cn
