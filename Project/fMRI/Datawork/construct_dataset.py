import re
import os
import pandas as pd
#MARK: - Construct Dataset model_data.csv, already done

def parse_dx(dx):
    if int(dx) == 0:
        return 0
    else:
        return 1
    
def move_files():
    base_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"
    dataset_dir = "/root/autodl-tmp/CNNLSTM/Project/Data"
    files_list = []
    files = [f for f in os.listdir(dataset_dir) if f.endswith('1_bold.nii.gz')] 
    print(len(files))
    print(files[0:9])
    # for file in files:
    #     nums = re.findall(r'\d+', file)
    #     file_id = None
    #     for num in nums: 
    #         if len(num) > 1: 
    #             file_id = int(num)
                
    #     files_list.append({"ScanDir ID": file_id, "Image": file} )

    # images_df = pd.DataFrame(files_list)

    # adhd_info = pd.read_csv("./data/adhd200_preprocessed_phenotypics.tsv", delimiter="\t")[['ScanDir ID','DX']]

    # model_data = adhd_info.merge(images_df, on='ScanDir ID')

    # for index,row in model_data.iterrows():
    #     if row['DX'] == 'pending':
    #         model_data.drop(index,axis=0,inplace=True)

    # model_data['DX'] = model_data['DX'].apply(parse_dx)

    # model_data.to_csv(os.path.join("./data", "model_data.csv"), index=False)
move_files()