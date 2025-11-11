import argparse

from preprocess import preprocess_BATADAL, preprocess_SWaT, preprocess_WaDi

parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['SWaT', 'WaDi', 'BATADAL'], type=str)
args = parser.parse_args()

if args.dataset == 'SWaT':
    original_data_path = ('data/original/swat/SWaT_Dataset_Normal_v1.xlsx', 'data/original/swat/SWaT_Dataset_Attack_v0.xlsx')
    processed_data_path = ('data/processed/swat/train.csv', 'data/processed/swat/test.csv')
    preprocess_SWaT(original_data_path, processed_data_path)
elif args.dataset == 'WaDi':
    original_data_path = ('data/original/wadi/WADI_14days_new.csv', 'data/original/wadi/WADI_attackdataLABLE.csv')
    processed_data_path = ('data/processed/wadi/train.csv', 'data/processed/wadi/test.csv')
    preprocess_WaDi(original_data_path, processed_data_path)
elif args.dataset == 'BATADAL':
    original_data_path = ('data/original/BATADAL/BATADAL_dataset03.csv', 'data/original/BATADAL/BATADAL_dataset04.csv')
    processed_data_path = ('data/processed/BATADAL/train.csv', 'data/processed/BATADAL/test.csv')
    preprocess_BATADAL(original_data_path, processed_data_path)
else:
    parser.error(f'Processing the \'{args.dataset}\' dataset is not currently supported.')
