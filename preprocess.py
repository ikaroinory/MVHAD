import argparse

from preprocess import preprocess_swat, preprocess_wadi

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
args = parser.parse_args()

if args.dataset == 'swat':
    original_data_path = ('data/original/swat/SWaT_Dataset_Normal_v1.xlsx', 'data/original/swat/SWaT_Dataset_Attack_v0.xlsx')
    processed_data_path = ('data/processed/swat/train.csv', 'data/processed/swat/test.csv')
    preprocess_swat(original_data_path, processed_data_path)
elif args.dataset == 'wadi':
    original_data_path = ('data/original/wadi/WADI_14days_new.csv', 'data/original/wadi/WADI_attackdataLABLE.csv')
    processed_data_path = ('data/processed/wadi/train.csv', 'data/processed/wadi/test.csv')
    preprocess_wadi(original_data_path, processed_data_path)
else:
    parser.error(f'Processing the \'{args.dataset}\' dataset is not currently supported.')
