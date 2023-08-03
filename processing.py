import argparse
import os
import logging
import warnings

import pandas as pd
import numpy as np
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(format='%(asctime)s %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    # Get the processing job arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.2)
    parser.add_argument('--data-location', type=str, default='/opt/ml/processing/input/data')
    parser.add_argument('--train-location', type=str, default='/opt/ml/processing/train')
    parser.add_argument('--test-location', type=str, default='/opt/ml/processing/test')
    args, _ = parser.parse_known_args()
    logger.info(f'Received arguments {args}')

    # Get the input data
    files = os.listdir(args.data_location)
    logger.debug(f'Files: {files}')
    if len(files) == 1:
        df = pd.read_csv(os.path.join(args.data_location, files[0]), index_col='idx')
    else:
        raise Exception('This script does not support multiple files.')

    # Prepare data
    df['police_report_available'] = df['police_report_available'].replace('?', np.nan)
    # Drop rows with no data about police reports
    df = df.dropna(subset=['police_report_available'])

    # Drop the months as customer column
    df = df.drop(['months_as_customer'], axis=1)

    # Transform categorical values.
    le = LabelEncoder()
    for i in list(df.columns):
        if df[i].dtype == 'object':
            df[i] = le.fit_transform(df[i])

    split_ratio = args.train_test_split_ratio
    logger.debug(f'Splitting data into train and test sets with ratio {split_ratio}')
    train, test = train_test_split(
        df, test_size=split_ratio
    )
    logger.debug(f'Train data shape after preprocessing: {train.shape}')
    logger.debug(f'Test data shape after preprocessing: {test.shape}')

    train_output_path = os.path.join(args.train_location, 'train.csv')
    test_output_path = os.path.join(args.test_location, 'test.csv')

    logger.debug(f'Writing train data to {train_output_path}')
    pd.DataFrame(train).to_csv(train_output_path, header=False, index=False)

    logger.debug(f'Writing test data to {test_output_path}')
    pd.DataFrame(test).to_csv(test_output_path, header=False, index=False)

    logger.info('Done')
