import numpy as np
import pandas as pd
import os
import argparse


def main():
    parser = argparse.ArgumentParser(description="Create dataset based on train/test/validation slide list")
    parser.add_argument("--data_path", type=str, default="raw data/ds_binary_from_og.csv", help="Path to input data csv file.")
    parser.add_argument("--slide_list", type=str, default="raw data/slide_list.csv", help="Path to list designating each slide to train/test/val.")
    parser.add_argument("--save_name", type=str, default="CV_dataset", help="Path to list designating each slide to train/test/val.")
    parser.add_argument("--multiclass", action="store_true", help="Whether creating binary or multiclass dataset.")
    args = parser.parse_args()
    
    data = pd.read_csv(args.data_path)
    slide_list = pd.read_csv(args.slide_list)
    
    if not args.multiclass:
        data["Class"] = data["Class"].apply(lambda x: 1 if x == "ncancerous" else 0)
    else:
        class_mapping = {
        'adenocarcinoma': 0,
        'benign mucosa': 1,
        'smooth muscle': 2,
        'inflammatory cells': 3,
        'serosa': 4,
        'submucosa': 5
        }

        data["Class"] = data["Class"].map(class_mapping)
    
    print("Unique Class Counts:", data["Class"].value_counts())
    
    test_set = pd.DataFrame()
    val_set = pd.DataFrame()
    train_set = pd.DataFrame()

    # Filter data based on substring matching
    for _, row in slide_list.iterrows():
        slide_substring = row["Slide"]
        dataset = row["Set"]
        
        if dataset == "Test":
            test_set = pd.concat([test_set, data[data['Slide'].str.contains(slide_substring, na=False)]]).sort_index()
        elif dataset == "Val":
            val_set = pd.concat([val_set, data[data['Slide'].str.contains(slide_substring, na=False)]]).sort_index()
        elif dataset == "Train":
            train_set = pd.concat([train_set, data[data['Slide'].str.contains(slide_substring, na=False)]]).sort_index()
    
    print("Test Slides:", test_set['Slide'].unique())
    print("Validation Slides:", val_set['Slide'].unique())
    print("Train Slides:", train_set['Slide'].unique())
    
    train_set = train_set.drop(columns=['Slide', 'Y', 'X'])
    test_set = test_set.drop(columns=['Slide', 'Y', 'X'])
    val_set = val_set.drop(columns=['Slide', 'Y', 'X'])

    y_train = train_set['Class'].values
    y_test = test_set['Class'].values
    y_val = val_set['Class'].values

    X_train = train_set.drop(columns='Class')
    X_test = test_set.drop(columns='Class')
    X_val = val_set.drop(columns='Class')

    np.savez(f"Datasets/{args.save_name}.npz", X_train=X_train, y_train=y_train, 
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)


if __name__ == '__main__':
    main()