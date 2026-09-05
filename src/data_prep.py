import os
import gc
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df

def main():
    print("Loading data...")
    dataset_dir = 'dataset'
    
    app_train = pd.read_csv(os.path.join(dataset_dir, 'application_train.csv'))
    bureau = pd.read_csv(os.path.join(dataset_dir, 'bureau.csv'))
    prev_app = pd.read_csv(os.path.join(dataset_dir, 'previous_application.csv'))
    installments = pd.read_csv(os.path.join(dataset_dir, 'installments_payments.csv'))

    app_train = reduce_mem_usage(app_train)
    bureau = reduce_mem_usage(bureau)
    prev_app = reduce_mem_usage(prev_app)
    installments = reduce_mem_usage(installments)

    print("Aggregating bureau data...")
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'DAYS_CREDIT': 'mean',
        'DAYS_CREDIT_ENDDATE': 'mean',
        'AMT_CREDIT_MAX_OVERDUE': 'mean',
        'AMT_CREDIT_SUM': 'sum',
        'AMT_CREDIT_SUM_DEBT': 'sum'
    }).reset_index()
    bureau_agg.columns = ['SK_ID_CURR', 'BUREAU_LOAN_COUNT', 'BUREAU_DAYS_CREDIT_MEAN', 
                          'BUREAU_DAYS_CREDIT_ENDDATE_MEAN', 'BUREAU_AMT_CREDIT_MAX_OVERDUE_MEAN',
                          'BUREAU_AMT_CREDIT_SUM_SUM', 'BUREAU_AMT_CREDIT_SUM_DEBT_SUM']
    del bureau; gc.collect()

    print("Aggregating previous applications...")
    prev_agg = prev_app.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'AMT_APPLICATION': 'mean',
        'AMT_CREDIT': 'mean',
        'DAYS_DECISION': 'mean'
    }).reset_index()
    prev_agg.columns = ['SK_ID_CURR', 'PREV_APP_COUNT', 'PREV_AMT_APPLICATION_MEAN', 
                        'PREV_AMT_CREDIT_MEAN', 'PREV_DAYS_DECISION_MEAN']
    del prev_app; gc.collect()

    print("Aggregating installments payments...")
    installments['DELAY'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
    installments['PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
    
    inst_agg = installments.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'DELAY': 'mean',
        'PAYMENT_DIFF': 'mean',
        'AMT_PAYMENT': 'sum'
    }).reset_index()
    inst_agg.columns = ['SK_ID_CURR', 'INSTAL_COUNT', 'INSTAL_DELAY_MEAN', 'INSTAL_PAYMENT_DIFF_MEAN', 'INSTAL_AMT_PAYMENT_SUM']
    del installments; gc.collect()

    print("Merging features into application_train...")
    df = app_train.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
    
    del app_train, bureau_agg, prev_agg, inst_agg; gc.collect()

    print("Handling missing values and encoding categorical features...")
    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['category', 'object']).columns.tolist()
    
    if 'TARGET' in num_cols:
        num_cols.remove('TARGET')
    if 'SK_ID_CURR' in num_cols:
        num_cols.remove('SK_ID_CURR')
        
    print("Imputing numerical columns...")
    imputer_num = SimpleImputer(strategy='median')
    df[num_cols] = imputer_num.fit_transform(df[num_cols])
    
    print("Imputing categorical columns...")
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    
    print("Label encoding categorical columns...")
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        
    df = reduce_mem_usage(df)
    
    print("Applying SMOTE...")
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET'].astype('int')
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    df_res = pd.concat([X_res, y_res], axis=1)
    
    print("Saving processed dataset to parquet...")
    df_res.columns = df_res.columns.astype(str)
    df_res.to_parquet(os.path.join(dataset_dir, 'processed_train.parquet'), index=False)
    print("Phase 1 complete. Dataset saved to dataset/processed_train.parquet")

if __name__ == "__main__":
    main()
