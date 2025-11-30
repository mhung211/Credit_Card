import numpy as np

def read_BankChurners(path):
    dtypes = [
        ('CLIENTNUM', int),
        ('Attrition_Flag', 'U20'),  
        ('Customer_Age', int),
        ('Gender', 'U10'),
        ('Dependent_count', int),
        ('Education_Level', 'U30'),
        ('Marital_Status', 'U20'),
        ('Income_Category', 'U20'),
        ('Card_Category', 'U20'),
        ('Months_on_book', int),
        ('Total_Relationship_Count', int),
        ('Months_Inactive_12_mon', int),
        ('Contacts_Count_12_mon', int),
        ('Credit_Limit', float),
        ('Total_Revolving_Bal', int),
        ('Avg_Open_To_Buy', float),
        ('Total_Amt_Chng_Q4_Q1', float),
        ('Total_Trans_Amt', int),
        ('Total_Trans_Ct', int),
        ('Total_Ct_Chng_Q4_Q1', float),
        ('Avg_Utilization_Ratio', float)
    ]
    data = np.genfromtxt(
        path, 
        delimiter=',', 
        dtype=dtypes, 
        skip_header=1,   
        encoding='utf-8'
    )
    col_names = [dtype[0] for dtype in dtypes]
    return data, col_names

def remove_outlier(col):
    Q1, Q3 = np.percentile(col, [25, 75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    for i in col:
        if i < lower:
            i = lower
        elif i > upper:
            i = upper
    return col

def minmax_scaler(col):
    min_val = col.min()
    max_val = col.max()
    if max_val != min_val:
        return (col - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(col)
    
def feature_encode(col):
    unique_vals = np.unique(col)
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    encoded = np.array([mapping[val] for val in col])
    return encoded, mapping

def encode_label(y):
    return (y == "Attrited Customer").astype(int)