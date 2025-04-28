# English Summary:

Starting with a baseline XGBoost model (RMSE = 1.23151), introducing an LSTM-based approach yielded a modest 6.3% reduction in error (RMSE = 1.15367). However, by combining extensive feature engineering with a LightGBM regressor, we achieved a further 27.5% drop in RMSE, ultimately reducing the error by 32.0% from the original XGBoost baseline (down to RMSE = 0.83687). This highlights the substantial value of tailored features in boosting predictive accuracy beyond model architecture alone.




# Future Sales Prediction Feature Engineering (RMSE: 1.21721)

This document details the core feature engineering strategies implemented to achieve RMSE 1.21721 in the sales prediction competition.

Dataset url:(https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)

Kaggle url:(https://www.kaggle.com/code/seanchenxinyu/baseline-with-xgb)

## Key Feature Categories

### 1. Temporal Features
- **Date Block Number**: 
  - Fundamental time unit representing month intervals (0-34)
  - Used as primary temporal axis for lag feature generation

- **Lag Features**:

### Generate Lagged Sales Features
```python
lags = [1, 2, 3] #Previous 1-3 months
create_lag_feature(df,lags,['item_cnt_month'])
```

- Created sliding window statistics for sales patterns
- 3-month lookback period optimized through experimentation

### 2. Entity Relationship Features
- **Shop Meta Features**:
- `shop_city`: Extracted from shop names using regex split
- `shop_category`: Derived from name patterns (e.g., "STC", "TRC")
- Ordinal encoded for tree model compatibility

- **Item Hierarchy Features**:
- **Category Breakdown**:

```python
item_cat_df['item_category_type'] = ... #Main category (e.g.,"Games")
item_cat_df['item_category_sub_type'] = ... # Sub-category (e.g. "PC")
```

- **Item Text Features**:
- Cleaned item names using regex normalization
- Split names into 3 components using bracket/parenthesis delimiters
- Ordinal encoded high-frequency (>20 occurrences) item types

### 3. Cross-Domain Interaction Features
- **Shop-Item Pair History**:

### Generate Complete Shop-item-month matrix

```python
matrix = []
for i in range(34):
   sales = raw_df[raw_df['date_block']==i]
   matrix.append(np.array(list(product[i], sales['shop_id'].unique(), sales['item_id'].unique()), dtype=np.int16))
```

- Ensures continuity for new shop-item combinations
- Prevents zero-inflation in validation/test sets

- **Revenue Signals**:


```python
raw_df['revenue'] = raw_df['item_cnt_day']*raw_df['item_price']
```

- Secondary target variable for potential multi-task learning
- Not used in final model but preserved for future experiments

### 4. Data Sanitization Features
- **Outlier Treatment**:


### Price and Sales quantity thresholds

```python
raw_df = raw_df[(raw_df['item_price']<300000) & (raw_df['item_cnt-day']<1000)]
raw_df = raw_df[raw_df['item_price']>0]
```

- 99.7th percentile cutoff for price/sales volume
- Post-prediction clipping (0-20) to match competition constraints

- **Missing Value Strategy**:
- Forward-fill for lag features (temporal continuity)
- Zero-imputation for new shop-item combinations

## Feature Encoding
| Feature Type          | Encoding Method          | Dimension | Notes                          |
|-----------------------|--------------------------|-----------|--------------------------------|
| Shop City             | Ordinal                  | 8-bit     | High-cardinality cities binned |
| Shop Category         | Ordinal                  | 8-bit     | Top-5 categories preserved    |
| Item Type             | Frequency-based Ordinal  | 16-bit    | 20+ occurrence threshold       |
| Category Hierarchy    | Nested Ordinal           | 8+16-bit  | Type-subtype relationships    |
| Temporal Features     | Raw Integer              | 8-bit     | Month-normalized              |

## Validation Strategy
- **Temporal Split**:
- Train: Months 4-32 (29 blocks)
- Validation: Month 33 (Strict future split)
- Test: Month 34 (Competition target)

- **Evaluation Metric**:

```python
np.sqrt(mean_squared_error(y_true,y_pred))
```

- Optimized for RMSE rather than RMSLE due to clipped target nature

## Performance Drivers
1. **Lag Feature Optimization**: 3-month window provided optimal balance between recency and stability
2. **Entity Relationship Modeling**: Shop-item pair matrix prevented cold-start issues
3. **Semantic Feature Extraction**: Name parsing uncovered latent categorical relationships
4. **Memory-Efficient Encoding**: 8-bit/16-bit types enabled full-history retention







# ========= V2 Predict with LSTM (RMSE 1.11770)

Kaggle url:(https://www.kaggle.com/code/seanchenxinyu/time-series-lstm-training-improvement)




# Introduction: Long Shor-Term Memory (LSTM) Architecture

- LSTM is a specialized recurrent neural network designed to capture long-range dependencies through gated mechanisms and memory cells . Its core components are:


# 1. Memory Cell ($$C_t$$)

The central innovation of LSTM, acting as a "conveyor belt" for long-term information storage: 

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
$$

- Function: Stores contextual information across arbitrary time intervals
- Key Properties:
   - Linear updates prevent gradient vanishing/exploding
   - Capacity grows with hidden layer size (typically 100-1000 units)
   - Retains information unless explicitly modified by gates


# 2. Gates Mechanisims

Three adaptive gates regulate information flow through sigmoid activations (σ) and element-wise multiplication (⊙):


# 2.1 Forget Gates ($$f_t$$)

Decides what to discard from previous cell state:

$$
f_t = \sigma(W_f \cdot h_{t-1}, x_t + b_f)
$$

- Inputs: Previous hidden state $$h_{t-1}$$ and current input $$x_t$$
- Weights: $$W_f$$ (gate-specific parameter matrix)
- Behavior:
   - 0 = "Completely forget"
   - 1 = "Completely retain"



# 2.2 Input Gate ($$i_t$$) & Candidate State ($$\tilde{C}_t$$)

Controls new information addition:

![](https://latex.codecogs.com/svg.image?\begin{align}i_t&=%20\sigma(W_i%20\cdot%20[h_{t-1},x_t]%20+%20b_i)%5C%5C\tilde{C}_t&=%20\tanh(W_C%20\cdot%20[h_{t-1},x_t]%20+%20b_C)\end{align})


- $$i_t$$: Sigmoid gate determines update magnitude
- $$\tilde{C}_t$$: Tanh-activated candidate values (-1 to 1)


# 2.3 Output Gate ($$o_t$$)

Determines exposed cell state to next layer:

![](https://latex.codecogs.com/svg.image?\begin{align}o_t&=%20\sigma(W_o%20\cdot%20[h_{t-1},x_t]%20+%20b_o)%5C%5Ch_t&=%20o_t%20\odot%20\tanh(C_t)\end{align})

- Final output combines filtered cell state and gate regulation


# 3. Complete Computational Flow

At each timestep $$t$$:

1. Gate Activations:
   - Compute $$f_t,i_t,o_t$$ via sigmoid transforms

2. State Updates:
   - Generate candidate $$\tilde{C}_t$$ with tanh
   - Update $$\tilde{C}_t$$ using forget/input gates

3. Output Generation:
   - Filter $$\tilde{C}_t$$ through output gate to get $$h_t$$




# 4. Mathematical Implementation

Full sequence processing equations:

![](https://latex.codecogs.com/svg.image?\begin{align}f_t&=%20\sigma(W_f%20\cdot%20[h_{t-1},x_t]%20+%20b_f)%5C%5Ci_t&=%20\sigma(W_i%20\cdot%20[h_{t-1},x_t]%20+%20b_i)%5C%5C\tilde{C}_t&=%20\tanh(W_C%20\cdot%20[h_{t-1},x_t]%20+%20b_C)%5C%5CC_t&=%20f_t%20\odot%20C_{t-1}%20+%20i_t%20\odot%20\tilde{C}_t%5C%5Co_t&=%20\sigma(W_o%20\cdot%20[h_{t-1},x_t]%20+%20b_o)%5C%5Ch_t&=%20o_t%20\odot%20\tanh(\tilde{C}_t)\end{align})


Dimissions:
 - $$W_* \in \mathbb{R}^{d_h \times (d_h + d_x)}$$ for hidden size $$d_h$$ and input size $$d_x$$
 - All gates share same hidden/input concatenation $$[h_{t-1},x_t]$$



# 5. Design Advantages 

- Gradient Stability: Cell state linearity prevents multiplicative gradient decay
- Contextual Awareness: Gates adaptively filter irrelevant features (e.g., noise)
- Multi-Scale Learning:
  - $$h_t$$ captures short-term patterns
  - $$C_t$$ preserves long-term trends


Benchmark Performance:

- 35% higher accuracy than vanilla RNNs on IMDB sentiment analysis
- 15%-20% lower prediction error than ARIMA in financial forecasting





# 1. Data Loading & Preprocess

### 1.1 Load Datasets


```python
import pandas as pd

# Load raw data

train_df = pd.read_csv('/kaggle/input/.../sales_train.csv') # Historical sales data
test_df = pd.read_csv('/kaggle/input/.../test.csv') # Test set with shop/item IDs

items_df = pd.read_csv('/kaggle/input/.../items.csv') # Item metadata

# Verify data integrity

print("Missing values in trainning data:", train_df.isna().sum()) # 0
print("Mising vales in test data:", test_df.isna().sum()) # 0
```

### 1.2 Temporal Alignment

```python
# Prepare test set for merging

test_Data = test_df.copy()

test_Data['date_block_num'] = 34 # Future prediction point (month 34)
test_Data['item_cnt_day'] = 0 #Placeholder column

# Reshape training data into (shop_id, item_id) x time matrix

train_Data = train_df.pivot_table(index=['shop_id', 'item_id'], columns='date_block_num', values='item_cnt_day', aggfunc='sum', fill_value=0) # Shape: (unique shop-item pairs) x 33 months (0-32)

# Merge test set with historical data
Combine_train_test = pd.merge(test_Data[['shop_id', 'item_id', 'date_block_num']], train_Data, on=['shop_id', 'item_id'], how='left').fillna(0)
```

---

## 2. Feature Engineering


### 2.1 Normalization


```Python
from sklearn.preprocessing import MinMaxScaler

# Keep only numerical features for scaling
numerical_data = Combine_train_test.iloc[:, 3:-1]   #Exclude metadata columns

#Scale to [0,1] range
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(numerical_data)
```

### 2.2 Sequence Creation 

```python

# Training data: Use months 0-32 to predict month 33

X_train = scaled_data[:,:33].reshape(-1,33,1) # Shape:(214200,33,1)
y_train = scaled_data[:,33] # Target values

# Teat data: Use months 1-33 to predict month 34

X_test = scaled_data[:,1:34].reshape(-1,33,1) # Shape: (214200,33,1)
```

---


## 3. LSTM Model


### 3.1 Architecture

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(33,1))) # First LSTM layer

model.add(LSTM(50, return_sequences=False)) # Second LSTM layer

model.add(Dense(1)) # Regression output

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.summary()
```

**Model Structure**:

**Layer (type) Output Shape Param**

LSTM 10400 (None,33,50)

LSTM 20200 (None,50)


**Dense (None,1)   51**

Total params: 30,651

### 3.2 Training

```python
history = model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1)
```


---


## 4. Prediction & Result


### 4.1 Inverse Transformation

```python

# Create dummy array for inverse scaling
dummy_array = np.zeros((X_test.shape[0],34))

dummy_array[:,1:34] = X_test.reshape(X_test.shape(0),33)

# Predict and inverse-transform

predicted = model.predict(X_test)

dummy_array[:,-1] = predicted.flatten()

final_predictions = scaler.inverse_transform(dummy_array)[:,-1]
```



### 4.2 Post-processing

```python

# Apply competition constraints

final_predictions = np.clip(final_predictions,0,30).round().astype(int)

# Generate submission

submission = pd.read_csv('.../sample_submission.csv')
submission['item_cnt_month'] = final_predictions
submission.to_csv('submission.csv',index=False)
```


### 4.3 Final Metric
**RMSE**: 1.23151 (Root Mean Squared Error on test set)

---

## 5. Key Implementation Notes

1. **Temporal Structure**:
   - Training sequences: 33 months (0-32) → Predict month 33
   - Test sequences: 33 months (1-33) → Predict month 34

2. **Data Shapes**:
   - Original training data: `(2,935,849, 5)` rows
   - Pivoted training data: `(34,250, 33)` (unique shop-item pairs x months)
   - Final input shape: `(214,200, 33, 1)` (test samples x sequence length x features)

3. **Critical Operations**:
   - Left join preserves all test samples (214,200 rows)
   - MinMax scaling prevents large-value dominance in LSTM
   - Sequence reshaping enables time-step processing






# ========= V3 Predict Future with Ultimate Feature Engineering & LightGBM (RMSE 0.83687)

Kaggle url:(https://www.kaggle.com/code/seanchenxinyu/future-sales-predictions)


# Introduction

This project demonstrates an end-to-end pipeline for predicting future sales volumes in a retail setting using a LightGBM regressor. We cover:

- Memory optimization for large tabular data.
- Advanced temporal and categorical feature engineering.
- Bag-of-words text features from product names.
- LightGBM training with hyperparameter tuning and early stopping.
- Model evaluation (RMSE) and feature importance analysis.


## Data Loading & Preprocessing

**1. Importing Libraries**

```python
import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
```

Ensure consistent random seeds for reproducibility:

```python
import random
import os
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
```

**2. Load CSV Files**

```python
items = pd.read_csv('data/items.csv')        # (22170, 3)
shops = pd.read_csv('data/shops.csv')        # (60, 2)
sales = pd.read_csv('data/sales_train.csv') # (~3 million rows)
test  = pd.read_csv('data/test.csv')        # (~214200 rows)
```

**3. Memory Optimization**

```reduce_mem_usage``` is designed to dramatically lower the RAM footprint of a pandas DataFrame by automatically downcasting column data types and encoding category-like columns. Its operation can be broken down into these steps:

- **1. Iterate over each column**
- **2. Numeric downcasting** which aims to shrink the integer width (e.g. int64 -> int32 -> int16, etc.) as long as it still fits the existing values.
  - **Floats:** After integer downcasting, any float columns are cast to float32 (half the size of float64).
- **3. Categorical encoding**
  - Non-numeric, non-datetime columns (typically object strings) are optionally factorized: each unique string is mapped to a small integer code.
  - This replaces bulky Python objects with simple int arrays.
- **4. Datetime and sparse columns**
- **5. Silent mode & reporting**

```python
def reduce_mem_usage(df, float_dtype='float32'):  
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast='integer')
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(float_dtype)
        else:
            df[col], _ = pd.factorize(df[col])
    return df
# Apply:
items = reduce_mem_usage(items)
shops = reduce_mem_usage(shops)
sales = reduce_mem_usage(sales)
```

Example: A 3 million-row sales table with mixed 64-bit types might shrink from ~4 GB to ~1.5 GB (≈60% reduction) after applying ```reduce_mem_usage```



**4. Data Cleaning**

```python
# Parse dates
sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')
# Merge duplicate shops per Kaggle discussion
sales['shop_id'] = sales['shop_id'].replace({0:57,1:58,11:10,40:39})
# Filter shops present in test set
valid_shops = test['shop_id'].unique()
sales = sales[sales['shop_id'].isin(valid_shops)].copy()
# Remove outliers
sales = sales[(sales['item_price'] > 0) & (sales['item_price'] < 50000)]
sales = sales[(sales['item_cnt_day'] > 0) & (sales['item_cnt_day'] < 1000)]
```

### Feature Engineering

### Overview of the Step:

Generate a master matrix of shape  where each row represents one (month, shop, item) and columns are features.

# 1.Base Matrix Construction:

```create_monthly_matrix```

```python
def create_monthly_matrix(sales, test):
    rows = []
    for block in sales['date_block_num'].unique():
        shops = sales[sales.date_block_num==block]['shop_id'].unique()
        items = sales[sales.date_block_num==block]['item_id'].unique()
        rows.append(list(itertools.product([block], shops, items)))
    matrix = pd.DataFrame(np.vstack(rows), columns=['date_block_num','shop_id','item_id'])
    # Aggregate monthly sales & revenue
grouped = sales.groupby(['date_block_num','shop_id','item_id']).agg(
    item_cnt_month=('item_cnt_day','sum'),
    item_revenue_month=('item_price','sum')
)
matrix = matrix.merge(grouped, how='left', on=['date_block_num','shop_id','item_id']).fillna(0)
# Append test month
... return matrix
```

- **Shape: ~4 million rows x features**

# 2.Categorical & Text Features

## 2.1 Fuzzy Name Groups

- Use ```fuzzywuzzy.partial_ratio``` to assign each ```item_name``` into ```item_name_group```.
- Threshold similarity = 65.

## 2.2 Artist / First Word

- For music items, extract artist name patterns (uppercase, double spaces).
- Else: first non-stopword token

## 2.3 Bag-of-Words on ```item_name```

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
vectorizer = CountVectorizer(stop_words=stopwords)
X_bow = vectorizer.fit_transform(items['item_name_clean'])
# Select top 50 words predictive of sales
selector = SelectKBest(f_regression, k=50)
selector.fit(X_bow, y)
```

# 3.Temporal Features

## 3.1 Time Differences & Averages

- ```month_first_day```,```month_last_day```,```month_length```
- ```first_shop_date```,```first_item_date```,```first_shop_item_date```
- Compute days active to drive ```item_cnt_day_avg```:


![](https://latex.codecogs.com/svg.image?%5Ctexttt%7Bitem%5C_cnt%5C_day%5C_avg%7D=%5Cfrac%7B%5Ctexttt%7Bitem%5C_cnt%5C_month%7D%7D%7Bmin(%5Ctexttt%7Bdays%5C_in%5C_shop%7D,%5Ctexttt%7Bmonth%5C_length%7D)%7D)

- Features: ```shop_age```,```item_age```,```new_item```,```last_shop_item_sale_days```,```month```


## 3.2 Price Trends

- Monthly mean price & normalized deviation per category.
- Lagged price: previous month values.


# 4.Rolling & Lag Features

## 4.1 Percent Change

Compute $$\frac{v_t - v_{t-k}}{v_{t-k}}$$ clamped to [-3,3] for k=1,12.


## 4.2 Rolling/Expanding/EWMA

```python
# Example: 12-month rolling mean of item_cnt_month per (shop_id,item_id)
matrix['shop_item_12m_mean'] = matrix.groupby(['shop_id','item_id'])['item_cnt_month'] \
    .rolling(window=12, min_periods=1).mean().reset_index(0,drop=True)
```

- Generate features for multiple windows (1,3,6,12 months) and group combinations.


## 4.3 Mean Encoding

```python
# Mean sales per item_id lagged by 1 month
mean_sales = matrix.groupby(['date_block_num','item_id'])['item_cnt_month'].mean().rename('item_mean')
mean_sales = mean_sales.reset_index()
mean_sales['date_block_num'] += 1
matrix = matrix.merge(mean_sales, on=['date_block_num','item_id'], how='left')
```



# Model Training & Evaluation

## 1. Data Split

```python
train = matrix[matrix.date_block_num < 33]
val   = matrix[matrix.date_block_num == 33]
X_train, y_train = train.drop(['item_cnt_month'], axis=1), train['item_cnt_month']
X_val, y_val     = val.drop(['item_cnt_month'], axis=1), val['item_cnt_month']
```

## 2. LightGBM Training Function

```python
import lightgbm as lgb
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 966,
    'learning_rate': 0.01,
    'n_estimators': 8000,
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'min_child_samples': 27,
    'early_stopping_round': 30
}
model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train,y_train),(X_val,y_val)],
    categorical_feature=['item_category_id','month'],
    verbose=100
)
```

- **Training RMSE** convergence log printed every 100 rounds.


## 3. Evaluation

- Compute RMSE on validation set:

```python
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_val, model.predict(X_val), squared=False)
print(f"Validation RMSE: {rmse:.4f}")
```

## 4. Feature Importance

```python
lgb.plot_importance(model, importance_type='gain', max_num_features=30)
plt.title('Top 30 Feature Importances');
```



# Prediction & Submission

```python
# Prepare test data for month 34
test_matrix = ...  # same features
preds = model.predict(test_matrix.drop('item_cnt_month', axis=1)).clip(0,20)
submission = pd.DataFrame({
    'ID': test['ID'],
    'item_cnt_month': preds
})
submission.to_csv('submission.csv', index=False)
```
histpo
# LightGBM Technical Overview

- **Gradient Boosting**: Fit trees on residual errors.
- **Histogram-based splitting**: Bin continuous features into discrete buckets.
- **Leaf-wise growth**: choose leaf with maximum split gain.
- **Regularization**: controlled via parameters ```lambda_l1```,```lambda_l2```,```min_gain_to_split```.


Regularization term:

![](https://latex.codecogs.com/svg.image?%5COmega(f)=%5Cgamma%20T&plus;%5Cfrac%7B1%7D%7B2%7D%5Clambda%5Csum_%7Bj=1%7D%5E%7BT%7Dw_j%5E2)


# Advantages on Small Datasets

1. **Quick Convergence**: Leaf-wise splitting captures complex patterns with fewer trees.
2. **Built-in Overfitting Control**: early stopping, ```min_child_samples```, L1/L2 regularization.
3. **Low Overhead**: histogram binning reduces computation and memory footprint.
4. **Native Categorical Handling**: automatically handles categorical features without one-hot encoding.



# Features Handling


- 1. Memory Optimization (reduce_mem_usage)

- 2.Base Matrix Construction (create_testlike_train)

- 3.Fuzzy Name Grouping (add_item_name_groups)

- 3.Artist or First-Word Feature (add_first_word_features)

- 4.Name Length Features

- 5.Time-Based Features (add_time_features)

- 6.Price Features (add_price_features)

- 7.Platform ID Mapping

- 8.Supercategory ID Mapping

- 9.City Code Encoding (add_city_codes)

- 10.Category Clustering (cluster_feature)

- 11.Shop Clustering (cluster_feature)

- 12.Unique Count Features (uniques)

- 13.Lagged Percent-Change Features (add_pct_change)

- 14.Rolling, Expanding & EWMA Window Features (add_rolling_stats)

- 15.Simple Lag Features (simple_lag_feature)

- 16.Mean-Encoding Features (create_apply_ME)

- 17.Bag-of-Words Text Features + SelectKBest (name_token_feats)






