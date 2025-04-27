# Future Sales Prediction Feature Engineering (RMSE: 1.21721)

This document details the core feature engineering strategies implemented to achieve RMSE 1.21721 in the sales prediction competition.

Dataset url:(https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/data)

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


## 1. Data Loading & Preprocessing

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

``
