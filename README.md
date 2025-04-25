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
