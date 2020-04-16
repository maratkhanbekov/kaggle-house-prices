# Regression for predicting the house sales price

Two interesting functions were created to solve this problem effectively.

#### Box-cox transformation

Proper use of statistical methods implies that variables have normal distribution. In practice, however, we rarely observe the normality of data.
So we need to do something with skewed variables.
One of the solution is using Box-Cox transformation:

<p align="center"> 
    <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/b565ae8f1cce1e4035e2a36213b8c9ce34b5029d">
</p>
Transformation is pretty simple, however we need to know which lambda parameter to use. Here comes our function.
For every skewed variable in dataframe this function finds best lambda parameter and transform variable using the box-cox formula

```python
from scipy.special import boxcox1p

def boxcox_transform_selective(df):
    float_vars = df.select_dtypes('float')
    for fv in float_vars:
        skew_value = skew(df[fv])
        if skew_value > 0.5:
            sk = []
            for lam in range(-100, 100, 1):
                sk.append([abs(boxcox1p(df[fv], lam/1000).skew()), lam/1000])
            lmbda = min(sk, key=lambda x: x[0])[1]
            df[fv] = boxcox1p(df[fv], lmbda)
    return df
```

#### Semi-auto feature type definition

Initial feature format definition can save bunch of time on the next stages of data processing.

```python
def dtypes_selection(df, id_col):
    st1 = df.nunique().reset_index(name='count_unique')
    st2 = df.describe().T.reset_index()
    st3 = round(df.isna().sum()/df.shape[0], 2).reset_index(name='p_miss')
    st = st1.merge(st2, on='index', how='left').merge(st3, on='index', how='left')
    st.fillna(-1, inplace=True)

    conds = [
        st['count'] == -1,
        st['count_unique'] == 2,
        st['index']==id_col
    ]

    res = [
        'object',
        'bool',
        'int64'
    ]
    st['dtype'] = np.select(conds, res, default='float64')
    return st

# Firstly we read the data and use python types intepretator
df_train = pd.read_csv('train.csv')

# Then we calculate stats for every feature and make a dictinary
st = dtypes_selection(df_train, 'Id')
dtypes = {col: dtype for col, dtype in st[['index', 'dtype']].values}

# Now we read again data but with our own types
df_train = pd.read_csv('train.csv', dtype=dtypes)
```
