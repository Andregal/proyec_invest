import pandas as pd
df=pd.read_csv(r'C:\Users\full_\OneDrive\Escritorio\Tesis 2 Version 2\Libro1.csv')
print(df.head)

import statsmodels.formula.api as smf
reg = smf.ols('NBI ~ CC + NOC + NOM + IPM',data=df)
res = reg.fit()
print(res.summary())
print(res.rsquared)
print(res.params)