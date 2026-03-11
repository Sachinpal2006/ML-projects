import pandas as pd

data = {'City': ['Delhi','Mumbai','Bengalore'],'Price': [1000,2000,3000]}
df = pd.DataFrame(data)

print("Original Data:")
print(df)

df_dummies = pd.get_dummies(df, columns=['City'], drop_first=True)

print("\nDummy Variable Data:")
print(df_dummies)

