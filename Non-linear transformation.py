from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
# Sample data for demonstration
data = {'Hours': [1, 2, 3, 4, 5], 'Marks': [50, 55, 65, 70, 80]}
df = pd.DataFrame(data)

poly = PolynomialFeatures(degree=2)
poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(df[['Hours']])

model = LinearRegression()
model.fit(X_poly,df['Marks'])

print (model.predict(poly.transform([[6]])))