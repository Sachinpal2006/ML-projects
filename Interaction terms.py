import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# Example data for demonstration
data = {
	'Hours': [1, 2, 3, 4, 5],
	'Sleep': [6, 7, 8, 9, 10],
	'Marks': [50, 60, 70, 80, 90]
}
df = pd.DataFrame(data)

X = df[['Hours', 'Sleep']]
y = df['Marks']
y = df['Marks']

poly = PolynomialFeatures(degree=2,interaction_only=True)

X_interaction = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_interaction,y)

print(model.predict(poly.transform([[3,7]])))