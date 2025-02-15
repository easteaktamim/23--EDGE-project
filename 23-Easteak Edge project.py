import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv('Salary Data.csv')

X = df[['Experience Years']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.2, random_state=2)


model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

print(mean_absolute_error(y_test,y_pred))
print(f'{(mean_absolute_percentage_error(y_test,y_pred)*100):.2f}%')
print(r2_score(y_test,y_pred))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


ax1.scatter(X_test, y_test, label="Actual Data")
ax1.plot(X_test, y_pred, c='r', label="Predictions")
ax1.set_xlabel("X_test")
ax1.set_ylabel("y_test")
ax1.set_title("Scatter Plot")
ax1.legend()


ax2.bar(['MAE/1000','MEA%','R2_score'], [mean_absolute_error(y_test,y_pred)/1000,(mean_absolute_percentage_error(y_test,y_pred)*100),r2_score(y_test,y_pred)], label="Bar Data", color=['r','lime','b'])  # Use X_test for x-axis if appropriate
ax2.set_xlabel("X_test") 
ax2.set_ylabel("Bar Data Values") 
ax2.set_title("Bar Plot")
ax2.legend()

plt.show()

