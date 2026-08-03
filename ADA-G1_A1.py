
import pandas as pd

train = pd.read_csv("train_data.csv")
test = pd.read_csv("test_data.csv")

# report the correlations between all explanatory variables with the dependent variable
correlations = (
    train
    .drop(columns=["ID"])
    .corr()["Y"]
    .drop("Y")
    .sort_values(ascending=False)
)

print(correlations)

# report the chosen scatter plot of explanatory variable with the dependent variable
import matplotlib.pyplot as plt

plt.scatter(train['X34'], train['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X34 vs Y')
plt.show()

plt.scatter(train['X21'], train['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X21 vs Y')
plt.show()

plt.scatter(train['X48'], train['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X48 vs Y')
plt.show()

# predict the dependent variable using the chosen explanatory variable x34
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train[['X34']], train['Y'])

# use estimated model to predict the dependent variable for the test data
print("β0 =", model.intercept_)
print("β1 =", model.coef_[0])

# provide the R square for the estimated model
print("R square =", model.score(train[['X34']], train['Y']))

# predict Y with the estimated betas
test_predictions = model.predict(test[['X34']])


# save the predictions to a CSV file
submission = pd.read_csv("submission.csv")
submission["Y"] = test_predictions
submission.to_csv("submission_final.csv", index=False)
