import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi  
import pandas as pd

api_key = "KGAT_e8f00f8e7a07bf425e34ed96e848ccf6"
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)

with open(os.path.expanduser("~/.kaggle/access_token"), "w") as f:
    f.write(api_key)

api = KaggleApi()
api.authenticate()

COMPETITION_ID = "ecom-90025-2026-sm-2-ada-assignment-and-practice"

DESTINATION_PATH = "./data"
api.competition_download_files(competition=COMPETITION_ID, path=DESTINATION_PATH)

zip_filepath = os.path.join(DESTINATION_PATH, f"{COMPETITION_ID}.zip")
with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
    zip_ref.extractall(DESTINATION_PATH)
os.remove(zip_filepath)

print(f"Files extracted successfully to '{DESTINATION_PATH}' directory.")

# Read data
train = pd.read_csv("./data/train_data.csv")
test = pd.read_csv("./data/test_data.csv")

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
submission = pd.read_csv("./data/submission.csv")
submission["Y"] = test_predictions
submission.to_csv("./data/submission_final.csv", index=False)

api.competition_submit(
    file_name="./data/submission_final.csv",
    message="submission_Assign1",
    competition=COMPETITION_ID
)

subs = api.competition_submissions(COMPETITION_ID)
for s in subs:
    print(vars(s))
