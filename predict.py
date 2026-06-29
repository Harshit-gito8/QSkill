import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("Housing.csv")

# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df.drop("price", axis=1)
y = df["price"]

categorical = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder="passthrough"
)

# ==========================================================
# SPLIT DATA
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================================
# MODEL
# ==========================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==========================================================
# FUNCTIONS
# ==========================================================

def dataset_head():
    print("\n========== FIRST 10 RECORDS ==========\n")
    print(df.head(10))


def dataset_info():
    print("\n========== DATASET INFO ==========\n")
    df.info()


def dataset_statistics():
    print("\n========== DATASET STATISTICS ==========\n")
    print(df.describe())


def missing_values():
    print("\n========== MISSING VALUES ==========\n")
    print(df.isnull().sum())


def model_performance():

    print("\n========== MODEL PERFORMANCE ==========\n")

    print("R² Score :", r2_score(y_test, predictions))
    print("MAE      :", mean_absolute_error(y_test, predictions))
    print("MSE      :", mean_squared_error(y_test, predictions))
    print("RMSE     :", np.sqrt(mean_squared_error(y_test, predictions)))


def correlation_heatmap():

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(8,6))

    plt.imshow(corr, cmap="coolwarm")

    plt.colorbar()

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.show()


def area_vs_price():

    plt.figure(figsize=(8,6))

    plt.scatter(df["area"], df["price"])

    plt.xlabel("Area")

    plt.ylabel("Price")

    plt.title("Area vs Price")

    plt.grid(True)

    plt.show()


def actual_vs_predicted():

    plt.figure(figsize=(8,6))

    plt.scatter(y_test, predictions)

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )

    plt.xlabel("Actual Price")

    plt.ylabel("Predicted Price")

    plt.title("Actual vs Predicted Prices")

    plt.grid(True)

    plt.show()


def predict_new_house():

    print("\n========== ENTER HOUSE DETAILS ==========\n")

    area = float(input("Area : "))
    bedrooms = int(input("Bedrooms : "))
    bathrooms = int(input("Bathrooms : "))
    stories = int(input("Stories : "))

    mainroad = input("Main Road (yes/no) : ").lower()
    guestroom = input("Guest Room (yes/no) : ").lower()
    basement = input("Basement (yes/no) : ").lower()
    hotwaterheating = input("Hot Water Heating (yes/no) : ").lower()
    airconditioning = input("Air Conditioning (yes/no) : ").lower()

    parking = int(input("Parking Spaces : "))

    prefarea = input("Preferred Area (yes/no) : ").lower()

    print("\nFurnishing Status")
    print("1. Furnished")
    print("2. Semi-Furnished")
    print("3. Unfurnished")

    choice = input("Choose option : ")

    if choice == "1":
        furnishingstatus = "furnished"
    elif choice == "2":
        furnishingstatus = "semi-furnished"
    else:
        furnishingstatus = "unfurnished"

    new_house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    predicted_price = model.predict(new_house)

    print("\n======================================")
    print("Predicted House Price : {:.2f}".format(predicted_price[0]))
    print("======================================\n")


# ==========================================================
# MENU
# ==========================================================

while True:

    print("\n========================================")
    print("      HOUSE PRICE PREDICTION SYSTEM")
    print("========================================")
    print("1. View First 10 Records")
    print("2. Dataset Information")
    print("3. Dataset Statistics")
    print("4. Check Missing Values")
    print("5. Model Performance")
    print("6. Predict Price of New House")
    print("7. Correlation Heatmap")
    print("8. Area vs Price Graph")
    print("9. Actual vs Predicted Graph")
    print("0. Exit")
    print("========================================")

    option = input("Enter your choice: ")

    if option == "1":
        dataset_head()

    elif option == "2":
        dataset_info()

    elif option == "3":
        dataset_statistics()

    elif option == "4":
        missing_values()

    elif option == "5":
        model_performance()

    elif option == "6":
        predict_new_house()

    elif option == "7":
        correlation_heatmap()

    elif option == "8":
        area_vs_price()

    elif option == "9":
        actual_vs_predicted()

    elif option == "0":
        print("\nThank You for using House Price Prediction System.")
        break

    else:
        print("\nInvalid Choice! Please try again.")