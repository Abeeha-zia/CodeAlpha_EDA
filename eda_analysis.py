import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("books_dataset_5_pages.csv")
print("Dataset loaded successfully!")
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())


print("\nStatistical Summary:")
print(df.describe())

print("\nAverage Price:")
print(df["Price"].mean())

print("\nMinimum Price:")
print(df["Price"].min())

print("\nMaximum Price:")
print(df["Price"].max())

print("\nMedian Price:")
print(df["Price"].median())


print("\nRating Distribution:")
print(df["Rating"].value_counts())

print("\nPrice Category Distribution:")
print(df["Price Category"].value_counts())

print("\nAvailability Distribution:")
print(df["Availability"].value_counts())

plt.figure(figsize=(8, 5))

plt.boxplot(df["Price"])

plt.title("Price Distribution and Outlier Detection")
plt.ylabel("Price")

plt.tight_layout()

plt.savefig("price_outliers.png")

plt.show()


plt.figure(figsize=(8, 5))

plt.hist(df["Price"], bins=10)

plt.title("Distribution of Book Prices")
plt.xlabel("Price")
plt.ylabel("Number of Books")

plt.tight_layout()

plt.savefig("price_distribution.png")

plt.show()



rating_counts = df["Rating"].value_counts()

plt.figure(figsize=(8, 5))

rating_counts.plot(kind="bar")

plt.title("Distribution of Book Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Books")

plt.tight_layout()

plt.savefig("rating_distribution.png")

plt.show()

category_counts = df["Price Category"].value_counts()

plt.figure(figsize=(8, 5))

category_counts.plot(kind="bar")

plt.title("Books by Price Category")
plt.xlabel("Price Category")
plt.ylabel("Number of Books")

plt.tight_layout()

plt.savefig("price_category_eda.png")

plt.show()


rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

df["Rating Numeric"] = df["Rating"].map(rating_map)


correlation = df["Price"].corr(df["Rating Numeric"])

print("\nPrice-Rating Correlation:")
print(correlation)

plt.figure(figsize=(8, 5))

plt.scatter(df["Price"], df["Rating Numeric"])

plt.title("Relationship Between Book Price and Rating")
plt.xlabel("Price")
plt.ylabel("Rating")

plt.tight_layout()

plt.savefig("price_rating_relationship.png")

plt.show()