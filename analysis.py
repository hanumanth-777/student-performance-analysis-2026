import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("students.csv")

# 1. Display first 5 records
print("\nFIRST 5 RECORDS")
print(df.head())

# 2. Display last 5 records
print("\nLAST 5 RECORDS")
print(df.tail())

# 3. Dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# 4. Dataset information
print("\nDATASET INFORMATION")
df.info()

# 5. Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())
print("\nMISSING VALUES")
print(df.isnull().sum())
df["SQL"] = df["SQL"].fillna(df["SQL"].mean())
print("\nDUPLICATE RECORDS")
print(df.duplicated().sum())
df = df.drop_duplicates()
print("\nAFTER DATA CLEANING")
print("Missing values:")
print(df.isnull().sum())

print("Duplicate records:", df.duplicated().sum())

print("Final dataset shape:", df.shape)
print(df)
print("\nSUBJECT-WISE AVERAGES")

subjects = ["Python", "SQL", "Mathematics"]

print(df[subjects].mean())
df["Total_Marks"] = df[subjects].sum(axis=1)
df["Percentage"] = (df["Total_Marks"] / 300) * 100
topper = df.loc[df["Total_Marks"].idxmax()]

print("\nTOPPER DETAILS")
print(topper)
class_average = df["Percentage"].mean()

print("\nCLASS AVERAGE:", round(class_average, 2))
df["Rank"] = df["Total_Marks"].rank(
    ascending=False,
    method="dense"
)

df = df.sort_values("Rank")

print("\nSTUDENT RANKING")
print(df[["Name", "Total_Marks", "Percentage", "Rank"]])
# Pass/Fail Analysis

df["Result"] = np.where(
    (df["Python"] >= 40) &
    (df["SQL"] >= 40) &
    (df["Mathematics"] >= 40),
    "Pass",
    "Fail"
)

print("\nPASS/FAIL RESULTS")
print(df[["Name", "Total_Marks", "Result"]])

print("\nRESULT COUNTS")
print(df["Result"].value_counts())
total_students = len(df)

passed_students = (df["Result"] == "Pass").sum()

pass_percentage = (passed_students / total_students) * 100

print("\nPASS PERCENTAGE:", round(pass_percentage, 2), "%")
df["Attendance_Status"] = np.where(
    df["Attendance"] < 75,
    "Low Attendance",
    "Good Attendance"
)

print("\nATTENDANCE ANALYSIS")
print(df[["Name", "Attendance", "Attendance_Status"]])

print("\nATTENDANCE COUNTS")
print(df["Attendance_Status"].value_counts())
subjects = ["Python", "SQL", "Mathematics"]

subject_averages = df[subjects].mean()

plt.figure(figsize=(8, 5))

plt.bar(subject_averages.index, subject_averages.values)

plt.title("Subject-wise Average Marks")
plt.xlabel("Subjects")
plt.ylabel("Average Marks")

plt.savefig("subject_average.png")

plt.show()
plt.figure(figsize=(9, 5))

plt.bar(df["Name"], df["Percentage"])

plt.title("Student Performance")
plt.xlabel("Students")
plt.ylabel("Percentage")

plt.xticks(rotation=45)

plt.savefig("student_performance.png")

plt.show()
result_counts = df["Result"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    result_counts.values,
    labels=result_counts.index,
    autopct="%1.1f%%"
)

plt.title("Pass/Fail Distribution")

plt.savefig("pass_fail_distribution.png")

plt.show()
df.to_csv("cleaned_students.csv", index=False)

print("\nCleaned dataset exported successfully!")
