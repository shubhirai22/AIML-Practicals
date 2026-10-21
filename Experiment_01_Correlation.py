
import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "Hours_Studied": [2,3,4,5,6,7,8,9,10,11],
    "Attendance": [60,65,70,72,75,80,82,85,90,92],
    "Assignments": [4,5,5,6,6,7,8,8,9,10],
    "Marks": [45,50,55,60,63,68,72,78,84,88]
})

corr = data.corr(numeric_only=True)

print("Correlation Matrix:")
print(corr)

print("\nCorrelation with Marks:")
print(corr["Marks"].sort_values(ascending=False))

plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
