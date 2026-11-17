import polars as pl
import matplotlib.pyplot as plt

file_path = r"./data/wtbdata_245days.csv"
df = pl.read_csv(file_path)

# 筛选 1号机组
df_t1 = df.filter(pl.col("TurbID") == 1)

print("1号机组数据量：", df_t1.shape)
print(df_t1.head())

plt.figure(figsize=(14,4))
plt.plot(df_t1["Patv"])
plt.title("Wind Power Time Series (TurbID=1, Patv)")
plt.ylabel("Active Power Patv")
plt.tight_layout()
plt.show()
