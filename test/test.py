import polars as pl

labels = pl.read_csv("output/ramp_labels.csv")

print("ramp_labels 总行数：", labels.shape[0])
print("\n===== 爬坡标签 dir 分布 =====")
print(labels["dir"].value_counts())
print("\ndir 所有唯一取值：")
print(labels["dir"].unique().to_list())

print("\n===== 时间连续性检查 =====")
print(labels.select(["t_idx","Day","step"]).head(8))
print(labels.select(["t_idx","Day","step"]).tail(8))
