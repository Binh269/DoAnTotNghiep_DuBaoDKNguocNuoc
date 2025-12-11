import pandas as pd

df = pd.read_csv("CSV/ThuyDienQuangTriCraw.csv")
df_new = df.iloc[:, [0, 3]]
df_new.columns = ["date", "value"]
df_new = df_new.dropna()
df_new["date"] = df_new["date"].str.split(" ").str[0]

# Xuất file
df_new.to_csv("CSV/DuLieuThuyDienQuangTriDaXuLy.csv", index=False, encoding="utf-8-sig")

print("✔️ File DuLieuThuyDienQuangTriDaXuLy.csv đã tạo xong (đã loại bỏ NaN).")
