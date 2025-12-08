import pandas as pd

df = pd.read_csv("CSV/evn.csv")

df_new = df.iloc[:, [0, 3]]

# Đặt tên cột
df_new.columns = ["date", "value"]

df_new["date"] = df_new["date"].str.split(" ").str[0]

df_new.to_csv("CSV/DuLieuTuyenQuangDaXuLy.csv", index=False, encoding="utf-8-sig")

print("✔️ File DuLieuTuyenQuangDaXuLy.csv đã tạo xong.")
