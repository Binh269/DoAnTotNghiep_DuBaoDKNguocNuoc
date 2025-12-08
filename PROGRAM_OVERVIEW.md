# Tong Quan Chuong Trinh - Du Bao Nhu Cau Nuoc

Tai lieu tom tat nay mo ta cau truc, pipeline du lieu va thuat toan cua he thong.

## Muc Tieu
- Xay dung he thong du bao nhu cau nuoc sinh hoat dua tren chuoi thoi gian mo (Fuzzy Time Series)
- Toi uu tham so mo hinh (so khoang, trang thai markov lag-1) bang K-Means va Differential Evolution
- Giao dien truc tuyen bang Streamlit, du lieu luu trong SQL Server (SSMS)

## Kien Truc Tong Quan

- `app.py`: Giao dien Streamlit - lua chon nguon du lieu, tien xu ly, chay pipeline va hien thi bieu do
- `src/ket_noi_db.py`: Ket noi toi SQL Server (pyodbc) va cac ham nhap/xuat du lieu
- `src/xuly_du_lieu.py`: Doc du lieu, chuan hoa cot ngay/gia tri, resample (neu can) va moving average
- `src/mohinh/chuoi_thoi_gian_mo.py`: Trien khai mo hinh chuoi thoi gian mo (lag-1) voi ham membership Gaussian
- `src/opt/de.py`: Trien khai Differential Evolution de toi uu tham so
- `src/chinh_dieu_phoi_hop.py`: Pipeline chinh dieu phoi - chia train/val/test; chay pipeline 7 buoc va danh gia
- `tao_du_lieu_ao.py`: Sinh du lieu ao (theo tham so: nam bat dau, so ngay) va insert vao bang DuLieuNuocAo

## Du Lieu

- Lay truc tiep tu SQL Server (khong dung CSV trong pipeline chinh)
- Bang chinh: DuLieuNuoc (ID, NgayThang, LuongNuoc)
- Bang du lieu ao: DuLieuNuocAo (ID, NgayThang, LuongNuoc) - dung de test va demo
- Ho tro import: DuLieuNuocImport - import CSV/Excel

## Pipeline Du Lieu Va Mo Hinh (7 Buoc)

1. Doc du lieu tu SSMS bang lay_du_lieu_tu_db(table) - chuan hoa cot date va value
2. Tien xu ly (UI):
   - Chon phan giai: Daily hoac Monthly (resample/agg)
   - Lam muot bang Moving Average (cua so do nguoi dung chon)
3. Chia tap: Train 70%, Val 15%, Test 15%
4. K-Means tuy chon k:
   - Auto-select k dung Silhouette Score (k=2..12)
   - Tra ve so cum toi uu va tham lay tam cum
5. Tao khoang mor:
   - Khoi tao ranh gioi tu trung diem giua tam cum K-Means
   - Cac khoang A_0, A_1, ..., A_k-1
6. Fuzzify du lieu:
   - Chuyen gia tri thuc thanh membership vector bang ham Gaussian
   - Ghi nhan khoang phu hop nhat (argmax)
7. Xay luat Markov:
   - Xay dung luong suat/xac suat chuyen tiep A_t-1 -> A_t
   - Luu trong dict: quan_he[prev_state][next_state] = count hoac prob
8. Du bao + Defuzzify:
   - Tra cuu luat tu trang thai hien tai
   - Tinh weighted average cua tam khoang
   - Blend 30% mo hinh + 70% persistence
9. Danh gia:
   - Tinh RMSE, MSE, MAE, MAPE tren tap test

## Thuat Toan Chinh (Tom Tat Ky Thuat)

Fuzzification:
- Voi moi khoang co tam c_i
- Tinh membership mu_i(x) bang ham Gaussian voi sigma phu thuoc overlap

Rule Building:
- Voi tung cap (x_t-1, x_t), cong trong so membership vao luat
- Chuan hoa hang luat thanh phan phoi xac suat hau qua

Forecast:
- Tu trang thai hien tai (membership tai gia tri cuoi), ket hop luat de co phan phoi hau qua
- Defuzzify: y = sum(p_j * center_j)

Markov Lag-1:
- Trang thai hien tai la label cua gia tri cuoi cung
- Xac suat chuyen tiep tra tu bang tron

Differential Evolution:
- Toi uu cac ranh gioi khoang de giam MSE tren validation set
- Khong shuffle du lieu thoi gian

## Thuc Thi Va Tham So Chinh

- Chay UI: streamlit run app.py
- Tien xu ly:
  - Phan giai: Daily/Monthly
  - Moving Average window: 1-60
- Pipeline:
  - Auto k-selection bang Silhouette Score
  - Lag-1 Markov model
  - DE toi uu: popsize=10, iters=30

## Han Che Va Ghi Chu

- Mo hinh mo lag-1 don gian, phu hop voi du lieu co tinh Markov bac 1
- Blending 30/70 de tranh drift cao
- K-Means auto-select toi uu hoa cho dataset hien tai
- Ket noi DB dung pyodbc - hoat dong tot
- Toc do: DE co the cham tren nhieu ham/so vong; chay tren sample nho khi test

## Muon Mo Rong

- Thay DE bang Bayesian Optimization de toi uu hieu qua hon
- Mo rong sang lag-p cao hon
- Luu model va du bao dinh ky vao DB

---

Xem huong dan nhanh va vi du trong README.md
Xem chi tiet tung buoc trong QUY_TRINH_CHI_TIET.md
Xem cau truc trong HUONG_DAN_CAU_TRUC.md
