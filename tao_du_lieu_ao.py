"""
Script sinh dữ liệu ảo và insert vào bảng DuLieuNuocAo trong SSMS.

Chạy: python tao_du_lieu_ao.py
Hoặc: gọi hàm tao_va_insert_du_lieu_ao() từ các module khác
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pyodbc
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def tao_va_insert_du_lieu_ao(server='BOSS\\SQLEXPRESS', database='DuDoanSuDungNuoc',
                              table='DuLieuNuocAo', num_days=1095, 
                              nam_bat_dau=2021, thang_bat_dau=1, seed=42):
    """Sinh dữ liệu ảo và insert vào bảng DuLieuNuocAo.
    
    Tham số:
    - num_days: số ngày dữ liệu (mặc định 1095 = 3 năm)
    - nam_bat_dau: năm bắt đầu (mặc định 2021)
    - thang_bat_dau: tháng bắt đầu (mặc định 1 = tháng 1)
    - seed: seed ngẫu nhiên để có kết quả lặp lại
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    try:
        # Kết nối SSMS
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID=sa;'
            f'Trusted_Connection=yes;'
        )
        cursor = conn.cursor()
        
        print(f'[INFO] Sinh du lieu ao ({num_days} ngay, tu thang {int(thang_bat_dau)}/{int(nam_bat_dau)})')
        
        # Sinh dữ liệu ảo
        start_date = f'{int(nam_bat_dau)}-{int(thang_bat_dau):02d}-01'
        idx = pd.date_range(start=start_date, periods=num_days, freq='D')
        t = np.arange(num_days)
        
        # Mô hình: base + trend + seasonal + noise
        seasonal = 15.0 * np.sin(2 * np.pi * t / 365.0)
        trend = 0.08 * t
        base = 110.0
        noise = np.random.normal(scale=4.0, size=num_days)
        values = base + trend + seasonal + noise
        values = np.maximum(values, 10)  # Đảm bảo giá trị dương
        
        df = pd.DataFrame({
            'NgayThang': idx,
            'LuongNuoc': values
        })
        
        print(f'[INFO] Kiem tra bang {table}, xoa du lieu cu...')
        cursor.execute(f'DELETE FROM [{table}]')
        cursor.commit()
        
        print(f'[INFO] Them {len(df)} dong vao bang {table}...')
        
        # Insert từng dòng
        for idx_row, row in df.iterrows():
            cursor.execute(
                f'INSERT INTO [{table}] (NgayThang, LuongNuoc) VALUES (?, ?)',
                (row['NgayThang'], float(row['LuongNuoc']))
            )
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f'[OK] Da them {len(df)} dong thanh cong!')
        print(f'    Khoang: {df["NgayThang"].min().strftime("%Y-%m-%d")} den {df["NgayThang"].max().strftime("%Y-%m-%d")}')
        print(f'    Luong nuoc: min={df["LuongNuoc"].min():.2f}, max={df["LuongNuoc"].max():.2f}, mean={df["LuongNuoc"].mean():.2f}')
        
    except Exception as e:
        print(f'[ERROR] {e}')
        print('\nKiem tra:')
        print('1. Server: BOSS\\SQLEXPRESS')
        print('2. Database: DuDoanSuDungNuoc')
        print('3. Bang: DuLieuNuocAo (co cot: ID, NgayThang, LuongNuoc)')
        print('4. ODBC Driver 17 da cai dat?')


def tao_bang_neu_chua_co(server='BOSS\\SQLEXPRESS', database='DuDoanSuDungNuoc',
                          table='DuLieuNuocAo'):
    """Tạo bảng DuLieuNuocAo nếu chưa tồn tại."""
    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID=sa;'
            f'Trusted_Connection=yes;'
        )
        cursor = conn.cursor()
        
        # Kiểm tra bảng đã tồn tại chưa
        cursor.execute(
            f"SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME='{table}'"
        )
        exists = cursor.fetchone()[0]
        
        if not exists:
            print(f'[INFO] Bang {table} chua ton tai, tao moi...')
            cursor.execute(f'''
                CREATE TABLE [{table}] (
                    ID INT PRIMARY KEY IDENTITY(1,1),
                    NgayThang DATETIME NOT NULL,
                    LuongNuoc FLOAT NOT NULL
                )
            ''')
            cursor.commit()
            print(f'[OK] Da tao bang {table} thanh cong')
        else:
            print(f'[OK] Bang {table} da ton tai')
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f'[ERROR] Khong the tao bang: {e}')


if __name__ == '__main__':
    print('=' * 60)
    print('Script tao du lieu ao va insert vao SSMS')
    print('=' * 60)
    
    # Tạo bảng nếu cần
    tao_bang_neu_chua_co()
    
    # Sinh dữ liệu ảo (mặc định: 3 năm từ 2021)
    tao_va_insert_du_lieu_ao(num_days=1095, nam_bat_dau=2021, thang_bat_dau=1)
