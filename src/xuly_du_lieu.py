import pandas as pd
import numpy as np
import os


def lay_du_lieu_tu_db(server='BOSS\\SQLEXPRESS', database='DuDoanSuDungNuoc', 
                       table='DuLieuNuoc', col_date='NgayThang', col_value='LuongNuoc'):
    from ket_noi_db import lay_du_lieu_tu_db as _lay_tu_db
    return _lay_tu_db(server=server, database=database, table=table, 
                      col_date=col_date, col_value=col_value)


def doc_du_lieu(table='DuLieuNuoc', server='BOSS\\SQLEXPRESS', database='DuDoanSuDungNuoc',
                col_date='NgayThang', col_value='LuongNuoc'):
    return lay_du_lieu_tu_db(server=server, database=database, table=table,
                             col_date=col_date, col_value=col_value)


def tien_xu_ly(df_goc, luu_phan_giai='D', lam_tron=True, cua_so_ma=7):
    """Tiền xử lý dữ liệu:

    - df_goc: DataFrame có cột `date` và `value`.
    - luu_phan_giai: 'D' cho daily, 'M' cho monthly (resample nếu cần)
    - lam_tron: nếu True thì áp dụng moving average
    - cua_so_ma: kích thước cửa sổ MA

    Trả về df_xuly có cột `date`, `value` (sau xử lý) và `ma` (moving average).
    """
    df = df_goc.copy()
    df = df.set_index('date')
    if luu_phan_giai is not None:
        df = df.resample(luu_phan_giai if luu_phan_giai != 'M' else 'ME').mean()
    df['value'] = df['value'].ffill().bfill()
    if lam_tron:
        df['ma'] = df['value'].rolling(window=cua_so_ma, min_periods=1, center=False).mean()
    else:
        df['ma'] = df['value']
    df = df.reset_index()
    return df


