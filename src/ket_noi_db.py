import pyodbc
import pandas as pd


def lay_du_lieu_tu_db(server='BOSS\\SQLEXPRESS', database='DuDoanSuDungNuoc', 
                       table='DuLieuNuoc', col_date='NgayThang', col_value='LuongNuoc'):
    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID=sa;' 
            f'Trusted_Connection=yes;'  
        )
    except Exception as e:
        raise Exception(
            f'Lỗi kết nối SSMS: {e}\n'
            f'Kiểm tra: server={server}, database={database}, table={table}'
        )
    
    try:
        query = f'SELECT [{col_date}], [{col_value}] FROM [{table}] ORDER BY [{col_date}]'
        df = pd.read_sql(query, conn)
        conn.close()
        
        if df.empty:
            raise Exception(f'Bảng {table} trống. Vui lòng thêm dữ liệu vào bảng hoặc chạy script insert_du_lieu_mau.py để thêm dữ liệu mẫu.')
        
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value']) 
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    except Exception as e:
        raise Exception(f'Lỗi khi truy vấn bảng {table}: {e}')


if __name__ == '__main__':
    # Test nhanh
    try:
        df = lay_du_lieu_tu_db()
        print(f'✓ Đã lấy {len(df)} dòng từ SSMS')
        print(df.head())
    except Exception as e:
        print(f'❌ {e}')
