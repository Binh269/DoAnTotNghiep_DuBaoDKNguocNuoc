import pyodbc
import pandas as pd
import warnings

# Suppress pandas warning about pyodbc connections
warnings.filterwarnings('ignore', message='.*pandas only supports SQLAlchemy.*')

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


def nhap_du_lieu_vao_db(df, server='BOSS\\SQLEXPRESS', database='DuDoanSuDungNuoc', 
                         table='DuLieuNuocImport', col_date='NgayThang', col_value='LuongNuoc'):
    """Import dữ liệu từ DataFrame vào bảng SQL.
    
    Args:
        df: DataFrame với cột 'date' và 'value'
        server: Tên SQL Server
        database: Tên database
        table: Tên bảng (mặc định: DuLieuNuocImport)
        col_date: Tên cột ngày tháng trong DB
        col_value: Tên cột giá trị trong DB
    
    Returns:
        Số dòng đã insert thành công
    """
    try:
        conn = pyodbc.connect(
            f'DRIVER={{ODBC Driver 17 for SQL Server}};'
            f'SERVER={server};'
            f'DATABASE={database};'
            f'UID=sa;' 
            f'Trusted_Connection=yes;'  
        )
        cursor = conn.cursor()
    except Exception as e:
        raise Exception(f'Lỗi kết nối SSMS: {e}')
    
    try:
        # Xóa dữ liệu cũ trong bảng (tùy chọn)
        cursor.execute(f'TRUNCATE TABLE [{table}]')
        
        # Chuẩn bị dữ liệu
        df_import = df.copy()
        # Thử đoán định dạng ngày: dayfirst=True cho định dạng DD/MM/YYYY (Việt Nam)
        try:
            df_import['date'] = pd.to_datetime(df_import['date'], dayfirst=True)
        except Exception:
            # Nếu thất bại, thử định dạng ISO8601 hoặc để pandas tự đoán
            try:
                df_import['date'] = pd.to_datetime(df_import['date'], format='ISO8601')
            except Exception:
                df_import['date'] = pd.to_datetime(df_import['date'], infer_datetime_format=True)
        df_import['value'] = pd.to_numeric(df_import['value'], errors='coerce')
        df_import = df_import.dropna(subset=['value'])
        
        # Insert từng dòng
        count = 0
        for idx, row in df_import.iterrows():
            date_val = row['date'].strftime('%Y-%m-%d')
            value_val = float(row['value'])
            insert_sql = f"INSERT INTO [{table}] ([{col_date}], [{col_value}]) VALUES ('{date_val}', {value_val})"
            cursor.execute(insert_sql)
            count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        return count
        
    except Exception as e:
        cursor.close()
        conn.close()
        raise Exception(f'Lỗi khi insert dữ liệu vào bảng {table}: {e}')


if __name__ == '__main__':
    # Test nhanh
    try:
        df = lay_du_lieu_tu_db()
        print(f'✓ Đã lấy {len(df)} dòng từ SSMS')
        print(df.head())
    except Exception as e:
        print(f'❌ {e}')
