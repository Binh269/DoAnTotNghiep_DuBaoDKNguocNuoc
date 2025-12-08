import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import numbers
import json
import pandas as _pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from chinh_dieu_phoi_hop import chay_pipeline_7_buoc
from xuly_du_lieu import doc_du_lieu, tien_xu_ly


def _make_serializable(obj):
    """Convert common numpy/pandas objects to plain Python types for JSON display."""
    try:
        if obj is None:
            return None
        if isinstance(obj, (str, bool, int, float)):
            return obj
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, _pd.Series):
            return _make_serializable(obj.tolist())
        if isinstance(obj, _pd.DataFrame):
            return _make_serializable(obj.to_dict(orient='records'))
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, numbers.Number):
            return float(obj)
        # fallback to string
        return str(obj)
    except Exception:
        return str(obj)

st.set_page_config(page_title='Hệ thống dự báo nhu cầu nước (Chuỗi thời gian mờ + DE)', layout='wide')
st.title('📊 Dự báo nhu cầu nước — Chuỗi thời gian mờ + DE')

if 'db_loaded' not in st.session_state:
    st.session_state['db_loaded'] = False
    st.session_state['df_xuly'] = None
    st.session_state['tien_xu_ly_done'] = False
    st.session_state['kq_pso'] = None
    st.session_state['current_source'] = None

st.sidebar.header('📁 Chọn dữ liệu')
option = st.sidebar.radio('Nguồn dữ liệu', ['Dữ liệu thực tế', 'Dữ liệu import', 'Dữ liệu ảo'], key='source_radio')

if st.session_state['current_source'] != option:
    st.session_state['df_xuly'] = None
    st.session_state['tien_xu_ly_done'] = False
    st.session_state['kq_pso'] = None
    st.session_state['current_source'] = option

df = None

if option == 'Dữ liệu thực tế':
    with st.spinner('⏳ Đang tải dữ liệu từ SSMS (bảng DuLieuNuoc)...'):
        try:
            df = doc_du_lieu(table='DuLieuNuoc')
            st.success(f'✓ Đã tải {len(df)} dòng từ DuLieuNuoc')
        except Exception as e:
            st.error(f'❌ Lỗi tải dữ liệu thực tế: {e}')

elif option == 'Dữ liệu import':
    st.sidebar.header('📥 Import dữ liệu')
    st.sidebar.markdown('**Tải file CSV và lưu vào bảng DuLieuNuocImport**')

    uploaded_file = st.sidebar.file_uploader('Chọn file (CSV hoặc Excel) để import', type=['csv', 'xls', 'xlsx'])

    if uploaded_file is not None:
        try:
            # Hỗ trợ đọc CSV và Excel
            filename = getattr(uploaded_file, 'name', '')
            if filename.lower().endswith(('.xls', '.xlsx')):
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                try:
                    df_import = pd.read_excel(uploaded_file, engine='openpyxl')
                except Exception as e:
                    # try without specifying engine
                    try:
                        uploaded_file.seek(0)
                        df_import = pd.read_excel(uploaded_file)
                    except Exception as e2:
                        raise Exception(f'Không đọc được file Excel: {e}; {e2}')
            else:
                # default: csv — try multiple encodings
                encodings = ['utf-8', 'cp1252', 'latin1']
                df_import = None
                for enc in encodings:
                    try:
                        uploaded_file.seek(0)
                    except Exception:
                        pass
                    try:
                        df_import = pd.read_csv(uploaded_file, encoding=enc)
                        break
                    except Exception:
                        df_import = None
                if df_import is None:
                    raise Exception('Không thể đọc file CSV. Hãy thử lưu file bằng UTF-8 hoặc CSV mã hóa Windows-1252.')

            st.sidebar.write(f'✓ Tải file thành công: {filename}')
            st.sidebar.dataframe(df_import.head(5))

            # Tùy chọn: chọn tên cột ngày và giá trị
            col1, col2 = st.sidebar.columns(2)
            with col1:
                col_date_name = st.selectbox('Chọn cột Ngày:', df_import.columns, key='col_date_import')
            with col2:
                col_value_name = st.selectbox('Chọn cột Giá trị:', df_import.columns, key='col_value_import')

            if st.sidebar.button('📤 Import vào DuLieuNuocImport', key='btn_import_db'):
                with st.spinner('⏳ Đang import dữ liệu vào bảng DuLieuNuocImport...'):
                    try:
                        from ket_noi_db import nhap_du_lieu_vao_db

                        # Chuẩn bị DataFrame với tên cột chuẩn
                        df_to_import = df_import[[col_date_name, col_value_name]].copy()
                        df_to_import.columns = ['date', 'value']

                        # Import vào database
                        count = nhap_du_lieu_vao_db(df_to_import, table='DuLieuNuocImport')
                        st.sidebar.success(f'✓ Đã import {count} dòng vào bảng DuLieuNuocImport')
                    except Exception as e:
                        st.sidebar.error(f'❌ Lỗi import: {e}')
        except Exception as e:
            st.sidebar.error(f'❌ Lỗi khi đọc file: {e}')
    
    with st.spinner('⏳ Đang tải dữ liệu từ SSMS (bảng DuLieuNuocImport)...'):
        try:
            df = doc_du_lieu(table='DuLieuNuocImport')
            st.success(f'✓ Đã tải {len(df)} dòng từ DuLieuNuocImport')
        except Exception as e:
            st.error(f'❌ Lỗi tải dữ liệu import: {e}')

elif option == 'Dữ liệu ảo':
    st.sidebar.write('**Dữ liệu ảo (được sinh và lưu trong SSMS)**')
    
    # Tùy chọn: chọn năm bắt đầu và số năm
    col_year, col_years = st.sidebar.columns(2)
    with col_year:
        nam_bat_dau = st.number_input('Năm bắt đầu:', min_value=2000, max_value=2050, value=2021, step=1, key='nam_bat_dau')
    with col_years:
        so_nam = st.number_input('Số năm:', min_value=1, max_value=10, value=3, step=1, key='so_nam')
    
    so_ngay = int(so_nam * 365)
    
    if st.sidebar.button('✓ Sinh/cập nhật dữ liệu ảo'):
        with st.spinner('⏳ Đang sinh dữ liệu ảo và thêm vào SSMS...'):
            try:
                from tao_du_lieu_ao import tao_bang_neu_chua_co, tao_va_insert_du_lieu_ao
                tao_bang_neu_chua_co()
                tao_va_insert_du_lieu_ao(num_days=so_ngay, nam_bat_dau=int(nam_bat_dau), thang_bat_dau=1)
                st.success('✓ Đã sinh dữ liệu ảo thành công')
            except Exception as e:
                st.error(f'❌ Lỗi: {e}')
    
    with st.spinner('⏳ Đang tải dữ liệu ảo từ SSMS...'):
        try:
            df = doc_du_lieu(table='DuLieuNuocAo')
            st.success(f'✓ Đã tải {len(df)} dòng từ DuLieuNuocAo')
        except Exception as e:
            st.warning(f'⚠️ Bảng DuLieuNuocAo trống. Hãy nhấn "Sinh/cập nhật dữ liệu ảo" trước.')




if df is not None:
    st.write('Dữ liệu (mẫu):')
    st.dataframe(df.head())
    
    st.subheader('📊 Biểu đồ 1: Chuỗi thời gian (dữ liệu gốc)')
    st.info('📌 Trục ngang: Thời gian (ngày/tháng/năm) | Trục dọc: Lượng nước (đơn vị trong dữ liệu)')
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df['date'], df['value'], linewidth=2, color='steelblue')
    ax.set_xlabel('Thời gian (Ngày)', fontsize=10)
    ax.set_ylabel('Lượng nước', fontsize=10)
    ax.set_title('Chuỗi thời gian gốc', fontsize=12, fontweight='bold')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader('📊 Biểu đồ 2: Phân bố (histogram) giá trị lượng nước')
    st.info('📌 Trục ngang: Giá trị lượng nước | Trục dọc: Tần suất (số lần xuất hiện)')
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.hist(df['value'].dropna(), bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Lượng nước', fontsize=10)
    ax.set_ylabel('Tần suất', fontsize=10)
    ax.set_title('Phân bố giá trị lượng nước', fontsize=12, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader('🔵 Xây dựng chuỗi thời gian mờ')
    st.markdown('**Chọn cấu hình tiền xử lý và nhấn "Áp dụng" để chuẩn bị dữ liệu cho mô hình:**')
    phan_giai = st.radio('Phân giải dữ liệu:', 
        ['D (Daily - Hàng ngày)', 'M (Monthly - Hàng tháng)'],
        horizontal=True,
        key='phan_giai_radio',
        help='D: Giữ nguyên dữ liệu hàng ngày | M: Gộp thành dữ liệu hàng tháng')
    phan_giai_val = 'D' if 'Daily' in phan_giai else 'M'
    cua_so_ma = st.slider('Cửa sổ Moving Average (ngày)', 1, 60, 7,
        key='cua_so_ma_slider',
        help='Số ngày dùng để làm mượt dữ liệu. Giá trị lớn = mượt hơn, chi tiết kém')
    
    if st.button('✓ Áp dụng tiền xử lý', key='btn_tien_xu_ly'):
        df2 = df.copy()
        df_xuly = tien_xu_ly(df2, luu_phan_giai=phan_giai_val, lam_tron=True, cua_so_ma=cua_so_ma)
        st.session_state['df_xuly'] = df_xuly
        st.session_state['tien_xu_ly_done'] = True
    
    if st.session_state.get('tien_xu_ly_done', False) and st.session_state['df_xuly'] is not None:
        df_xuly = st.session_state['df_xuly']
        st.subheader('📊 Dữ liệu chuỗi thời gian mờ (sau xử lý)')
        st.info('📌 Dữ liệu đã xử lý sẽ dùng để xây dựng mô hình. Trục ngang: Thời gian | Trục dọc: Lượng nước')
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_xuly['date'], df_xuly['value'], linewidth=2, label='Dữ liệu xử lý', color='steelblue')
        ax.plot(df_xuly['date'], df_xuly['ma'], linewidth=2, label=f'Moving Average (cửa sổ {cua_so_ma})', color='orange')
        ax.set_xlabel('Thời gian (Ngày)', fontsize=10)
        ax.set_ylabel('Lượng nước', fontsize=10)
        ax.set_title(f'Dữ liệu sau xử lý (Phân giải: {phan_giai_val})', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        st.success('✓ Tiền xử lý hoàn tất. Bây giờ bạn có thể chạy mô hình.')

        st.subheader('🟡 Chạy mô hình Chuỗi thời gian mờ')
        st.markdown('**Nhấn nút "🔥 Chạy mô hình" để tối ưu hóa khoảng mờ bằng K-Means**')
        st.info('Số khoảng mờ sẽ được xác định tự động bằng K-Means.')

        if st.sidebar.button('🔥 Chạy mô hình', key='btn_run_model'):
            with st.spinner('⏳ Đang chạy mô hình (có thể mất vài chục giây)...'):
                df_input = df_xuly[['date', 'value']].copy()
                try:
                    res = chay_pipeline_7_buoc(df_input, n_khoang=None)
                except Exception as e:
                    st.error(f'Lỗi khi chạy mô hình: {e}')
                    res = None

                if res is not None:
                    st.session_state['kq_pso'] = res
                    st.success(f'✓ Mô hình hoàn tất — **MSE trên tập Test = {res["test_rmse"]:.4f}**')

                    st.markdown('---')
                    st.header('Kết quả theo 7 bước')
                    steps_list = res.get('steps', [])
                    for step_idx, step in enumerate(steps_list, start=1):
                        st.subheader(step.get('ten'))
                        if step.get('mo_ta'):
                            st.write(step.get('mo_ta'))
                        kq = step.get('ket_qua')
                        
                        # Hiển thị các cấu trúc tùy theo loại nội dung
                        if isinstance(kq, list) and len(kq) > 0 and isinstance(kq[0], dict) and 'iter' in kq[0]:
                            # Lịch sử tối ưu hóa DE
                            df_hist = pd.DataFrame(kq)
                            st.write('Lịch sử tối ưu (DE):')
                            st.dataframe(df_hist)
                        elif isinstance(kq, list) and len(kq) > 0 and isinstance(kq[0], dict) and 'Khoảng' in kq[0]:
                            # Các khoảng tối ưu
                            df_bounds = pd.DataFrame(kq)
                            df_bounds = df_bounds.rename(columns={'left': 'Biên trái', 'right': 'Biên phải'})
                            st.write('Các khoảng mờ tối ưu:')
                            st.dataframe(df_bounds)
                        elif isinstance(kq, dict):
                            # Kiểm tra các cấu trúc chi tiết (edges, centers, rules, samples)
                            info_keys = {k: v for k, v in kq.items() if k in ('so_khoang', 'overlap', 'vmin', 'vmax')}
                            if info_keys:
                                st.write('Thông tin chính:')
                                st.table(pd.DataFrame(list(info_keys.items()), columns=['Thuộc tính', 'Giá trị']))
                            
                            # Ranh giới và trung tâm
                            edges = kq.get('edges')
                            centers = kq.get('centers')
                            if edges is not None and centers is not None:
                                df_ec = pd.DataFrame({
                                    'Trung tâm': centers,
                                    'Biên trái': edges[:-1],
                                    'Biên phải': edges[1:]
                                })
                                st.write('Bảng các khoảng và trung tâm:')
                                st.dataframe(df_ec)
                                
                                # Vẽ đường cong membership Gaussian
                                try:
                                    edges_arr = np.array(edges, dtype=float)
                                    centers_arr = np.array(centers, dtype=float)
                                    so_khoang_val = int(kq.get('so_khoang', len(centers_arr)))
                                    overlap_val = float(kq.get('overlap', 0.2))
                                    width = (edges_arr[-1] - edges_arr[0]) / max(1, so_khoang_val)
                                    sigma = width * (0.3 + 0.7 * overlap_val)
                                    x = np.linspace(edges_arr[0], edges_arr[-1], 400)
                                    fig_mem, ax_mem = plt.subplots(figsize=(10, 4))
                                    for j, c in enumerate(centers_arr):
                                        mu = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
                                        ax_mem.plot(x, mu, label=f'μ_{j}', linewidth=2)
                                    ax_mem.set_title('Hàm membership Gaussian cho các trung tâm')
                                    ax_mem.set_xlabel('Giá trị')
                                    ax_mem.set_ylabel('Membership')
                                    ax_mem.set_ylim(0, 1.05)
                                    ax_mem.legend(ncol=min(3, len(centers_arr)), fontsize='small')
                                    ax_mem.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    st.pyplot(fig_mem)
                                except Exception:
                                    pass
                            
                            # Tóm tắt quy tắc
                            rules = kq.get('rules_summary') or kq.get('rules')
                            if rules:
                                rows = []
                                for frm, tolist in (rules.items() if isinstance(rules, dict) else []):
                                    for it in tolist:
                                        rows.append({'Từ': int(frm), 'Đến': int(it.get('to')), 'Trọng số': float(it.get('weight'))})
                                if rows:
                                    st.write('Tóm tắt quy tắc (top hậu quả):')
                                    st.dataframe(pd.DataFrame(rows).sort_values(['Từ', 'Trọng số'], ascending=[True, False]))
                            
                            # Ví dụ membership
                            samples = kq.get('sample_memberships')
                            if samples:
                                mem_rows = []
                                for s in samples:
                                    row = {'Chỉ số': s.get('index'), 'Giá trị': s.get('value')}
                                    mem = s.get('membership') or []
                                    for j, m in enumerate(mem):
                                        col_name = f'μ_{j}'
                                        row[col_name] = float(m)
                                    mem_rows.append(row)
                                st.write('Ví dụ membership cho vài giá trị đầu:')
                                st.dataframe(pd.DataFrame(mem_rows))
                            
                            # Nếu không tìm thấy cấu trúc cụ thể, hiển thị JSON
                            if not (edges and centers) and not rules and not samples and not info_keys:
                                st.json(_make_serializable(kq))
                        elif isinstance(kq, list) and len(kq) > 0 and isinstance(kq[0], (int, float, str)):
                            st.write(kq)
                        else:
                            # Cách dự phòng: cố gắng hiển thị dưới dạng bảng
                            try:
                                st.dataframe(pd.DataFrame(kq))
                            except Exception:
                                st.write(kq)
                        
                        # Hiển thị biểu đồ dự báo chỉ ở bước cuối cùng (bước 7)
                        if step_idx == len(steps_list):
                            try:
                                preds = res.get('preds', [])
                                if len(preds) > 0:
                                    test_df = df_input.tail(len(preds)).copy()
                                    fig, ax = plt.subplots(figsize=(12, 4))
                                    dates = test_df['date'].values
                                    ax.plot(dates, test_df['value'].values, label='Thực tế', color='steelblue', linewidth=2)
                                    ax.plot(dates, preds, label='Dự báo', color='orange', linewidth=2)
                                    ax.set_title('So sánh thực tế và dự báo (Test)')
                                    ax.set_xlabel('Thời gian')
                                    ax.set_ylabel('Lượng nước')
                                    ax.legend(loc='best')
                                    ax.grid(True, alpha=0.3)
                                    plt.xticks(rotation=30)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                            except Exception:
                                pass

        
        if st.session_state.get('kq_pso') is not None:
            res = st.session_state['kq_pso']
            st.write('---')
            st.subheader('📈 Tóm tắt kết quả')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('MSE trên tập Test', f"{res['test_rmse']:.4f}", help='Sai số trung bình')
            with col2:
                num_intervals = len(res.get('best_edges', [])) - 1
                st.metric('Số khoảng mờ', f"{num_intervals}", help='Số khoảng được tối ưu')
            with col3:
                num_pred = len(res.get('preds', []))
                st.metric('Số dự báo', f"{num_pred}", help='Số mẫu dự báo trên tập Test')
else:
    st.warning('⚠️ Không thể tải dữ liệu từ SSMS. Vui lòng kiểm tra kết nối và thông tin server.')
