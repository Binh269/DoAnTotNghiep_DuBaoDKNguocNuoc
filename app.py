import streamlit as st
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from evaluate import chay_psu_toi_uu
from xuly_du_lieu import doc_du_lieu, tien_xu_ly

st.set_page_config(page_title='Hệ thống dự báo nhu cầu nước (mờ+PSO)', layout='wide')
st.title('📊 Dự báo nhu cầu nước sinh hoạt — Chuỗi thời gian mờ + PSO')

if 'db_loaded' not in st.session_state:
    st.session_state['db_loaded'] = False
    st.session_state['df_xuly'] = None
    st.session_state['tien_xu_ly_done'] = False
    st.session_state['kq_pso'] = None
    st.session_state['current_source'] = None

st.sidebar.header('📁 Chọn dữ liệu')
option = st.sidebar.radio('Nguồn dữ liệu', ['Dữ liệu thực tế', 'Dữ liệu ảo'], key='source_radio')

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

    st.subheader('⚙️ Tiền xử lý dữ liệu')
    st.markdown('**Chọn cấu hình tiền xử lý và nhấn "Áp dụng" để xem kết quả:**')
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
        st.subheader('📊 Biểu đồ 3: Dữ liệu sau xử lý')
        st.info('📌 Trục ngang: Thời gian | Trục dọc: Lượng nước | Màu xanh: Giá trị xử lý | Màu cam: Moving Average')
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
        st.success('✓ Tiền xử lý hoàn tất. Bây giờ bạn có thể chạy PSO ở phần dưới.')

        st.subheader('🚀 Tối ưu PSO (sau xử lý)')
        st.markdown('**Điều chỉnh tham số PSO ở sidebar rồi nhấn nút "🔥 Chạy tối ưu PSO"**')
        st.sidebar.header('🚀 Tối ưu PSO')
        so_hat = st.sidebar.slider('Số hạt (Particles)', 5, 30, 8,
            key='so_hat_slider',
            help='Số lượng tìm kiếm. Tăng = chậm hơn nhưng chính xác hơn')
        vong = st.sidebar.slider('Số vòng lặp (Iterations)', 5, 100, 30,
            key='vong_slider',
            help='Số vòng PSO chạy. Tăng = tìm kiếm lâu hơn, kết quả tốt hơn')
        
        if st.sidebar.button('🔥 Chạy tối ưu PSO', key='btn_pso'):
            with st.spinner('⏳ Đang chạy PSO (có thể mất vài chục giây)...'):
                df_input = df_xuly[['date', 'value']].copy()
                res = chay_psu_toi_uu(df_input, so_hat=so_hat, vong=vong)
                st.session_state['kq_pso'] = res
                st.success(f'✓ Tối ưu hoàn tất — **RMSE trên tập Test = {res["test_rmse"]:.4f}**')
                st.write('**Tham số tối ưu tìm được:**')
                st.write(f'  - Số khoảng mờ: {int(res["best"][0])}')
                st.write(f'  - Tỉ lệ chồng lấp: {res["best"][1]:.4f}')
                
                st.subheader('📊 Biểu đồ kết quả dự báo')
                st.info('📌 Trục ngang: Thời gian | Trục dọc: Lượng nước | Màu xanh: Giá trị thực | Màu cam: Dự báo')
                fig, ax = plt.subplots(figsize=(16, 5))
                idx_pred = len(df_input) - len(res['preds'])
                dates_pred = df_input['date'].values[idx_pred:]
                ax.plot(dates_pred, df_input['value'].values[idx_pred:], 
                       linewidth=2, label='Giá trị thực tế', color='steelblue', marker='o', markersize=3)
                ax.plot(dates_pred, res['preds'], 
                       linewidth=2, label='Dự báo mô hình fuzzy', color='orange', marker='s', markersize=3)
                ax.set_xlabel('Thời gian', fontsize=10)
                ax.set_ylabel('Lượng nước', fontsize=10)
                ax.set_title('Kết quả dự báo', fontsize=12, fontweight='bold')
                ax.legend(loc='best')
                plt.tight_layout()
                st.pyplot(fig)
        
        if st.session_state.get('kq_pso') is not None:
            res = st.session_state['kq_pso']
            st.write('---')
            st.subheader('📈 Tóm tắt kết quả')
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric('RMSE Test', f"{res['test_rmse']:.4f}", help='Sai số trên tập test')
            with col2:
                st.metric('Số khoảng mờ', f"{int(res['best'][0])}", help='Tham số tối ưu')
            with col3:
                st.metric('Tỉ lệ chồng lấp', f"{res['best'][1]:.4f}", help='Tham số overlap tối ưu')
else:
    st.warning('⚠️ Không thể tải dữ liệu từ SSMS. Vui lòng kiểm tra kết nối và thông tin server.')
