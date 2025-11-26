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
from evaluate import chay_psu_toi_uu
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
        st.success('✓ Tiền xử lý hoàn tất. Bây giờ bạn có thể chạy PSO ở phần dưới.')

        st.subheader('🟡 Ứng dụng PSO tối ưu tham số khoảng mờ')
        st.markdown('**Điều chỉnh tham số PSO ở sidebar rồi nhấn nút "🔥 Chạy PSO" để tối ưu hóa**')
        st.sidebar.header('🟡 PSO tối ưu')
        so_hat = st.sidebar.slider('Số hạt (Particles)', 5, 30, 8,
            key='so_hat_slider',
            help='Số lượng tìm kiếm. Tăng = chậm hơn nhưng chính xác hơn')
        vong = st.sidebar.slider('Số vòng lặp (Iterations)', 5, 100, 30,
            key='vong_slider',
            help='Số vòng PSO chạy. Tăng = tìm kiếm lâu hơn, kết quả tốt hơn')
        
        if st.sidebar.button('🔥 Chạy PSO', key='btn_pso'):
            with st.spinner('⏳ Đang chạy PSO (có thể mất vài chục giây)...'):
                df_input = df_xuly[['date', 'value']].copy()
                res = chay_psu_toi_uu(df_input, so_hat=so_hat, vong=vong)
                st.session_state['kq_pso'] = res
                st.success(f'✓ Tối ưu hoàn tất — **RMSE trên tập Test = {res["test_rmse"]:.4f}**')
                st.write('**Tham số tối ưu tìm được:**')
                st.write(f'  - Số khoảng mờ: {int(res["best"][0])}')
                st.write(f'  - Tỉ lệ chồng lấp: {res["best"][1]:.4f}')
                
                st.subheader('🔴 Đánh giá sai số dự báo')
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
                # Hiển thị chi tiết các bước (nếu có)
                if 'steps' in res:
                    st.markdown('---')
                    st.subheader('🔎 Kết quả chi tiết theo từng bước')
                    for i, step in enumerate(res.get('steps', []), start=1):
                        # Kết hợp số thứ tự và tên bước: 'Bước i - Tên bước' nếu có tên
                        if step.get('ten'):
                            label = f"Bước {i} - {step.get('ten')}"
                        else:
                            label = f'Bước {i}'
                        with st.expander(label, expanded=False):
                            mo_ta = step.get('mo_ta', '')
                            if mo_ta:
                                st.write(mo_ta)
                            ket_qua = step.get('ket_qua')
                            # Nếu ket_qua là danh sách (ví dụ grid search), hiển thị bảng
                            if isinstance(ket_qua, list):
                                try:
                                    st.dataframe(pd.DataFrame(ket_qua))
                                except Exception:
                                    for it in ket_qua:
                                        st.write(it)
                            # Nếu ket_qua là dict chứa thông tin tạo tập mờ (steps_info), hiển thị nhiều bảng
                            elif isinstance(ket_qua, dict):
                                # Hiển thị các trường đơn giản (so_khoang, overlap, vmin, vmax)
                                info_keys = {k: v for k, v in ket_qua.items() if k in ('so_khoang', 'overlap', 'vmin', 'vmax')}
                                if info_keys:
                                    st.write('Thông tin chính:')
                                    st.table(pd.DataFrame(list(info_keys.items()), columns=['Thuộc tính', 'Giá trị']))

                                # edges / centers
                                edges = ket_qua.get('edges')
                                centers = ket_qua.get('centers')
                                if edges is not None and centers is not None:
                                    df_ec = pd.DataFrame({
                                        'Trung tâm': centers,
                                        'Biên trái': edges[:-1],
                                        'Biên phải': edges[1:]
                                    })
                                    st.write('Bảng các khoảng và trung tâm:')
                                    st.dataframe(df_ec)
                                    # Vẽ hàm membership Gaussian cho từng trung tâm
                                    try:
                                        edges_arr = np.array(edges, dtype=float)
                                        centers_arr = np.array(centers, dtype=float)
                                        so_khoang_val = int(ket_qua.get('so_khoang', len(centers_arr)))
                                        overlap_val = float(ket_qua.get('overlap', 0.2))
                                        width = (edges_arr[-1] - edges_arr[0]) / max(1, so_khoang_val)
                                        sigma = width * (0.3 + 0.7 * overlap_val)
                                        x = np.linspace(edges_arr[0], edges_arr[-1], 400)
                                        fig_mem, ax_mem = plt.subplots(figsize=(8, 3))
                                        for j, c in enumerate(centers_arr):
                                            mu = np.exp(-((x - c) ** 2) / (2 * sigma ** 2))
                                            ax_mem.plot(x, mu, label=f'μ_{j}')
                                        ax_mem.set_title('Hàm membership Gaussian cho các trung tâm')
                                        ax_mem.set_xlabel('Giá trị')
                                        ax_mem.set_ylabel('Membership')
                                        ax_mem.set_ylim(0, 1.05)
                                        ax_mem.legend(ncol=2, fontsize='small')
                                        plt.tight_layout()
                                        st.pyplot(fig_mem)
                                    except Exception:
                                        pass

                                # rules_summary
                                rules = ket_qua.get('rules_summary') or ket_qua.get('rules')
                                if rules:
                                    rows = []
                                    for frm, tolist in (rules.items() if isinstance(rules, dict) else []):
                                        for it in tolist:
                                            rows.append({'Từ': int(frm), 'Đến': int(it.get('to')), 'Trọng số': float(it.get('weight'))})
                                    if rows:
                                        st.write('Tóm tắt quy tắc (top hậu quả):')
                                        st.dataframe(pd.DataFrame(rows).sort_values(['Từ', 'Trọng số'], ascending=[True, False]))

                                # sample_memberships
                                samples = ket_qua.get('sample_memberships')
                                if samples:
                                    # tạo dataframe với cột membership_0..membership_n
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
                                # Nếu dict không có cấu trúc trên, hiển thị JSON để debug
                                if not (edges and centers) and not rules and not samples:
                                    st.json(_make_serializable(ket_qua))
                            else:
                                st.json(_make_serializable(ket_qua))
        
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
