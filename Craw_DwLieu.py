import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from datetime import datetime, timedelta
import pandas as pd
import time


def fetch_day(date_obj):
    date_str = date_obj.strftime("%d/%m/%Y %H:%M")
    encoded_date = quote(date_str)

    url = (
        "https://hochuathuydien.evn.com.vn/"
        f"PageHoChuaThuyDienEmbedEVN.aspx?td={encoded_date}&vm=90&lv=9&hc=1"
    )

    for retry in range(3):
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if r.status_code == 200:
                return parse_table(r.text, date_str)
        except:
            pass

        time.sleep(1)

    print(f"❌ Lỗi khi lấy dữ liệu ngày {date_str}")
    return []


def parse_table(html, date_str):
    soup = BeautifulSoup(html, "html.parser")

    container = soup.find("div", id="myHeader")
    if container is None:
        print(f"⚠️ Không tìm thấy myHeader tại {date_str}")
        return []

    table = container.find("table")
    if table is None:
        print(f"⚠️ Không tìm thấy bảng tại {date_str}")
        return []

    rows = table.find_all("tr")

    # Bỏ 2 dòng header đầu
    data_rows = rows[2:]

    # Lọc hàng có <td>
    real_rows = [tr for tr in data_rows if tr.find("td")]

    if not real_rows:
        print(f"⚠️ Không có dữ liệu tại {date_str}")
        return []

    # Lấy hàng cuối cùng
    last_tr = real_rows[-1]
    cols = [td.get_text(strip=True) for td in last_tr.find_all("td")]

    return [[date_str] + cols]


def run_scraper(start, end):
    all_rows = []
    day = start

    while day <= end:
        print("Lấy ngày:", day.strftime("%d/%m/%Y %H:%M"))
        rows = fetch_day(day)
        all_rows.extend(rows)

        day += timedelta(days=1)

    df = pd.DataFrame(all_rows)

    output_path = "CSV/evn.csv"

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"✔️ Xong, lưu file tại: {output_path}")



run_scraper(
    datetime(2020, 1, 1, 21, 0),
    datetime(2025, 12, 7, 21, 0)
)
