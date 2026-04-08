import os
import pandas as pd
import numpy as np

from datetime import datetime
from core.connect import exchange, config


# ✅ 新增：定義全域變數，防止 NameError 崩潰
BLACKLIST = ['USDC/USDT:USDT', 'BUSD/USDT:USDT', 'EUR/USDT:USDT'] # 穩定幣黑名單
NET_FLOW_SIGMA = 2.0         # Z-Score 資金流入門檻
MIN_IMBALANCE_RATIO = 0.2    # 買盤牆厚度門檻

STATUS_DIR = "status"
STATUS_FILE = f"{STATUS_DIR}/btc_status_long.csv"
STATUS_COLUMNS = ['timestamp', 'btc_price', 'target_price', 'sma20', 'sma50', 'signal_code', 'decision_text']

if not os.path.exists(STATUS_DIR): os.makedirs(STATUS_DIR)


def log_status_to_csv(data_dict):
    row = {col: '' for col in STATUS_COLUMNS}
    row.update(data_dict)
    row['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pd.DataFrame([row], columns=STATUS_COLUMNS).to_csv(STATUS_FILE, mode='a', index=False,
                                                       header=not os.path.exists(STATUS_FILE))


# def get_btc_regime():
#     """BTC 導航：判斷整體市場多空環境"""
#     try:
#         ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', timeframe='1h', limit=60)
#         df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
#         curr_p = df['c'].iloc[-1]
#         sma20 = df['c'].rolling(20).mean().iloc[-1]
#         sma50 = df['c'].rolling(50).mean().iloc[-1]
#
#         dev_threshold = config['STRATEGY']['btc_dev_threshold']
#         target_long = sma20 * (1 + dev_threshold)
#
#         cond_price = curr_p > target_long
#         cond_trend = sma20 > sma50
#
#         tick_p = "✅" if cond_price else "❌"
#         tick_t = "✅" if cond_trend else "❌"
#
#         if cond_price and cond_trend:
#             status, signal = "🟢 GREEN   (Bullish - All in)", 1
#         elif cond_price or cond_trend:
#             status, signal = "🟡 YELLOW  (Conditions unmet - Standby)", 0
#         else:
#             status, signal = "🔴 RED     (Bearish - Do not enter)", -1
#
#         report_data = {
#             'btc_price': round(curr_p, 2),
#             'target_price': round(target_long, 2),
#             'sma20': round(sma20, 2),
#             'sma50': round(sma50, 2),
#             'signal_code': signal,
#             'decision_text': status
#         }
#         log_status_to_csv(report_data)
#
#         print("-" * 60)
#         print(f"📈 BTC Live Status (Long) | Price: {curr_p:.0f}")
#         print(f"1️⃣ Price Threshold: Current({curr_p:.0f}) > Target({target_long:.0f}) {tick_p}")
#         print(f"2️⃣ Trend Confirmation: SMA20({sma20:.0f}) > SMA50({sma50:.0f}) {tick_t}")
#         print(f"🚦 Final Decision: {status}")
#         print("-" * 60)
#
#         return signal
#     except Exception as e:
#         print(f"⚠️ Navigation Fault: {e}")
#         return 0


def get_btc_regime():
    """🚀 終極導航 (做多版)：HMA 交叉 + ADX 趨勢過濾 + 均量過濾"""
    try:
        # ⚠️ 必須拉長到 150，確保 HMA50 和 ADX 有足夠數據計算
        ohlcv = exchange.fetch_ohlcv('BTC/USDT:USDT', timeframe='1h', limit=150)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        curr_p = df['c'].iloc[-1]
        curr_v = df['v'].iloc[-1]

        # ==========================================
        # 1️⃣ 極速趨勢引擎：計算 HMA 20 與 HMA 50
        # ==========================================
        def calc_hma(s, period):
            half_length = int(period / 2)
            sqrt_length = int(np.sqrt(period))
            # WMA (加權移動平均) 輔助函數
            weights_half = np.arange(1, half_length + 1)
            weights_full = np.arange(1, period + 1)
            weights_sqrt = np.arange(1, sqrt_length + 1)

            wma_half = s.rolling(half_length).apply(lambda x: np.dot(x, weights_half) / weights_half.sum(), raw=True)
            wma_full = s.rolling(period).apply(lambda x: np.dot(x, weights_full) / weights_full.sum(), raw=True)

            s_diff = (2 * wma_half) - wma_full
            hma = s_diff.rolling(sqrt_length).apply(lambda x: np.dot(x, weights_sqrt) / weights_sqrt.sum(), raw=True)
            return hma

        df['hma20'] = calc_hma(df['c'], 20)
        df['hma50'] = calc_hma(df['c'], 50)

        # 🚀 修改：條件 1：HMA20 升穿 HMA50 (無滯後升勢確立)
        hma20_val = df['hma20'].iloc[-1]
        hma50_val = df['hma50'].iloc[-1]
        cond_trend = hma20_val > hma50_val  # <--- LONG 關鍵：20 大過 50

        # ==========================================
        # 2️⃣ 趨勢強度濾網：計算 ADX (14)
        # ==========================================
        df['up'] = df['h'] - df['h'].shift(1)
        df['down'] = df['l'].shift(1) - df['l']
        df['+dm'] = np.where((df['up'] > df['down']) & (df['up'] > 0), df['up'], 0)
        df['-dm'] = np.where((df['down'] > df['up']) & (df['down'] > 0), df['down'], 0)
        df['tr'] = np.maximum(df['h'] - df['l'],
                              np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))

        atr_14 = df['tr'].ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * (pd.Series(df['+dm']).ewm(alpha=1 / 14, adjust=False).mean() / atr_14)
        minus_di = 100 * (pd.Series(df['-dm']).ewm(alpha=1 / 14, adjust=False).mean() / atr_14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_val = dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]

        # 條件 2：ADX > 22 (過濾無方向橫盤，多空通用)
        cond_adx = adx_val > 22

        # ==========================================
        # 3️⃣ 成交量濾網：當前 1 小時量 > 20小時均量
        # ==========================================
        sma_v_20 = df['v'].rolling(20).mean().iloc[-1]
        cond_vol = curr_v > sma_v_20

        # ==========================================
        # 4️⃣ 整合訊號與輸出
        # ==========================================
        tick_t = "✅" if cond_trend else "❌"
        tick_a = f"✅ (ADX: {adx_val:.1f})" if cond_adx else f"❌ (ADX: {adx_val:.1f})"
        tick_v = f"✅ (Vol: {curr_v:.0f} > {sma_v_20:.0f})" if cond_vol else f"❌ (Vol: {curr_v:.0f} < {sma_v_20:.0f})"

        # 必須三個條件同時滿足才開綠燈
        if cond_trend and cond_adx and cond_vol:
            status, signal = "🟢 GREEN   (Bullish Trend, ADX & Vol Validated)", 1
        elif cond_trend or cond_adx:
            status, signal = "🟡 YELLOW  (Standby - Waiting for confluence)", 0
        else:
            status, signal = "🔴 RED     (Sideways / Bearish)", -1  # <--- 沒趨勢或是跌勢就停火

        # 兼容 CSV 紀錄 (借用欄位名)
        report = {
            'btc_price': round(curr_p, 2),
            'target_price': round(hma50_val, 2),
            'sma20': round(hma20_val, 2),
            'sma50': round(adx_val, 2),  # 借用位置記錄 ADX
            'signal_code': signal,
            'decision_text': status
        }
        log_status_to_csv(report)

        print("-" * 60)
        print(f"📊 BTC 實時戰報 (Long 多頭 HMA+ADX版) | 現價: {curr_p:.0f}")
        print(f"1️⃣ 極速升勢: HMA20({hma20_val:.0f}) > HMA50({hma50_val:.0f}) {tick_t}")  # <--- 改為大於
        print(f"2️⃣ 趨勢強度: ADX > 22 {tick_a}")
        print(f"3️⃣ 動能確認: 當前成交量 > 20H均量 {tick_v}")
        print(f"🚦 最終決策: {status}")
        print("-" * 60)

        return signal
    except Exception as e:
        print(f"⚠️ 導航故障: {e}")
        return 0


# def scouting_top_coins(n=5):
#     try:
#         tickers = exchange.fetch_tickers()
#         data = []
#         for s, t in tickers.items():
#             if s.endswith(':USDT') and s not in BLACKLIST and t['percentage'] is not None:
#                 data.append({'symbol': s, 'volume': t['quoteVolume'], 'change': t['percentage'], 'ask': t.get('ask'),
#                              'bid': t.get('bid')})
#
#         df = pd.DataFrame(data)
#         if df.empty: return []
#
#         # 🚀 關鍵：動態計算成交量的 80% 分位數
#         # 這代表只有成交量排名前 20% 的幣種能通過，無論市場熱還是冷
#         dynamic_min_volume = df['volume'].quantile(0.8)
#
#         # 過濾 Spread 與成交量
#         df = df[(df['volume'] >= dynamic_min_volume)]
#         df['spread'] = (df['ask'] - df['bid']) / df['bid']
#         df = df[df['spread'] < 0.0010]
#
#         # 從這批實力幣中選最強的 n 隻
#         return df.sort_values('volume', ascending=False).head(50).sort_values('change', ascending=False).head(n)[
#             'symbol'].tolist()
#     except Exception as e:
#         print(f"⚠️ Dynamic Scouting Error: {e}")
#         return []
#
#
# 🛠️ 舊代碼保留 (請將您原本的 def scouting_top_coins 或是 scouting_weak_coins 全部選起來，按 Ctrl+/ 變成註解)
# def scouting_top_coins(n=5):
#     ... (您原本的代碼) ...
def scouting_top_coins(n=5):
    """海選強勢幣 (過濾 Spread)"""
    try:
        tickers = exchange.fetch_tickers()
        data = []
        for s, t in tickers.items():
            if s.endswith(':USDT') and s not in BLACKLIST and t['percentage'] is not None:
                ask = t.get('ask')
                bid = t.get('bid')
                if ask and bid and bid > 0:
                    spread = (ask - bid) / bid
                    # if spread < 0.0015: # 🛠️ 舊代碼保留
                    if spread < 0.0010:  # 🚀 新增：大幣專用，極嚴格 Spread 門檻
                        data.append({'symbol': s, 'volume': t['quoteVolume'], 'change': t['percentage']})

        df = pd.DataFrame(data)
        if df.empty: return []

        # return df.sort_values('volume', ascending=False).head(20).sort_values('change', ascending=False).head(n)['symbol'].tolist() # 🛠️ 舊代碼保留

        # 🚀 新增：動態成交量過濾 (全市場 Top 20% 資金)，確保入選嘅一定係大幣
        dynamic_min_volume = df['volume'].quantile(0.8)
        df_filtered = df[df['volume'] >= dynamic_min_volume]
        return df_filtered.sort_values('change', ascending=False).head(n)['symbol'].tolist()

    except Exception as e:
        print(f"⚠️ Scouting Error: {e}")
        return []


# # ✅ 新增代碼：修正回傳值，增加 z_score 給 main.py 用嚟計 Risk
# def apply_lee_ready_logic(symbol):
#     """Lee-Ready 資金流邏輯 + 訂單簿失衡度 (Imbalance) + P95濾網 [終極做多版]"""
#     try:
#         ob = exchange.fetch_order_book(symbol, limit=20)
#         midpoint = (ob['bids'][0][0] + ob['asks'][0][0]) / 2
#
#         bid_vol = sum([b[1] for b in ob['bids']])
#         ask_vol = sum([a[1] for a in ob['asks']])
#         imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
#         imbalance = max(-1, min(1, imbalance))
#
#         trades = exchange.fetch_trades(symbol, limit=500)
#         df = pd.DataFrame(trades, columns=['price', 'amount', 'timestamp'])
#         df['dir'] = np.where(df['price'] > midpoint, 1, np.where(df['price'] < midpoint, -1, 0))
#         df['tick'] = df['price'].diff().apply(np.sign).replace(0, np.nan).ffill().fillna(0)
#         df['final'] = np.where(df['dir'] != 0, df['dir'], df['tick'])
#
#         df['usd_val'] = df['amount'] * df['price']
#         p95 = df['usd_val'].quantile(0.95)
#         df['usd_val_clipped'] = df['usd_val'].clip(upper=p95)
#
#         df['weighted_flow'] = df['final'] * df['usd_val_clipped']
#         net_flow = df['weighted_flow'].sum()
#         flow_std = df['weighted_flow'].std()
#
#         if pd.isna(flow_std) or flow_std <= 0:
#             z_score = 0
#         else:
#             z_score = net_flow / flow_std
#             z_score = np.clip(z_score, -10, 10)
#
#         is_strong = (z_score > NET_FLOW_SIGMA) and (imbalance > MIN_IMBALANCE_RATIO)
#
#         if is_strong:
#             print(f"📈 {symbol} Long Validated | Z-Score: {z_score:.2f} | Imbalance: {imbalance:.2f} | P95 Cap: {p95:.0f}")
#         elif z_score > NET_FLOW_SIGMA:
#             print(f"⚠️ {symbol} Fake-Pump Prevented | Z-Score: {z_score:.2f} but Imbalance is {imbalance:.2f} (Sell Wall in the way)")
#
#         # 🚀 修正：回傳值增加 z_score 以配合 main.py 需求 (4 個變數)
#         return net_flow, df['price'].iloc[-1], is_strong, z_score
#
#     except Exception as e:
#         print(f"❌ 錯誤 [{symbol}] Lee-Ready: {e}")
#         # 🚀 修正：錯誤時也要回傳 4 個值，防止 ValueError
#         return 0, 0, False, 0
#
#
# 🛠️ 舊代碼保留 (請將您原本的 def apply_lee_ready_logic 全部變成註解)
# def apply_lee_ready_logic(symbol):
#     ... (您原本的代碼) ...

# 🛠️ 將您 strategy.py 裡面原本的 apply_lee_ready_logic 內容全部加上 # 註解保留，然後貼上以下全新邏輯

def apply_lee_ready_logic(symbol):
    """Lee-Ready 狙擊手版本 (實裝 ABC 改進)"""
    try:
        # 🚀 改進 C：結合訂單簿 (Orderbook) 失衡預判
        ob = exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum([b[1] for b in ob['bids']])
        ask_vol = sum([a[1] for a in ob['asks']])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) if (bid_vol + ask_vol) > 0 else 0
        imbalance = max(-1, min(1, imbalance))

        # 🚀 改進 B：拉取更長視窗 (200筆) 進行長短窗對比
        trades = exchange.fetch_trades(symbol, limit=200)
        if not trades: return 0, 0, False, 0

        df = pd.DataFrame(trades)
        df['price_change'] = df['price'].diff()
        df['direction'] = np.where(df['price_change'] > 0, 1, np.where(df['price_change'] < 0, -1, 0))
        # 🚀 修正 Pandas 報錯：先將 0 轉為 NaN，再向前填充，最後將開頭無法填充的填回 0
        df['direction'] = df['direction'].replace(0, np.nan).ffill().fillna(0)

        # 🚀 新增：大單加權 (大於平均量 2 倍的單，權重 x 2)
        avg_vol = df['amount'].mean()
        df['weight'] = np.where(df['amount'] > avg_vol * 2, 2.0, 1.0)
        df['net_flow'] = df['direction'] * df['amount'] * df['price'] * df['weight']

        # 計算長短窗 Net Flow
        short_window_flow = df['net_flow'].tail(50).sum()

        # 🚀 改進 A：加速度 (Acceleration) - 對比最近 25 筆與前 25 筆
        recent_25_flow = df['net_flow'].tail(25).sum()
        prev_25_flow = df['net_flow'].iloc[-50:-25].sum()
        acceleration = recent_25_flow - prev_25_flow

        is_strong = False
        z_score = 0
        if df['net_flow'].std() > 0:
            z_score = short_window_flow / df['net_flow'].std()

        # 🎯 條件 1 [狙擊模式]：短窗有正向流入 + 加速度向上爆發 + 買盤失衡厚度 > 15% (搶跑進場)
        if (short_window_flow > 0) and (acceleration > 0) and (imbalance > 0.15):
            is_strong = True
            print(f"🔥 {symbol} Sniper Entry! Accel: {acceleration:.0f} | Imbalance: {imbalance:.2f}")

        # 🎯 條件 2 [穩健模式]：傳統 Z-Score 突破 (適應大型幣，門檻設 1.2)
        elif z_score > 1.2:
            is_strong = True
            print(f"📈 {symbol} Z-Score Validated: {z_score:.2f}")

        return short_window_flow, df['price'].iloc[-1], is_strong, z_score

    except Exception as e:
        print(f"❌ 錯誤 [{symbol}] LR Sniper: {e}")
        return 0, 0, False, 0


def get_market_metrics(symbol):
    """計算 ATR 與波動率"""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=50)
        df = pd.DataFrame(ohlcv, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
        df['tr'] = np.maximum(df['h'] - df['l'],
                              np.maximum(abs(df['h'] - df['c'].shift(1)), abs(df['l'] - df['c'].shift(1))))
        atr = df['tr'].rolling(14, min_periods=1).mean().iloc[-1]

        if pd.isna(atr) or atr == 0: return None, False
        is_volatile = (atr / df['c'].iloc[-1]) > 0.0005
        return atr, is_volatile
    except:
        return None, False