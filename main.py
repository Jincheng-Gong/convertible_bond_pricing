import pandas as pd
import sys
import numpy as np
import talib as ta
import math
import multiprocessing as mp
import random
# import warnings
# warnings.filterwarnings("ignore")

def get_monte_carlo_price(all_info, bond_name, date, simulations):
    '''
    pricing the convertible bonds based on monte carlo simulation
    return model price and real price on that date

    bond_name - the name of the convertible bonds
    date - the pricing date
    simulations - the number of paths in each simulation
    '''
    # assume that the dividend rate of stocks is 0
    dividend_rate = 0  # 默认股票分红率为0
    # get convertible bond info
    # all_info = rq.convertible.all_instruments()  # 取得所有可转债基础信息
    info = all_info[all_info['symbol'] == bond_name]  # 取出当前模拟的可转债信息
    order_book_id = info['order_book_id'].iloc[0]  # 取得可转债合约代码
    stock_code = info['stock_code'].iloc[0]  # 取得可转债合约对应股票代码
    maturity_date = info['maturity_date'].iloc[0]  # 可转债到期日
    conversion_start_date = info['conversion_start_date'].iloc[0]  # 转换期起始日
    conversion_end_date = info['conversion_end_date'].iloc[0]  # 转换期到期日
    coupon = info['coupon_rate'].iloc[0] * 100 # coupon
    value_date = info['value_date'].iloc[0]  # 起息日
    de_listed_date = info['de_listed_date'].iloc[0]  # 债券摘牌日
    # print(order_book_id)

    # get the close price of underlying stock on pricing date，获取当前可转债合约的历史数据
    # real_price = \
    #     rq.get_price(order_book_id, start_date=date, end_date=date, frequency='1d', fields=None, adjust_type='pre',
    #                  skip_suspended=False)['close'].iloc[0]
    real_price = pd.read_parquet('conversible_bond_price.parquet')
    real_price = real_price.reset_index()
    real_price = real_price[real_price['order_book_id'] == order_book_id]
    real_price = real_price[real_price.date == date]
    real_price = float(real_price.close)

    # get the credit rating of the bond
    # credit = credit_rating[credit_rating['symbol'] == bond_name]['credit_rating'].iloc[0]

    # get the current yield curve
    # 获取收益率曲线，2002至date的中债国债收益率曲线
    # yc = rq.get_yield_curve(start_date=date, end_date=date).iloc[-1]
    yc = pd.read_parquet('yield_curve.parquet')
    yc = yc[yc.index == date].iloc[-1]

    # get the treasury yield according to the remaining duration of the bond
    date_pd = pd.Timestamp(date)
    time_remain_month = ((maturity_date - date_pd).days) / 30
    time_remain_year = ((maturity_date - date_pd).days) / 365

    if time_remain_year <= 1:
        if time_remain_month <= 1:
            risk_free = yc['1M']
        elif time_remain_month <= 2:
            risk_free = yc['2M']
        elif time_remain_month <= 3:
            risk_free = yc['3M']
        elif time_remain_month <= 6:
            risk_free = yc['6M']
        elif time_remain_month <= 9:
            risk_free = yc['9M']
        else:
            risk_free = yc['1Y']
    else:
        year = math.ceil(time_remain_year)
        risk_free = yc[str(year) + 'Y']
        if str(risk_free) == 'nan':
            risk_free = (yc[str(year - 1) + 'Y'] + yc[str(year + 1) + 'Y']) / 2

    # add risk premium based on credit rating
    # if credit == 'A+':
    #     rf = risk_free + 0.03
    # if credit == 'AA-':
    #     rf = risk_free + 0.027
    # if credit == 'AA':
    #     rf = risk_free + 0.025
    # elif credit == 'AA+':
    #     rf = risk_free + 0.02
    # elif credit == 'AAA-':
    #     rf = risk_free + 0.017
    # elif credit == 'AAA':
    #     rf = risk_free + 0.015

    # get the underlying stock prices
    # start = date[:2] + str(int(date[2:4]) - 3) + date[4:]  
    # 获取当前可转债合约对应的正股的历史数据
    start = str(int(date.split('-')[0]) - 1) + '-' + date.split('-')[1] + '-' + date.split('-')[2]
    # stock_price = rq.get_price([stock_code], start, date, adjust_type='post').dropna(subset=['close'])
    # stock_price_noadj = rq.get_price([stock_code], start, date, adjust_type='none').dropna(subset=['close'])
    stock_price = pd.read_parquet('stock_price.parquet')
    stock_price = stock_price.reset_index()
    stock_price = stock_price[stock_price.order_book_id == stock_code]
    stock_price = stock_price[stock_price.date <= date]
    stock_price = stock_price[stock_price.date >= start]
    spot_price = stock_price['close'].iloc[-1]

    # calculate volatility of return(EMA65)
    stock_price['ret'] = stock_price['close'] / stock_price['close'].shift(1) - 1
    stock_price['vol'] = pd.Series(dtype='float64')
    # for i in range(120, len(stock_price)):
    #     stock_price['vol'].iloc[i] = np.std(stock_price['ret'].iloc[i - 120:i])
    stock_price['vol'] = np.sqrt(252) * stock_price['ret'].rolling(window=120).std()
    stock_price['vol_ema'] = ta.EMA(stock_price['vol'], 65)
    spot_vol = stock_price['vol_ema'].iloc[-1]

    # get the conversion price
    # conversion_price = rq.convertible.get_conversion_price(order_book_id)
    conversion_price = pd.read_parquet('conversion_price.parquet')
    conversion_price = conversion_price.reset_index()
    conversion_price = conversion_price[conversion_price.order_book_id == order_book_id]
    date_pd = pd.Timestamp(date)
    strike_price = -1

    if date_pd >= conversion_price['effective_date'].iloc[-1]:
        strike_price = conversion_price['conversion_price'].iloc[-1]
    else:
        for i in range(len(conversion_price) - 1):
            if date_pd >= conversion_price['effective_date'].iloc[i] and date_pd < \
                    conversion_price['effective_date'].iloc[i + 1]:
                strike_price = conversion_price['conversion_price'].iloc[i]
                break

    if strike_price == -1:
        print(bond_name + ':计算时间在初次转股价格发布日之前')
        sys.exit(0)

    # get the conditions of redemption, or we said call provision
    # redemption = pd.read_excel('赎回条款.xlsx')
    # redemption = redemption.dropna(subset = ['证券代码'])
    # redemption_con = redemption[redemption['证券简称'].str.contains(bond_name)]
    # redemption_time_period = int(redemption_con.iloc[0,6])
    # redemption_count_period = int(redemption_con.iloc[0,7])
    # redemption_percent = redemption_con.iloc[0,8]/100
    redemption_time_period = 30
    redemption_count_period = 15
    redemption_percent = 1.3
    redemption_price = all_info[all_info['symbol'] == bond_name]['redemption_price'].iloc[0]

    # get the conditions of put provision
    # put = pd.read_excel('put.xlsx')
    # put = put.dropna(subset=['证券代码'])
    # put_con = put[put['证券简称'].str.contains(bond_name)]
    # put_start_date = put_con.iloc[0, 3]
    # put_time_period = int(put_con.iloc[0, 5])
    # put_count_period = int(put_con.iloc[0, 6])
    # put_percent = put_con.iloc[0, 7]/100
    put_percent = 0.7
    put_price = 100  # no remaining interest

    # get the number of trading days till maturity(steps of simulation)
    natrual_days = conversion_end_date - date_pd
    natrual_days = natrual_days.days
    n = int(natrual_days / 365 * 252)

    # get the start date of conversion
    conversion_start_days = conversion_start_date - date_pd
    conversion_start_days = conversion_start_days.days
    conversion_start = int(conversion_start_days / 365 * 252)
    # print(conversion_start_date)

    # get the start date of put
    # put_start_days = put_start_date - date_pd
    # put_start_days = put_start_days.days
    # put_start = int(put_start_days/365*252)
    put_start = conversion_start + 30

    # simulate the stock prices
    v = spot_vol
    r = risk_free  # r = rf
    S = spot_price
    AT = 1 / 252
    pv = 0
    # print(conversion_start)

    # 生成股票路径
    # stockvalue = np.zeros((simulations, n))
    # stockvalue[:, 0] = S
    # for i in range(simulations):
    #     for j in range(1, n):
    #         stockvalue[i, j] = stockvalue[i, j - 1] * \
    #                            math.exp((r - 0.5 * v ** 2) * AT + v *
    #                                     math.sqrt(AT) * gauss(0, 1.0))
    stockvalue = S * np.exp(np.cumsum((r - 0.5 * v ** 2) * AT +
                            v * math.sqrt(AT) *
                            np.random.standard_normal((n, simulations)), axis=0))
    stockvalue[0] = S
    stockvalue = stockvalue.T

    conversion_start = conversion_start + 30 if conversion_start > 0 else 30

    # ! calculate the payoff of each path based on the condition of provisions (call, put and reset provisions)
    payoff = np.zeros((simulations, n))
    for i in range(simulations):
        strike = strike_price
        convertion_ratio = 100 / strike
        for j in range(conversion_start, n):
            select = stockvalue[i, j - redemption_time_period:j]
            redemption_check = [x for x in select if x > strike * redemption_percent]

            if len(redemption_check) >= redemption_count_period:
                payoff[i, j] = np.mean(redemption_check) * convertion_ratio # * Call Provision
                payoff[i, n - 1] = payoff[i, j] * np.exp(r * (n - 1 - j) / 252)
                break

            elif j > put_start:
                put_check = [x for x in select if x < strike * put_percent]
                if len(put_check) == 30:
                    prob = random.uniform(0, 1) # * Probability of Reset Provision
                    if prob >= 0.6:
                        payoff[i, j] = put_price # * Put Provision
                        payoff[i, n - 1] = payoff[i, j] * np.exp(r * (n - 1 - j) / 252)
                        break
                    else:
                        strike = min(strike, np.mean(stockvalue[i, j - 20: j]) * 1.1)
                        convertion_ratio = 100 / strike

    # if neither redemption clause nor put clause is triggered, we compare the conversion value of the last date with the redemption price
    payoff = payoff.T
    payoff[-1, payoff[-1, :] == 0] = np.clip(stockvalue.T[-1, payoff[-1, :] == 0] * convertion_ratio + max(time_remain_year, 1) * coupon, a_min = redemption_price, a_max = np.inf)

    # discount the payoffs and calculate the model price
    pv = np.mean(payoff.T[:, n - 1]) * np.exp(r * -n / 252)

    return pv, real_price


def get_table(bond_name):
    """
    function to calculate time series of model price by bond name
    """
    all_info = pd.read_parquet('all_convertible_bond.parquet')
    table = []
    # get bond info
    info = all_info[all_info['symbol'] == bond_name]
    order_book_id = info['order_book_id'].iloc[0]
    conversion_end_date = info['conversion_end_date'].iloc[0]
    conversion_start_date = info['conversion_start_date'].iloc[0]
    value_date = info['value_date'].iloc[0]
    today = pd.Timestamp('2023-06-12') #datetime.date.today()
    if (conversion_end_date - today).days > 30:
        end_date = today
    else:
        end_date = conversion_end_date + pd.Timedelta(days=-30)

    # get the close price of underlying stocks
    # real_price = rq.get_price(order_book_id, start_date=value_date, end_date=end_date, frequency='1d',
    #                           fields=None,
    #                           adjust_type='pre', skip_suspended=False, market='cn')
    real_price = pd.read_parquet('conversible_bond_price.parquet')
    real_price = real_price.reset_index()
    real_price = real_price[real_price['order_book_id'] == order_book_id]
    real_price = real_price[real_price.date >= value_date]
    date_str = real_price.date
    total_length = len(date_str)
    count = 0

    for date in date_str:
        try:
            date = str(date)[:10]
            mc_times = 500
            result = get_monte_carlo_price(all_info, bond_name, date, mc_times)
            table.append([date, bond_name, result[0], result[1]])
            count += 1
            print('Bond: ' + bond_name + ', ' + 'Validation: ' + str(count) + '/' + str(total_length))
        except:
            count += 1
            print('With Error Bond: ' + bond_name + ', ' + 'Validation: ' + str(count) + '/' + str(total_length))
            continue
    print(bond_name + ' is finished')

    return table


if __name__ == '__main__':

    # all = pd.read_parquet('all_convertible_bond.parquet')
    # bond_list = all['symbol']
    # res_list = []
    # for bond_name in bond_list[-5:]:
    #     res_list.append(get_table(bond_name))
    # df = pd.DataFrame()
    # for res in res_list:
    #     b = pd.DataFrame(res)
    #     df = df.append(b)
    # df.to_excel('2023Result.xlsx')

    all = pd.read_parquet('all_convertible_bond.parquet')
    all = all[all['bond_type'] == 'cb']
    # all['date_delta'] = all['maturity_date'] - pd.Timestamp('2023-06-12')
    # all = all[all['date_delta'] >= pd.Timedelta(days=765)]
    all = all[pd.isnull(all['stop_trading_date'])]
    all = all[all['value_date'] > pd.Timestamp('2019-01-01')]
    bond_list = all['symbol']
    
    # date = '2023-03-02'
    # mc_times = 5
    # bond_name = '联诚转债'
    # result1, result2 = get_monte_carlo_price(all, bond_name, date, mc_times)
    # result = get_table('联诚转债')
    # print(result)
    # bond_list = ['宏川转债', '联诚转债']
    
    res_list = []
    number_processes = mp.cpu_count()
    pool = mp.Pool(number_processes)
    pool_outputs = pool.map(get_table, bond_list[-10:])
    pool.close()
    pool.join()
    df = pd.DataFrame()
    for res in pool_outputs:
        tmp = pd.DataFrame(res)
        # df = df.append(tmp)
        df = pd.concat([df, tmp], ignore_index=True)
    df.to_excel('MC_result_Pool2.xlsx')
