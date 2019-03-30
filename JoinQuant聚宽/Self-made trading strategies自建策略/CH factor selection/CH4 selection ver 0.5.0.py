#CH4 selection ver 0.5.0 
#CH4 选股模型 + 对冲 （+ 指数增强 for ver 1.0.0）
#joinquant  2005-01-04开始才有TR数据

import statsmodels.api as sm
from statsmodels import regression         
import numpy as np
import pandas as pd
import time 
from datetime import date
from jqdata import *
from jqfactor import standardlize,neutralize,winsorize
from dateutil.relativedelta import relativedelta

'''
================================================================================
总体回测前
================================================================================
'''
#总体回测前要做的事情
def initialize(context):
    set_params()        #1设置策参数
    set_variables()     #2设置中间变量
    set_backtest()      #3设置回测条件
    set_benchmark('000300.XSHG')
    set_subportfolios([SubPortfolioConfig(cash=context.portfolio.starting_cash*(1.0/1.3) ,type='stock'),SubPortfolioConfig(cash=context.portfolio.starting_cash*(0.3/1.3),type='index_futures')])
    # 1/3 for indext futures, 2/3 for stock 
    
#1
#设置策参数
def set_params():
    g.tc=20  # 调仓频率
    g.yb=63  # 样本长度
    g.N=15   # 持仓数目
    g.NoF=4  # 三因子模型还是五因子模型
    g.pre_future = ''#储存先前期货合约的名称
    
#2
#设置中间变量
def set_variables():
    g.t=0               #记录连续回测天数
    g.rf=0.031         #无风险利率
    g.if_trade=False    #当天是否交易
    #g.powerup = 3       #指数增强信号发送日
    
    #将2005-01-04至今所有交易日弄成列表输出
    today=date.today()     #取当日时间xxxx-xx-xx
    a=get_all_trade_days() #取所有交易日:[datetime.date(2005, 1, 4)到datetime.date(2016, 12, 30)]
    g.ATD=['']*len(a)      #获得len(a)维的单位向量
    for i in range(0,len(a)):
        g.ATD[i]=a[i].isoformat() #转换所有交易日为iso格式:2005-01-04到2016-12-30
        #列表会取到2016-12-30，现在需要将大于今天的列表全部砍掉
        if today<=a[i]:
            break
    g.ATD=g.ATD[:i]        #iso格式的交易日：2005-01-04至今
    
#3
#设置回测条件
def set_backtest():
    set_option('use_real_price', True) #用真实价格交易
    log.set_level('order', 'error')
    set_slippage(FixedSlippage(0))     #将滑点 设置为0


'''
================================================================================
每天开盘前
================================================================================
'''
#每天开盘前要做的事情
def before_trading_start(context):
    if g.t%g.tc==0:
        #每g.tc天，交易一次行
        g.if_trade=True 
        # 设置手续费与手续费
        set_slip_fee(context) 
        # 设置可行股票池：获得当前开盘的沪深300股票池并剔除当前或者计算样本期间停牌的股票
        g.all_stocks = set_feasible_stocks(get_index_stocks('000300.XSHG'),g.yb,context)
    g.t+=1

    #if g.tc%g.powerup== 
     
    # 指数增强？？



######################################## blocked for construction ################################################



#4 根据不同的时间段设置滑点与手续费
def set_slip_fee(context):
    # 将滑点设置为0
    set_slippage(FixedSlippage(0)) 
    # 根据不同的时间段设置手续费
    dt=context.current_dt
    log.info(type(context.current_dt))
    
    if dt>datetime.datetime(2013,1, 1):
        set_commission(PerTrade(buy_cost=0.0003, sell_cost=0.0013, min_cost=5)) 
        
    elif dt>datetime.datetime(2011,1, 1):
        set_commission(PerTrade(buy_cost=0.001, sell_cost=0.002, min_cost=5))
            
    elif dt>datetime.datetime(2009,1, 1):
        set_commission(PerTrade(buy_cost=0.002, sell_cost=0.003, min_cost=5))
                
    else:
        set_commission(PerTrade(buy_cost=0.003, sell_cost=0.004, min_cost=5))


    if dt>datetime.datetime(2015,9,7):
         g.futures_margin_rate = 0.2 
    else:
         g.futures_margin_rate = 0.1 
    set_option('futures_margin_rate', g.futures_margin_rate)


#5
# 设置可行股票池：
# 过滤掉当日停牌的股票,且筛选出前days天未停牌股票
# 输入：stock_list-list类型,样本天数days-int类型，context（见API）
# 输出：颗星股票池-list类型
def set_feasible_stocks(stock_list,days,context):
    # 得到是否停牌信息的dataframe，停牌的1，未停牌得0
    suspened_info_df = get_price(list(stock_list), start_date=context.current_dt, end_date=context.current_dt, frequency='daily', fields='paused')['paused'].T
    # 过滤停牌股票 返回dataframe
    unsuspened_index = suspened_info_df.iloc[:,0]<1
    # 得到当日未停牌股票的代码list:
    unsuspened_stocks = suspened_info_df[unsuspened_index].index
    # 进一步，筛选出前days天未曾停牌的股票list:
    feasible_stocks=[]
    current_data=get_current_data()
    for stock in unsuspened_stocks:
        if sum(attribute_history(stock, days, unit='1d',fields=('paused'),skip_paused=False))[0]==0:
            feasible_stocks.append(stock)
   
    return feasible_stocks


'''
================================================================================
每天交易时
================================================================================
'''

#每天交易时要做的事情
def handle_data(context, data):
############################# 选股过程：CH4四因子模型（改动版） ###########################
    if g.if_trade==True:
        # 获得调仓日的日期字符串
        todayStr=str(context.current_dt)[0:10]#去掉时分秒，保留年月日
        # 计算每个股票的ai
        ais=CH4(g.all_stocks,getDay(todayStr,-g.yb),getDay(todayStr,-1),g.rf)
        # 依打分排序，当前需要持仓的股票
        stock_sort=ais.sort('score')['code']
        # 计算hedge ratio 和 beta 并返回值
        hedge_ratio, beta = compute_hedge_ratio(context,stock_sort)
      
############################# 对冲 + 票仓资金分配 ########################################
        rebalance(hedge_ratio, beta, stock_sort,context)
        '''为每个持仓股票分配资金 ***ver 0.5.0 update：放弃原先的资金分配公式 使用rebalance
        
        g.everyStock=context.portfolio.portfolio_value/g.N
        order_stock_sell(context,data,stock_sort)      
        order_stock_buy(context,data,stock_sort)       
        '''       
    g.if_trade=False



''' #***ver 0.5.0 update 停用公式

#获得卖出信号，并执行卖出操作
#输入：context, data，已排序股票列表stock_sort-list类型
#输出：none
def order_stock_sell(context,data,stock_sort):
    # 对于不需要持仓的股票，全仓卖出
    for stock in context.portfolio.positions:
        #除去排名前g.N个股票（选股！）
        if stock not in stock_sort[:g.N]:
            stock_sell = stock
            order_target_value(stock_sell, 0)
'''

''' #***ver 0.5.0 update 停用公式
#获得买入信号，并执行买入操作
#输入：context, data，已排序股票列表stock_sort-list类型
#输出：none
def order_stock_buy(context,data,stock_sort):
    # 对于需要持仓的股票，按分配到的份额买入
    for stock in stock_sort:
        stock_buy = stock
        order_target_value(stock_buy, g.everyStock)
'''

#计算对冲比例函数以及beta
def compute_hedge_ratio(context,stocks_in_postions):

    prices = history(g.yb, '1d', 'close', stocks_in_postions)
    # 取指数在样本时间内的价格
    index_prices = list(attribute_history('000300.XSHG', g.yb, '1d', 'close').close)
    # 计算股票在样本时间内的日收益率
    rets = [(prices.iloc[i+1,:]-prices.iloc[i,:])/prices.iloc[i,:] for i in range(g.yb-1)]
    # 计算日收益率平均
    mean_rets = [np.mean(x) for x in rets]
    # 计算指数的日收益率
    index_rets = [(y-x)/x for (x,y) in zip(index_prices[:-1],index_prices[1:])]
    # 计算组合和指数的协方差矩阵
    cov_mat = np.cov(mean_rets, index_rets)
    # 计算组合的系统性风险
    beta = cov_mat[0,1]/cov_mat[1,1]
    # 计算并返回对冲比例
    return 1+beta*g.futures_margin_rate+beta/5, beta ## <--------怎么确定对冲比例的？？？？可以修改对冲比例



# 调仓函数
# 输入对冲比例
def rebalance(hedge_ratio, beta, stock_list,context):
    # 计算资产总价值
    if g.t == 0:
        total_value = context.portfolio.starting_cash
    else:
        total_value = context.portfolio.total_value
    # 计算预期的股票账户价值
    expected_stock_value = total_value/hedge_ratio
    
    # 将两个账户的钱调到预期的水平
    transfer_cash(1, 0, min(context.subportfolios[1].transferable_cash, max(0, expected_stock_value-context.subportfolios[0].total_value)))
    transfer_cash(0, 1, min(context.subportfolios[0].transferable_cash, max(0, context.subportfolios[0].total_value-expected_stock_value)))


    # 计算股票账户价值（预期价值和实际价值其中更小的那个）
    stock_value = min(context.subportfolios[0].total_value, expected_stock_value)
    # 计算相应的期货保证金价值
    futures_margin = stock_value * beta * g.futures_margin_rate
    
    # 调整股票仓位，在 stork_list 里的等权分配
    for stock in context.subportfolios[0].long_positions.keys():
        if stock not in stock_list:
            order_target(stock,0,pindex=0)  
    for stock in stock_list:
        order_target_value(stock, stock_value/len(stock_list), pindex=0)
    for stock in stock_list:
        order_target_value(stock, stock_value/len(stock_list), pindex=0)
    
    # 获取下月连续合约 string
    current_future = get_next_month_future(context,'IF')
    # 如果下月合约和原本持仓的期货不一样
    if g.pre_future!='' and g.pre_future!=current_future:
        # 就把仓位里的期货平仓
        order_target(g.pre_future, 0, side='short', pindex=1)
    # 现有期货合约改为刚计算出来的
    g.pre_future = current_future
    # 获取沪深300价格
    index_price = attribute_history('000300.XSHG',1, '1d', 'close').close.iloc[0]
    # 计算并调整需要的空单仓位
    order_target(current_future, int(futures_margin/(index_price*300*g.futures_margin_rate)), side='short', pindex=1)


# 取下月连续string
# 输入 context 和一个 string，后者是'IF'或'IC'或'IH'
# 输出一 string，如 'IF1509.CCFX'
def get_next_month_future(context, symbol):
    dt = context.current_dt
    month_begin_day = datetime.date(dt.year, dt.month, 1).isoweekday()
    third_friday_date = 20-month_begin_day + 7*(month_begin_day>5)
    # 如果没过第三个星期五或者第三个星期五（包括）至昨日的所有天都停盘
    if dt.day<=third_friday_date or (dt.day>third_friday_date and not any([datetime.date(dt.year, dt.month, third_friday_date+i) in get_all_trade_days() for i in range(dt.day-third_friday_date)])):
        year = str(dt.year+(dt.month)//12)[2:]
        month = str(dt.month%12+1)
    else:
        next_dt = dt + relativedelta(months=2)
        year = str(dt.year+(dt.month)//12)[2:]
        month = str((dt.month)%12+1)
    if len(month)==1:
        month = '0'+month
    return(symbol+year+month+'.CCFX')






#按照CH4规则计算k个参数并且回归，计算出股票的alpha并且输出
#输入：stocks-list类型； begin，end为“yyyy-mm-dd”类型字符串,rf为无风险收益率-double类型
#输出：最后的打分-dataframe类型
def CH4(stocks,begin,end,rf):
    #查询因子的语句
    LoS = len(stocks)
    q = query(
        valuation.code,
        valuation.market_cap.label('MC'),
        (1/valuation.pe_ratio).label("EP"),
        valuation.turnover_ratio.label('TR')
        ).filter(
        valuation.code.in_(stocks) 
    )

 
    df_1 = get_fundamentals(q,begin).sort('MC',ascending = False)[:int(LoS*0.7)]
    df_temp = df_1.copy()
    df_12 = df_1.sort('EP',ascending = False)
    df_13= df_1.sort('TR', ascending = False)
    sel_codes1 = df_1.code
    sel_codes2 = df_12.code
    df_1.index = sel_codes1.values
    df_12.index = sel_codes2.values
    df_1 =  standardlize(neutralize(winsorize(df_1.MC,qrange = [0.05,0.95]),date = begin))
    df_12 = standardlize(neutralize(winsorize(df_12.EP,qrange = [0.05,0.95]),date = begin))
   
    df_1 = pd.DataFrame({'code': sel_codes1, 'MC': list(df_1)},columns =['code', 'MC'])
    df_12 = pd.DataFrame({'code': sel_codes2, 'EP': list(df_12)},columns =['code', 'EP'])

   
    df_14= pd.DataFrame({'code':df_13.code, 'TR':df_13.TR},columns = ['code','TR'])
    sel_codes3 = df_14.code
    df_15= pd.DataFrame({'code':sel_codes3})
    default  = list(ones(int(LoS*0.7)))


    TR_lm_mean = 0
    TR_ly_mean = 0
    
   
    for i in range(-1,-253,-1):
        df_131 = get_fundamentals(q,getDay(begin,i)) 
        df_131.index = df_131.code.values
        df_131 = neutralize(df_131.TR,date = getDay(begin,i))
        if len(df_131) ==0:
            df_131 = pd.DataFrame({'code':sel_codes3, 'TR':default},columns = ['code','TR'] )
        else:
            df_131 = pd.DataFrame({'code':df_131.index, 'TR':list(df_131)},columns = ['code','TR'] )

        df_14 = df_14.merge(df_131, how ='inner', on ='code')
        
    df_16 = pd.merge(df_15,df_14,how ='left',on ='code').fillna(value = 0)
    TR_ly_mean = df_16.iloc[:,1:].apply(lambda x: x.mean(), axis = 1)
    TR_lm_mean = df_16.iloc[:,1:22].apply(lambda x:x.mean(),axis = 1)  
    TR_df = pd.Series(list_divide(list(TR_lm_mean),list(TR_ly_mean))).fillna(0)#.tolist()
    TR_df = standardlize(winsorize(TR_df,qrange = [0.05,0.95]))
    T_df = pd.DataFrame({'code':sel_codes3, 'TR': list(TR_df)},columns =['code','TR']).fillna(value = 0)


    stocks = list(df_temp['code'])
    LoS = int(LoS *0.7)
    median = np.median(df_temp['MC'].values)

    # 选出特征股票组合
    V=df_12['code'][:int(LoS*0.3)]
    M=df_12['code'][int(LoS*0.3):int(LoS*0.7)] 
    G=df_12['code'][int(LoS*0.7):LoS]
    S=df_1['code'][df_1['MC']<= median]
    B=df_1['code'][df_1['MC']> median]
    T1=T_df['code'][:int(LoS*0.3)]
    T2=T_df['code'][int(LoS*0.3):int(LoS*0.7)]
    T3=T_df['code'][int(LoS*0.7):LoS]
    
    # 获得样本期间的股票价格并计算日收益率
    df_2 = get_price(stocks,begin,end,'1d')
    df_3= df_2['close'][:]
    df_4=np.diff(np.log(df_3),axis=0)+0*df_3[1:]
    #求因子的值
   
   
    PMO= sum(df_4[T3].T)/len(V) - sum(df_4[T1].T)/len(G)
    
    VMG = sum(df_4[V].T)/len(V)- sum(df_4[G].T)/len(G)

    SMB = sum(df_4[S].T)/len(S) - sum(df_4[B].T)/len(B)

    dp=get_price('000300.XSHG',begin,end,'1d')['close']
    MKT=diff(np.log(dp))-rf/252
      



    #将因子们计算好并且放好
    X=pd.DataFrame({"PMO":PMO,"VMG":VMG,"SMB":SMB,"MKT":MKT}).fillna(value = 0)
    #取前g.NoF个因子为策略因子
    factor_flag=["PMO","VMG","SMB","MKT"][:g.NoF]
    print factor_flag
    X=X[factor_flag]
    
    # 对样本数据进行线性回归并计算ai 
    t_scores=[0.0]*LoS
    for i in range(LoS):
        t_stock=stocks[i]
        sample=pd.DataFrame()
        t_r=linreg(X,df_4[t_stock]-rf/252,len(factor_flag))
        t_scores[i]=t_r[0]
    
    #这个scores就是alpha 
    scores=pd.DataFrame({'code':stocks,'score':t_scores})

    return scores

#计算两条list之比并返回新list    
def list_divide(a,b):
    c = []
    for i in range(len(a)):
        if b[i] == 0:
            c.append(np.nan)
        else:
            c.append(a[i]/b[i])
    for i in range(len(a)):
        if isnan(c[i]) == True:
            c[i] = max(c)
        
    return c

# 辅助线性回归的函数
# 输入:X:回归自变量 Y:回归因变量 完美支持list,array,DataFrame等三种数据类型
#      columns:X有多少列，整数输入，不输入默认是3（）
# 输出:参数估计结果-list类型
def linreg(X,Y,columns=3):
    X=sm.add_constant(array(X))
    Y=array(Y)
    if len(Y)>1:
        results = regression.linear_model.OLS(Y, X).fit()
        return results.params
    else:
        return [float("nan")]*(columns+1)


# 日期计算之获得某个日期之前或者之后dt个交易日的日期
# 输入:precent-当前日期-字符串（如2016-01-01）
#      dt-整数，如果要获得之前的日期，写负数，获得之后的日期，写正数
# 输出:字符串（如2016-01-01）
def getDay(precent,dt):
    for i in range(0,len(g.ATD)):
        if precent<=g.ATD[i]:
            t_temp = i
            if t_temp+dt>=0:
                return g.ATD[t_temp+dt]#present偏移dt天后的日期
            else:
                t= datetime.datetime.strptime(g.ATD[0],'%Y-%m-%d')+datetime.timedelta(days = dt)
                t_str=datetime.datetime.strftime(t,'%Y-%m-%d')
                return t_str

'''
================================================================================
每天收盘后
================================================================================
'''
#每天收盘后要做的事情
def after_trading_end(context):
    return
# 进行长运算（本模型中不需要）




