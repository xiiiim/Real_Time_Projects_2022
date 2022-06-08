import datetime
import os
import pickle
import numpy as np 
import pandas as pd
from sklearn import linear_model
from sklearn import neighbors
from utility import *
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from math import sqrt

def my_train_model(X,y):
    k = int(sqrt(len(X)))
    knn = neighbors.KNeighborsRegressor(n_neighbors = k, n_jobs = -1)
    knn.fit(X,y)

    return knn



class OrderBook:
    cols = ['Side','Price','Quantity','Exchange','News'] #dataframe cols
    
    def __init__(self,log_file_path = None):
        self.books = {}
        if log_file_path:
            df = pd.read_csv(log_file_path,index_col=0)
            for symbol in df['Symbol'].unique():
                self.books[symbol] = {'Description':df['Description'].iloc[0], 'df':df[df['Symbol']==symbol][OrderBook.cols]}

    def modify_order(self, message):
        if message:
            o = msg_to_dict(message)
          
            if o['Symbol'] not in self.books:
                self.books[o['Symbol']]= {'Description':o['Description'],'df':pd.DataFrame()} 
            
            df = self.books[o['Symbol']]['df'] #dataframe that will record all real-time updates of each stock

            for col in OrderBook.cols:
                df.loc[o['OrderID'],col] = pd.Series(o)[col] 

    def get_mid_price(self, symbol):
        df = self.get_symbol_df(symbol)
        if df is None:
            return 'Symbol does not exist in orderbook'
        if 'B' not in df['Side'].unique():
            return 'Buy side is missing'
        if 'S' not in df['Side'].unique():
            return 'Sell side is missing'
    
        highest_buy_price = df[df['Side'] =='B']['Price'].max()
        lowest_sell_price = df[df['Side'] =='S']['Price'].min()
        return (highest_buy_price + lowest_sell_price)/2

    def process_market_order(self, order):
        df = self.get_symbol_df(order['Symbol'])
        if df is None:
            return None
        
        df = df[df['Side']!= order['Side']]
        if df.empty:
            print('No existing seller/buyer, order cancelled')
            return None
        
        if order['Side'] == 'B':
            ser = df.sort_values('Price', ascending = True).iloc[0]
        else:
            ser = df.sort_values('Price', ascending = False).iloc[0]
        order_id = ser.name

        if ser['Quantity'] < order['Quantity']:
            print('Exceeds required quantity, order cancelled')
            return None

        modified_order = ser.to_dict()
        modified_order['Quantity'] -= order['Quantity']
        modified_order['Action'] = 'M'
        modified_order['OrderID'] = order_id
        modified_order['Symbol'] = order['Symbol']
        return modified_order

    def get_symbol_df(self, symbol):
        if symbol not in self.books:
            return None
        df = self.books[symbol]['df']
        return df

    def export_book(self, folder_path = 'log', custom_name = None):
        if len(self.books):
            df_list = []
            for ticker, d in self.books.items():
                df = d['df']
                df['Symbol'] = ticker
                df['Description'] = d['Description']
                df_list.append(df.copy())
            df_book = pd.concat(df_list)
            if custom_name:
                file_name = custom_name+'.csv'
            else:
                file_name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') +'.csv'
            file_path= os.path.join(folder_path,file_name)
            df_book.to_csv(file_path)

class TradingStrategy:
    def __init__(self, load_models = True, model_folder = 'models'):
        # save all stock models in a dictionary
        self.models = {} 

        if load_models:
            model_list = os.listdir(model_folder) 

            for f_name in model_list: 
                f_path = os.path.join(model_folder,f_name) 
                self.update_model(f_path) 
 
    def update_model(self, file_path):
        '''
        (file_path) will be Symbol.pkl， 例如AAL.pkl, AAPL.pkl, ...
        '''
        f_name = os.path.basename(file_path) 
        symbol = os.path.splitext(f_name)[0] # remove '.pkl' and get the stock symbok

        self.models[symbol] = read_obj(file_path)
    
    def handle_message(self, message):
        _order = msg_to_dict(message)
        _model = self.models[_order['Symbol']]

        X = prepare_X(message,_model)
        y_pred = _model.predict(X)[0]

        y_actual = _order['Price']


        print('Actual Price is ','{:.2f}'.format(y_actual), 'Predicted Price is ','{:.2f}'.format(y_pred))
        
        # threshold can be changed
        if y_actual< y_pred*0.9:
            my_transaction = {'Side':'B', 'Symbol':_order['Symbol'], 'Quantity':100, 'OrderType': 'market'}
        elif y_actual> y_pred*1.01:
            my_transaction = {'Side':'S', 'Symbol':_order['Symbol'], 'Quantity':100, 'OrderType': 'market'}
        else:
            my_transaction = None
        if my_transaction:
            print(my_transaction)

        return my_transaction
class TradeManager:
    def __init__(self,init_cash) -> None:
        df = pd.DataFrame(columns= ['Quantity','Last_Price','Holding_Value'])
        df.index.name = 'Symbol'
        df.loc['Cash','Quantity'] = init_cash
        df.loc['Cash','Last_Price'] = 1
        df.loc['Cash','Holding_Value'] = init_cash
        
        self.init_val = init_cash
        self.portfolio = df.copy()
        self.max_val = init_cash
        self.pnl = 0
        self.max_drawdown_pct = 0

    def handle_my_transaction(self, transaction, orderBook:OrderBook):
        if transaction['OrderType']=='market':
            modified_order = orderBook.process_market_order(transaction)
            if modified_order is None:
                return None

            quantity_changed = transaction['Quantity'] if transaction['Side'] == 'B' else -transaction['Quantity']
            cash_usage = quantity_changed*modified_order['Price']
            available_cash = self.portfolio.loc['Cash','Holding_Value']

            if cash_usage > available_cash:
                print('Not enough cash')
                return None

            sym = transaction['Symbol']
            
            if sym not in self.portfolio.index:
                for i in ['Quantity','Last_Price','Holding_Value']:
                    self.portfolio.loc[sym,i] = 0

            self.portfolio.loc[sym,'Last_Price'] = modified_order['Price']
            self.portfolio.loc[sym,'Quantity'] += quantity_changed
            self.portfolio.loc[sym,'Holding_Value'] += cash_usage
            self.portfolio.loc['Cash','Quantity'] -= cash_usage
            self.portfolio.loc['Cash','Holding_Value'] -= cash_usage

            return modified_order
    
    def update_all(self, orderBook:OrderBook):
        self.update_price(orderBook)
        self.update_holding_value()
        self.update_max_value()
        self.update_max_draw_down()

    def update_price(self, orderBook:OrderBook):
        for symbol in self.portfolio.index:
            if symbol != 'Cash':
                price = orderBook.get_mid_price(symbol)
                if isinstance(price, float):
                    self.portfolio.loc[symbol,'Last_Price'] = price

    def update_holding_value(self):
        self.portfolio['Holding_Value'] = self.portfolio['Quantity'] * self.portfolio['Last_Price']

    def update_max_value(self):
        total_val = self.portfolio['Holding_Value'].sum()
        
        if total_val> self.max_val:
            self.max_val = total_val

    def update_max_draw_down(self):
        total_val = self.portfolio['Holding_Value'].sum()
        drop_pct = total_val/self.max_val -1
        if drop_pct< self.max_drawdown_pct:
            self.max_drawdown_pct =  drop_pct

    def get_pnl(self):
        total_val = self.portfolio['Holding_Value'].sum()
        return total_val - self.init_val
    
    def print_all_stats(self):
        print('Portfolio：')
        print(self.portfolio)

        print('Total PnLs: ', '{:.2f}'.format(self.get_pnl()))
        print('Maximum Drawdown：', '{:.2%}'.format(self.max_drawdown_pct))





if __name__ == '__main__':
    if not os.path.exists('models'):
        os.mkdir('models')

    file_path_train = os.path.join('finance','finance.csv')
    df = pd.read_csv(file_path_train)




    for symbol, group_df in df.groupby('Symbol'):

    

        X = use_dummies(group_df)
        y = group_df['Price']


      
        model = my_train_model(X,y)

     
        f_path = os.path.join('models',symbol+'.pkl')
        pickle_obj(model,f_path)
