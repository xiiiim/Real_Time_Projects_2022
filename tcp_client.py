import socket
import json
#from messagebox import NO
from my_model import *
from utility import *

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('127.0.0.1', 9995))

my_book = OrderBook()
tm = TradeManager(init_cash = 1000000)
ts = TradingStrategy()

while True:
    msg = s.recv(1023).decode('utf-8').rstrip('\n\r ')
    if not msg:
        break
    else:
        my_book.modify_order(msg)

        tm.update_all(my_book)

        _transaction = ts.handle_message(msg)

        if _transaction is None:
            _order = None
        else:
            _order = tm.handle_my_transaction(_transaction,my_book)

        if _order:
            
            print(_order)
            tm.print_all_stats()
