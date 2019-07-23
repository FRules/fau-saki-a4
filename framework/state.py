from framework.vote import Vote
from framework.portfolio import Portfolio
from framework.stock_data import StockData
from framework.company import Company
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
import numpy as np


class State:
    portfolio: Portfolio
    vote_expert_a: Vote
    vote_expert_b: Vote
    last_stock_data_a: StockData
    last_stock_data_b: StockData
    prev_stock_data_a: StockData
    prev_stock_data_b: StockData

    def __init__(self, portfolio: Portfolio, expert_a: IExpert, expert_b: IExpert,
                 stock_market_data: StockMarketData):
        self.portfolio = portfolio
        self.last_stock_data_a = stock_market_data[Company.A].get_last()
        self.last_stock_data_b = stock_market_data[Company.B].get_last()

        if stock_market_data[Company.A].get_row_count() >= 2 and \
                stock_market_data[Company.B].get_row_count() >= 2:
            self.prev_stock_data_a = stock_market_data[Company.A].get_from_offset(stock_market_data[Company.A].get_row_count()-2)[0]
            self.prev_stock_data_b = stock_market_data[Company.B].get_from_offset(stock_market_data[Company.B].get_row_count()-2)[0]
        else:
            self.prev_stock_data_a = None
            self.prev_stock_data_b = None
        self.vote_expert_a = expert_a.vote(stock_market_data[Company.A])
        self.vote_expert_b = expert_b.vote(stock_market_data[Company.B])

    def get_input_vector_for_nn(self) -> np.ndarray:
        vector = []
        if self.vote_expert_a == Vote.BUY:
            vector.extend([1, 0, 0])
        elif self.vote_expert_a == Vote.SELL:
            vector.extend([0, 1, 0])
        else:
            vector.extend([0, 0, 1])
        if self.vote_expert_b == Vote.BUY:
            vector.extend([1, 0, 0])
        elif self.vote_expert_b == Vote.SELL:
            vector.extend([0, 1, 0])
        else:
            vector.extend([0, 0, 1])
        if self.portfolio.cash > 0:
            vector.append(1)
        else:
            vector.append(0)
        if self.portfolio.get_stock(Company.A) > 0:
            vector.append(1)
        else:
            vector.append(0)
        if self.portfolio.get_stock(Company.B) > 0:
            vector.append(1)
        else:
            vector.append(0)

        if self.prev_stock_data_a is None and self.prev_stock_data_b is None:
            vector.extend([0, 0])
        else:
            vector.extend([
                ((1/self.prev_stock_data_a[1])*self.last_stock_data_a[1])-1,
                ((1/self.prev_stock_data_b[1])*self.last_stock_data_b[1])-1
            ])

        return np.asarray([vector])

