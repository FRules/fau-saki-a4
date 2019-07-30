import random
from collections import deque
from typing import List
import stock_exchange
import numpy as np
from experts.obscure_expert import ObscureExpert
from framework.period import Period
from framework.portfolio import Portfolio
from framework.stock_market_data import StockMarketData
from framework.interface_expert import IExpert
from framework.interface_trader import ITrader
from framework.order import Order, OrderType
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from framework.order import Company
from framework.utils import save_keras_sequential, load_keras_sequential
from framework.logger import logger
from framework.vote import Vote
from enum import Enum


class ActionChoice(Enum):
    """
    Represents if the choice of the action should be decided by largest Q-Value or random
    """
    RANDOM = "random"
    LARGEST_Q = "largestQ"


class DeepQLearningTrader(ITrader):
    """
    Implementation of ITrader based on Deep Q-Learning (DQL).
    """
    RELATIVE_DATA_DIRECTORY = 'traders/dql_trader_data'

    def __init__(self, expert_a: IExpert, expert_b: IExpert, load_trained_model: bool = True,
                 train_while_trading: bool = False, color: str = 'black', name: str = 'dql_trader', ):
        """
        Constructor
        Args:
            expert_a: Expert for stock A
            expert_b: Expert for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save experts, training mode and name
        super().__init__(color, name)
        assert expert_a is not None and expert_b is not None
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.train_while_trading = train_while_trading

        # Actions
        self.actions = [(+0.0, +1.0),
                        (+0.2, +0.8),
                        (+0.4, +0.6),
                        (+0.6, +0.4),
                        (+0.8, +0.2),
                        (+1.0, +0.0),
                        (+0.0, +0.0),
                        (-1.0, -1.0),
                        (-1.0, +1.0),
                        (+1.0, -1.0),
                        (+0.0, -1.0),
                        (-1.0, +0.0)]

        # Parameters for neural network
        self.state_size = 2
        self.action_size = len(self.actions)
        self.hidden_size = 50
        self.discount = 0.5

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None
        self.last_action = None
        self.last_portfolio_value = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.get_name())
            logger.info(f"DQL Trader: Loaded trained model")
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this traders.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.get_name())
        logger.info(f"DQL Trader: Saved trained model")

    def trade(self, portfolio: Portfolio, stock_market_data: StockMarketData) -> List[Order]:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this traders
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        assert portfolio is not None
        assert stock_market_data is not None
        assert stock_market_data.get_companies() == [Company.A, Company.B]

        current_state = get_state(self, stock_market_data)

        if self.train_while_trading and self.last_state is not None:
            reward = get_reward(self, portfolio, stock_market_data)
            self.memory.append((self.last_state, self.last_action, reward, current_state))
            train_neural_net(self)

        action_index = get_index_for_action_to_execute(self, current_state)

        self.last_state = current_state
        self.last_action = action_index
        self.last_portfolio_value = portfolio.get_value(stock_market_data)

        return get_order_list(self, portfolio, stock_market_data)


def get_order_list(self, portfolio: Portfolio, stock_market_data: StockMarketData):
    stock_price_a = stock_market_data[Company.A].get_last()[-1]
    stock_price_b = stock_market_data[Company.B].get_last()[-1]

    order_list = []
    if self.actions[self.last_action][0] > 0:
        amount_to_buy_stock_a = int(portfolio.cash * self.actions[self.last_action][0] // stock_price_a)
        order_list.append(Order(OrderType.BUY, Company.A, amount_to_buy_stock_a))
    elif self.actions[self.last_action][0] < 0:
        # sell everything we have, look at the actions, we don't have -0.8 or sth just -1
        # we don't need any calculation for "amount_to_sell_stock_a"
        order_list.append(Order(OrderType.SELL, Company.A, portfolio.get_stock(Company.A)))

    if self.actions[self.last_action][1] > 0:
        amount_to_buy_stock_b = int(portfolio.cash * self.actions[self.last_action][1] // stock_price_b)
        order_list.append(Order(OrderType.BUY, Company.B, amount_to_buy_stock_b))
    elif self.actions[self.last_action][1] < 0:
        # sell everything we have, look at the actions, we don't have -0.8 or sth just -1
        # we don't need any calculation for "amount_to_sell_stock_b"
        order_list.append(Order(OrderType.SELL, Company.B, portfolio.get_stock(Company.B)))

    return order_list


def get_reward(self, portfolio: Portfolio, stock_market_data: StockMarketData):
    current_portfolio_value = portfolio.get_value(stock_market_data)
    if self.last_portfolio_value < current_portfolio_value:
        return 100 * (current_portfolio_value / self.last_portfolio_value)
    elif self.last_portfolio_value > portfolio.get_value(stock_market_data):
        return -100
    return -20


def get_index_for_action_to_execute(self, current_state: np.ndarray) -> int:
    choice = np.random.choice(a=[ActionChoice.RANDOM, ActionChoice.LARGEST_Q], size=1, p=[self.epsilon, 1 - self.epsilon])[0]

    if choice == ActionChoice.RANDOM and self.train_while_trading:
        action_index = get_random_action_index(self)
    else:
        action_index = get_highest_q_value_action_index(self, current_state)

    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay
    if self.epsilon < self.epsilon_min:
        self.epsilon = self.epsilon_min
    return action_index


def get_random_action_index(self) -> int:
    return np.random.randint(self.action_size)


def get_highest_q_value_action_index(self, current_state: np.ndarray) -> int:
    prediction = self.model.predict(current_state)
    return np.argmax(prediction[0])


def map_expert_vote(vote: Vote):
        if vote == vote.BUY:
            return 1
        elif vote == vote.SELL:
            return 2
        return 3


def get_state(self, stock_market_data: StockMarketData) -> np.ndarray:
        stock_data_a = stock_market_data[Company.A]
        expert_a_vote = map_expert_vote(self.expert_a.vote(stock_data_a))
        stock_data_b = stock_market_data[Company.B]
        expert_b_vote = map_expert_vote(self.expert_b.vote(stock_data_b))

        return np.array([[expert_a_vote, expert_b_vote]])


def train_neural_net(self):
    if len(self.memory) > self.min_size_of_memory_before_training:
        samples = random.sample(self.memory, self.batch_size)

        for old_state, last_action, reward, current_state in samples:
            new_q = reward + self.discount * np.amax(self.model.predict(current_state)[0])
            q_vector = self.model.predict(old_state)
            q_vector[0][last_action] = new_q
            self.model.fit(old_state, q_vector, epochs=1, verbose=1)


# This method retrains the traders from scratch using training data from TRAINING and test data from TESTING
EPISODES = 5
if __name__ == "__main__":
    # Create the training data and testing data
    # Hint: You can crop the training data with training_data.deepcopy_first_n_items(n)
    training_data = StockMarketData([Company.A, Company.B], [Period.TRAINING])
    testing_data = StockMarketData([Company.A, Company.B], [Period.TESTING])

    # Create the stock exchange and one traders to train the net
    stock_exchange = stock_exchange.StockExchange(10000.0)
    training_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), False, True)

    # Save the final portfolio values per episode
    final_values_training, final_values_test = [], []

    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")

        # train the net
        stock_exchange.run(training_data, [training_trader])
        training_trader.save_trained_model()
        final_values_training.append(stock_exchange.get_final_portfolio_value(training_trader))

        # test the trained net
        testing_trader = DeepQLearningTrader(ObscureExpert(Company.A), ObscureExpert(Company.B), True, False)
        stock_exchange.run(testing_data, [testing_trader])
        final_values_test.append(stock_exchange.get_final_portfolio_value(testing_trader))

        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, label='training', color="black")
    plt.plot(final_values_test, label='test', color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.legend(['training', 'test'])
    fig1 = plt.gcf()
    plt.show()
    fig1.savefig("training_nn.png")
