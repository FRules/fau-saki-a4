import random
from collections import deque
from typing import List
import stock_exchange
import numpy as np
from experts.obscure_expert import ObscureExpert
from framework.vote import Vote
from framework.state import State
from framework.experience import Experience
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
from framework.action_choice import ActionChoice


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

        self.actions = [[Vote.BUY, Vote.BUY],
                        [Vote.BUY, Vote.SELL],
                        [Vote.BUY, Vote.HOLD],
                        [Vote.SELL, Vote.BUY],
                        [Vote.SELL, Vote.SELL],
                        [Vote.SELL, Vote.HOLD],
                        [Vote.HOLD, Vote.BUY],
                        [Vote.HOLD, Vote.SELL],
                        [Vote.HOLD, Vote.HOLD]]

        # Parameters for neural network
        self.state_size = 9
        self.action_size = len(self.actions)
        self.hidden_size = 50

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
        self.last_actions = None
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

        # TODO Compute the current state
        state = State(portfolio, self.expert_a, self.expert_b, stock_market_data)
        input_vector = state.get_input_vector_for_nn()


        # TODO Store state as experience (memory) and train the neural network only if trade() was called before at least once
        if self.last_state is not None:
            reward = 5 if self.last_portfolio_value < portfolio.get_value(stock_market_data) else -1
            self.memory.append(Experience(self.last_state, self.last_actions, reward, state))

        if len(self.memory) >= self.min_size_of_memory_before_training:
            train_neural_net(self)


        # TODO Create actions for current state and decrease epsilon for fewer random actions
        predicted_output = self.model.predict(input_vector)[0]
        orders = get_orders(self, predicted_output_nn=predicted_output, portfolio=portfolio, stock_market_data=stock_market_data)


        # TODO Save created state, actions and portfolio value for the next call of trade()
        self.last_state = state
        self.last_portfolio_value = portfolio.get_value(stock_market_data)
        self.last_actions = orders
        return orders


def get_orders(self, predicted_output_nn, portfolio: Portfolio, stock_market_data: StockMarketData) -> [Vote, Vote]:
    choice = np.random.choice(a=[ActionChoice.RANDOM, ActionChoice.LARGEST_Q], size=1, p=[self.epsilon, 1 - self.epsilon])[0]

    if choice == ActionChoice.LARGEST_Q:
        actions = get_predicted_actions_from_nn_output(self, nn_output=predicted_output_nn)
    else:
        actions = get_random_actions(self)

    orders = []
    price_stock_a = stock_market_data[Company.A].get_last()[1]
    price_stock_b = stock_market_data[Company.B].get_last()[1]

    our_stock_a = portfolio.get_stock(Company.A)
    our_stock_b = portfolio.get_stock(Company.B)

    if actions[0] == Vote.BUY and actions[1] == Vote.BUY:
        # Special Case: If we want to buy both company stocks, we divide our cash value
        # by two and buy as many stocks from a and b as we can
        return [Order(OrderType.BUY, Company.A, (portfolio.cash/2)//price_stock_a),
                Order(OrderType.BUY, Company.B, (portfolio.cash/2)//price_stock_b)]
    if actions[0] == Vote.BUY:
        orders.append(Order(OrderType.BUY, Company.A, portfolio.cash // price_stock_a))
    if actions[1] == Vote.BUY:
        orders.append(Order(OrderType.BUY, Company.B, portfolio.cash // price_stock_b))
    if actions[0] == Vote.SELL:
        orders.append(Order(OrderType.SELL, Company.A, our_stock_a))
    if actions[1] == Vote.SELL:
        orders.append(Order(OrderType.SELL, Company.B, our_stock_b))

    return orders


def get_random_actions(self) -> [Vote, Vote]:
    actions = np.array(self.actions)
    indices = np.arange(len(actions))
    rnd_indices = np.random.choice(indices)

    return actions[rnd_indices]


def get_predicted_actions_from_nn_output(self, nn_output: np.ndarray) -> [Vote, Vote]:
    return self.actions[np.argmax(nn_output)]


def train_neural_net(self):
    copy_memory_as_list = list(self.memory.copy())
    random.shuffle(copy_memory_as_list)
    training_samples = copy_memory_as_list[0:self.batch_size]
    X = []
    for training_sample in training_samples:
        X.append(training_sample.state.get_input_vector_for_nn())
    return None

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
    plt.show()
