from framework.vote import Vote
from framework.state import State


class Experience:
    state: State
    actions: [Vote, Vote]
    reward: int
    follow_state: State
    q_values_follow_state: []

    def __init__(self, state: State, actions: [Vote, Vote], reward: int, follow_state: State, q_values_follow_state: []):
        self.state = state
        self.actions = actions
        self.reward = reward
        self.follow_state = follow_state
        self.q_values_follow_state = q_values_follow_state


