import enum
import random


class Action(enum.IntEnum):
    """
    Enum defining possible actions the agent can take.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Simulator:
    """
    Grid world environment simulator for reinforcement learning.
    Grid layout (2 rows × 5 columns, with holes at B2 and B4):
        A1(S)  A2  A3  A4  A5
        B1(T)  --  B3(T)  --  B5(T)
    Where:
    - A1 (0): initial state (S)
    - A2 (1), A3 (2), A4 (3), A5 (4): regular states
    - B1 (5): terminal state, reward = -1
    - B2: does not exist (hole)
    - B3 (6): terminal state, reward = -1
    - B4: does not exist (hole)
    - B5 (7): terminal state, reward = +3
    The environment is stochastic:
    - 0.7 probability that the agent moves in the chosen direction
    - 0.1 probability for each of the remaining 3 directions
    """

    # State to coordinate mapping (row, column)
    STATE_TO_COORD: dict[int, tuple[int, int]] = {
        0: (0, 0),  # A1
        1: (0, 1),  # A2
        2: (0, 2),  # A3
        3: (0, 3),  # A4
        4: (0, 4),  # A5
        5: (1, 0),  # B1
        6: (1, 2),  # B3
        7: (1, 4),  # B5
    }

    # Reverse mapping
    COORD_TO_STATE: dict[tuple[int, int], int] = {v: k for k, v in STATE_TO_COORD.items()}

    # Terminal states and their rewards
    TERMINAL_STATES: dict[int, float] = {
        5: -1.0,  # B1
        6: -1.0,  # B3
        7: 3.0,  # B5
    }

    # Grid dimensions
    ROWS: int = 2
    COLS: int = 5

    # Initial state
    INITIAL_STATE: int = 0  # A1

    def __init__(self, initial_state: int = INITIAL_STATE) -> None:
        """
        Initializes the simulator with an initial state.
        Args:
            initial_state: Initial state of the environment (if None, uses A1)
        """
        self.state: int = initial_state
        self.episode_length: int = 0
        self.max_episode_length: int = 100

    def step(self, action: Action) -> tuple[float, int, bool]:
        """
        Executes an action and updates the internal state.
        Args:
            action: Action chosen by the agent
        Returns:
            Tuple with:
            - reward: Reward obtained in this step
            - new_state: New state of the environment
            - done: Whether the episode is finished
        """
        # Check if state is terminal
        if self.state in self.TERMINAL_STATES:
            # In terminal state, any action returns the same reward and ends the episode
            reward = self.TERMINAL_STATES[self.state]
            self.state = self.INITIAL_STATE  # Reset to initial state
            self.episode_length = 0
            return reward, self.state, True

        # Stochastic action selection (0.7 chosen, 0.1 each of others)
        actual_action = self._get_stochastic_action(action)

        # Update state based on actual action
        self.state = self._update_state(actual_action)

        # Reward calculation - always 0 in this step, because we get the reward
        # only when we take an action IN the terminal state (in the next step())
        reward = 0.0

        # Check if episode is finished
        # Note: done will be True if we entered a terminal state,
        # but the reward is obtained only in the next step
        self.episode_length += 1
        done = self._is_done()

        return reward, self.state, done

    def _get_stochastic_action(self, intended_action: Action) -> Action:
        """
        Returns the actual action taking into account environment stochasticity.
        Args:
            intended_action: Action the agent wants to execute
        Returns:
            Actual action that will be executed
        """
        rand = random.random()

        if rand < 0.7:
            # With 0.7 probability, the chosen action is executed
            return intended_action

        # With 0.3 probability, one of the other actions is chosen
        all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        other_actions = [a for a in all_actions if a != intended_action]
        return random.choice(other_actions)

    def _update_state(self, action: Action) -> int:
        """
        Updates internal state based on action.
        Args:
            action: Action the agent executes
        Returns:
            New state
        """
        # Convert current state to coordinates
        row, col = self.STATE_TO_COORD[self.state]

        # Apply action
        if action == Action.UP:
            new_row, new_col = row - 1, col
        elif action == Action.DOWN:
            new_row, new_col = row + 1, col
        elif action == Action.LEFT:
            new_row, new_col = row, col - 1
        elif action == Action.RIGHT:
            new_row, new_col = row, col + 1
        else:
            new_row, new_col = row, col

        # Check if new position is out of bounds (hit a wall)
        if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
            # Agent stays in the same state
            return self.state

        # Check if new position is a hole (B2 or B4 don't exist)
        if (new_row, new_col) not in self.COORD_TO_STATE:
            # Agent stays in the same state (hit hole like a wall)
            return self.state

        # Convert coordinates back to state
        return self.COORD_TO_STATE[(new_row, new_col)]

    def _compute_reward(self) -> float:
        """
        Calculates reward for current state.
        Returns:
            Reward
        """
        # Check if state is terminal
        if self.state in self.TERMINAL_STATES:
            return self.TERMINAL_STATES[self.state]

        # In all other states there is no reward
        return 0.0

    def _is_done(self) -> bool:
        """
        Checks if episode is finished.
        Returns:
            True if episode is finished, False otherwise
        """
        # Episode ends if we are in a terminal state
        if self.state in self.TERMINAL_STATES:
            return True

        # Or if we reach maximum episode length
        if self.episode_length >= self.max_episode_length:
            return True

        return False

    def reset(self, initial_state: int | None = None) -> int:
        """
        Resets simulator to initial state.
        Args:
            initial_state: Initial state for new episode (if None, uses A1)
        Returns:
            Initial state
        """
        if initial_state is None:
            initial_state = self.INITIAL_STATE
        self.state = initial_state
        self.episode_length = 0
        return self.state

    def get_state_name(self, state: int) -> str:
        """
        Returns state name (e.g., 'A1', 'B2').
        Args:
            state: Numeric state identifier
        Returns:
            State name
        """
        row, col = self.STATE_TO_COORD[state]
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col + 1}"
