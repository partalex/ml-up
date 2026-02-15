import enum
import random


class Action(enum.IntEnum):
    """
    Enum that defines possible actions the agent can perform.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


arrow_map = {Action.UP: '↑', Action.DOWN: '↓', Action.LEFT: '←', Action.RIGHT: '→'}


class LearningRateType(enum.StrEnum):
    VARIABLE = "variable"
    CONSTANT = "constant"


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
    Environment is stochastic:
    - 0.7 probability that agent moves in chosen direction
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
        # Check if state is terminal - should not happen in normal flow
        # as we reset after reaching terminal state
        if self.state in self.TERMINAL_STATES:
            # Safety check - this shouldn't be called
            reward = self.TERMINAL_STATES[self.state]
            self.state = self.INITIAL_STATE
            self.episode_length = 0
            return reward, self.state, True

        # Stochastic action selection (0.7 chosen, 0.1 each of others)
        actual_action = self.get_stochastic_action(action)

        # Update state based on actual action
        new_state = self._update_state(actual_action)
        self.state = new_state

        self.episode_length += 1

        # Reward calculation - check if we entered a terminal state
        if self.state in self.TERMINAL_STATES:
            # We just entered a terminal state - give the reward
            reward = self.TERMINAL_STATES[self.state]
            done = True
        else:
            # Not in terminal state - no reward
            reward = 0.0
            done = self._is_done()

        return reward, self.state, done

    @staticmethod
    def get_stochastic_action(intended_action: Action) -> Action:
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

    def reset(self, initial_state: int = INITIAL_STATE) -> int:
        """
        Resets simulator to initial state.
        Args:
            initial_state: Initial state for new episode (if None, uses A1)
        Returns:
            Initial state
        """
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


if __name__ == "__main__":
    sim = Simulator()

    print("=== Grid World Simulator ===\n")
    print("Grid layout (2×5 with holes at B2 and B4):")
    print("  A1(S)  A2    A3    A4    A5")
    print("  B1(T)  --    B3(T) --    B5(T)")
    print()
    print("Terminal states:")
    print("  B1 (ID=5): reward = -1")
    print("  B3 (ID=6): reward = -1")
    print("  B5 (ID=7): reward = +3")
    print()
    print("Stochasticity:")
    print("  - 0.7 probability: goes in chosen direction")
    print("  - 0.1 probability: goes in each of the other 3 directions")
    print()
    print("⚠️  IMPORTANT: Reward is obtained immediately when entering")
    print("    a terminal state (B1, B3, or B5)!")
    print("\n" + "=" * 60 + "\n")

    # Run episode
    actions = [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.DOWN, Action.RIGHT]

    for step_num in range(25):
        # Agent chooses action
        action = actions[step_num % len(actions)]

        current_state = sim.state
        current_state_name = sim.get_state_name(current_state)

        # Simulator executes action and returns results
        reward, new_state, done = sim.step(action)

        new_state_name = sim.get_state_name(new_state)

        print(f"Step {step_num + 1}:")
        print(f"  State: {current_state_name} (ID={current_state})")
        print(f"  Action: {action.name}")
        print(f"  New state: {new_state_name} (ID={new_state})")

        # Special marker if we entered terminal state
        if new_state in sim.TERMINAL_STATES and reward != 0:
            print(f"  ⚠️  Entered terminal state! Received reward: {reward:.1f}")

        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print()

        if done:
            print(">>> Episode finished! Agent returns to initial state A1.\n")
            if step_num < 24:  # If not at the end, continue with new episode
                print("--- New episode ---\n")

    # Manual reset demonstration
    print("\n" + "=" * 60)
    print("Manual simulator reset...")
    initial = sim.reset()
    print(f"Simulator reset to state: {sim.get_state_name(initial)} (ID={initial})")
