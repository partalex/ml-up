# make int enum for actions
import enum
import random


class Action(enum.IntEnum):
    """
    Enum koji definiše moguće akcije koje agent može izvršavati.
    """
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Simulator:
    """
    Simulator grid world okruženja za reinforcement learning.

    Grid layout (2 reda × 5 kolona, sa rupama na B2 i B4):
        A1(S)  A2  A3  A4  A5
        B1(T)  --  B3(T)  --  B5(T)

    Gde su:
    - A1 (0): početno stanje (S)
    - A2 (1), A3 (2), A4 (3), A5 (4): obična stanja
    - B1 (5): terminalno stanje, nagrada = -1
    - B2: ne postoji (rupa)
    - B3 (6): terminalno stanje, nagrada = -1
    - B4: ne postoji (rupa)
    - B5 (7): terminalno stanje, nagrada = +3

    Okruženje je stohastičko:
    - 0.7 verovatnoća da agent ide u izabranom smeru
    - 0.1 verovatnoća za svaki od preostala 3 smera
    """

    # Mapiranje stanja na koordinate (red, kolona)
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

    # Obrnuto mapiranje
    COORD_TO_STATE: dict[tuple[int, int], int] = {v: k for k, v in STATE_TO_COORD.items()}

    # Terminalna stanja i njihove nagrade
    TERMINAL_STATES: dict[int, float] = {
        5: -1.0,  # B1
        6: -1.0,  # B3
        7: 3.0,  # B5
    }

    # Dimenzije grid-a
    ROWS: int = 2
    COLS: int = 5

    # Početno stanje
    INITIAL_STATE: int = 0  # A1

    def __init__(self, initial_state: int | None = None) -> None:
        """
        Inicijalizuje simulator sa početnim stanjem.

        Args:
            initial_state: Početno stanje okruženja (ako je None, koristi se A1)
        """
        if initial_state is None:
            initial_state = self.INITIAL_STATE
        self.state: int = initial_state
        self.episode_length: int = 0
        self.max_episode_length: int = 100

    def step(self, action: Action) -> tuple[float, int, bool]:
        """
        Izvršava akciju i ažurira interno stanje.

        Args:
            action: Akcija koju agent bira

        Returns:
            Tuple sa:
            - reward: Osvojena nagrada u ovom koraku
            - new_state: Novo stanje okruženja
            - done: Da li je epizoda završena
        """
        # Provera da li je stanje terminalno
        if self.state in self.TERMINAL_STATES:
            # U terminalnom stanju, bilo koja akcija vraća istu nagradu i završava epizodu
            reward = self.TERMINAL_STATES[self.state]
            self.state = self.INITIAL_STATE  # Resetuj na početno stanje
            self.episode_length = 0
            return reward, self.state, True

        # Stohastički izbor akcije (0.7 izabrana, 0.1 svaka od ostalih)
        actual_action = self._get_stochastic_action(action)

        # Ažuriranje stanja na osnovu stvarne akcije
        self.state = self._update_state(actual_action)

        # Računanje nagrade - u ovom koraku je uvek 0, jer nagradu dobijamo
        # tek kada u terminalnom stanju preduzme akciju (u sledećem step()-u)
        reward = 0.0

        # Provera da li je epizoda završena
        # Napomena: done će biti True ako smo ušli u terminalno stanje,
        # ali nagrada se dobija tek u sledećem koraku
        self.episode_length += 1
        done = self._is_done()

        return reward, self.state, done

    def _get_stochastic_action(self, intended_action: Action) -> Action:
        """
        Vraća stvarnu akciju uzimajući u obzir stohastičnost okruženja.

        Args:
            intended_action: Akcija koju agent želi da izvrši

        Returns:
            Stvarna akcija koja će biti izvršena
        """
        rand = random.random()

        if rand < 0.7:
            # Sa verovatnoćom 0.7 izvršava se izabrana akcija
            return intended_action
        else:
            # Sa verovatnoćom 0.3 bira se jedna od ostalih akcija
            all_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
            other_actions = [a for a in all_actions if a != intended_action]
            return random.choice(other_actions)

    def _update_state(self, action: Action) -> int:
        """
        Ažurira interno stanje na osnovu akcije.

        Args:
            action: Akcija koju agent izvršava

        Returns:
            Novo stanje
        """
        # Konvertuj trenutno stanje u koordinate
        row, col = self.STATE_TO_COORD[self.state]

        # Primeni akciju
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

        # Provera da li je nova pozicija van granica (udar u zid)
        if new_row < 0 or new_row >= self.ROWS or new_col < 0 or new_col >= self.COLS:
            # Agent ostaje u istom stanju
            return self.state

        # Provera da li je nova pozicija rupa (B2 ili B4 ne postoje)
        if (new_row, new_col) not in self.COORD_TO_STATE:
            # Agent ostaje u istom stanju (udario u rupu kao u zid)
            return self.state

        # Konvertuj koordinate nazad u stanje
        return self.COORD_TO_STATE[(new_row, new_col)]

    def _compute_reward(self) -> float:
        """
        Računa nagradu za trenutno stanje.

        Returns:
            Nagrada
        """
        # Provera da li je stanje terminalno
        if self.state in self.TERMINAL_STATES:
            return self.TERMINAL_STATES[self.state]

        # U svim ostalim stanjima nema nagrade
        return 0.0

    def _is_done(self) -> bool:
        """
        Proverava da li je epizoda završena.

        Returns:
            True ako je epizoda završena, False inače
        """
        # Epizoda se završava ako smo u terminalnom stanju
        if self.state in self.TERMINAL_STATES:
            return True

        # Ili ako dostignemo maksimalnu dužinu epizode
        if self.episode_length >= self.max_episode_length:
            return True

        return False

    def reset(self, initial_state: int | None = None) -> int:
        """
        Resetuje simulator na početno stanje.

        Args:
            initial_state: Početno stanje za novu epizodu (ako je None, koristi se A1)

        Returns:
            Početno stanje
        """
        if initial_state is None:
            initial_state = self.INITIAL_STATE
        self.state = initial_state
        self.episode_length = 0
        return self.state

    def get_state_name(self, state: int) -> str:
        """
        Vraća ime stanja (npr. 'A1', 'B2').

        Args:
            state: Brojčani identifikator stanja

        Returns:
            Ime stanja
        """
        row, col = self.STATE_TO_COORD[state]
        row_letter = chr(ord('A') + row)
        return f"{row_letter}{col + 1}"
