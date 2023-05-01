from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)

from scipy.optimize import linprog
class NashAgent(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(NashAgent, self).__init__(game_matrix)
        self.optimal_probabilities = self._find_nash_equilibrium(game_matrix)

    def _find_nash_equilibrium(self, game_matrix: np.ndarray) -> np.ndarray:
        n = game_matrix.shape[0]
        c = np.ones(n)
        b_ub = -np.ones(n)
        A_ub = -game_matrix.T
        result = linprog(c, A_ub=A_ub, b_ub=b_ub)
        probabilities = result.x / result.x.sum()
        return probabilities

    def make_move(self) -> int:
        return np.random.choice(self.n_moves, p=self.optimal_probabilities)

    def update_results(self, my_move, other_move):
        pass

    def reset(self):
        pass

class GoldfishAgent(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(GoldfishAgent, self).__init__(game_matrix)
        self.opponent_last_move = None

    def make_move(self) -> int:
        if self.opponent_last_move is None:
            return np.random.randint(0, self.n_moves)
        else:
            # Assume opponent will play the same move again
            expected_payoffs = self.game_matrix[:, self.opponent_last_move]
            return np.argmax(expected_payoffs)

    def update_results(self, my_move, other_move):
        self.opponent_last_move = other_move

    def reset(self):
        self.opponent_last_move = None


class MarkovAgent(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(MarkovAgent, self).__init__(game_matrix)
        self.transition_matrix = self._generate_transition_matrix()
        self.opponent_last_move = None

    def _generate_transition_matrix(self):
        matrix = np.random.rand(self.n_moves, self.n_moves)
        matrix /= matrix.sum(axis=1, keepdims=True)
        return matrix

    def make_move(self) -> int:
        if self.opponent_last_move is None:
            return np.random.randint(0, self.n_moves)
        else:
            # Use Markov process to predict opponent's next move
            opponent_next_move_probabilities = self.transition_matrix[self.opponent_last_move]
            opponent_predicted_move = np.argmax(opponent_next_move_probabilities)
            expected_payoffs = self.game_matrix[:, opponent_predicted_move]
            return np.argmax(expected_payoffs)

    def update_results(self, my_move, other_move):
        self.opponent_last_move = other_move

    def reset(self):
        self.opponent_last_move = None


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]

    return p1_score, p2_score

# # class StudentAgent(IteratedGamePlayer):
# #     """
# #     YOUR DOCUMENTATION GOES HERE!
# #     """
# #     def __init__(self, game_matrix: np.ndarray):
# #         """
# #         Initialize your game playing agent. here
# #         :param game_matrix: square payoff matrix for the game being played.
# #         """
# #         super(StudentAgent, self).__init__(game_matrix)
# #         self.opponent_move_counts = np.zeros(self.n_moves, dtype=int)
# #         self.total_moves = 0
# #         self.adaptive_threshold = 15
# #         pass

# #     def make_move(self) -> int:
# #         """
# #         Play your move based on previous moves or whatever reasoning you want.
# #         :return: an int in (0, ..., n_moves-1) representing your move
# #         """
# #         # YOUR CODE GOES HERE
# #         if self.total_moves < self.adaptive_threshold:
# #             # want to play randomly until we figure out who we are up against
# #             return np.random.randint(0, self.n_moves)
# #         else:
# #             # Exploit weaknesses in opponent now that we've sort of figured out their policy
# #             opponent_probabilities = self.opponent_move_counts / self.total_moves
# #             expected_payoffs = self.game_matrix @ opponent_probabilities
# #             return np.argmax(expected_payoffs)
# #             pass


# #     def update_results(self, my_move, other_move):
# #         """
# #         Update your agent based on the round that was just played.
# #         :param my_move:
# #         :param other_move:
# #         :return: nothing
# #         """
# #         # YOUR CODE GOES HERE
# #         self.opponent_move_counts[other_move] += 1
# #         self.total_moves += 1
# #         pass

# #     def reset(self):
# #         """
# #         This method is called in between opponents (forget memory, etc.).
# #         :return: nothing
# #         """
# #         # YOUR CODE GOES HERE
# #         self.opponent_move_counts.fill(0)
# #         self.total_moves = 0
# #         pass



# # class StudentAgent(IteratedGamePlayer):
# #     '''
# #     Changes: probabilities no longer static

# #     Reinforcement learning now involved - if rewarded, probability of selecting strategy/policy is increased
# #                                         - if punished, probability of selecting strategy/policy is decreased
    
# #     In the beginning, play a uniform distribution p_init (for 35 rounds) of different strategies and see which ones resulted in positive payoff - then adjust probabilities p accordingly'

# #     'predict' best strategy against firstmoveplayer and markov player
# #     - figure out which moves were used most, counter accordingly

# #     'random' - seems to work well against uniform, and nash
# #     - randomly choose a move

# #     'counter_copycat'
# #     - react accordingly to last move

# #     'counter_goldfish'
# #     - if last move is the same as second last move, act accordindly to second lat move
# #     - otherwise, act accordingly by reacting to last move
    


    
# #     '''
# #     def __init__(self, game_matrix: np.ndarray):
# #         super(StudentAgent, self).__init__(game_matrix)
# #         self.opponent_moves_history = np.zeros(self.n_moves, dtype=int)
# #         self.round = 0
# #         self.opponent_last_move = None
# #         self.opponent_second_last_move = None
# #         self.init_p = [0.2, 0.4, 0.2, 0.2]
# #         self.p = [0.2, 0.4, 0.2, 0.7]
        
# #         self.learning_rate = 0.05
# #         self.strategies = ['predict', 'random', 'counter_copycat', 'counter_goldfish']
# #         # self.strategies = ['predict', 'counter', 'random', 'counter_copycat', 'counter_goldfish', 'predict_goldfish', 'predict_markov']
# #         self.last_strategy = None


# #     def make_move(self) -> int:
# #         self.round += 1

# #         # Choose strategy
# #         if self.opponent_last_move is not None:
# #             if self.round <= 35:
# #                 self.last_strategy = np.random.choice(self.strategies, p=self.init_p)
# #             elif self.round > 35:
# #                 self.last_strategy = np.random.choice(self.strategies, p=self.p)
# #         else:
# #             self.last_strategy = np.random.choice(['predict', 'random'], p=[0.7, 0.3])


# #         strategy = self.last_strategy

# #         if strategy == 'predict':
# #             predicted_move = np.argmax(self.opponent_moves_history)
# #             move = (predicted_move + 1) % self.n_moves

# #         elif strategy == 'counter_copycat':
# #             if self.opponent_last_move == self.opponent_second_last_move:
# #                 move = (self.opponent_last_move + 2) % self.n_moves
# #             else:
# #                 move = (self.opponent_last_move + 1) % self.n_moves

# #         elif strategy == 'counter_goldfish':
# #             if self.opponent_second_last_move is not None:
# #                 move = (self.opponent_second_last_move + 1) % self.n_moves
# #             else:
# #                 move = (self.opponent_last_move + 1) % self.n_moves

# #         else:  # strategy == 'random'
# #             move = np.random.randint(0, self.n_moves)

# #         return move

# #     def update_results(self, my_move, other_move):
# #         self.opponent_moves_history[other_move] += 1
# #         self.opponent_second_last_move = self.opponent_last_move
# #         self.opponent_last_move = other_move

# #         # Update probabilities based on the last strategy's performance
# #         payoff = self.game_matrix[my_move, other_move]
# #         strategy_index = self.strategies.index(self.last_strategy)
# #         if payoff > 0:
# #             self.p[strategy_index] += self.learning_rate
# #         # elif payoff < 0 and self.p[strategy_index] - self.learning_rate > 0.0001:
# #         #     self.p[strategy_index] -= self.learning_rate

# #         self.p = [p / sum(self.p) for p in self.p]  # Normalize probabilities
# #         print(self.p)
# #         # print(self.p)



# #     def reset(self):
# #         self.opponent_moves_history = np.zeros(self.n_moves, dtype=int)
# #         self.round = 0
# #         self.opponent_last_move = None
# #         self.opponent_second_last_move = None
# #         self.p = [0.2, 0.2, 0.2, 0.2]






# class StudentAgent(IteratedGamePlayer):
#     '''
#     Changes: probabilities no longer static

#     Reinforcement learning now involved - if rewarded, probability of selecting strategy/policy is increased
#                                         - if punished, probability of selecting strategy/policy is decreased
    
#     Different strategies to cope with different players
    
#     '''
#     def __init__(self, game_matrix: np.ndarray):
#         super(StudentAgent, self).__init__(game_matrix)
#         self.opponent_moves_history = np.zeros(self.n_moves, dtype=int)
#         self.round = 0
#         self.opponent_last_move = None
#         self.opponent_second_last_move = None
#         self.init_p = [0.2, 0.2, 0.2, 0.2, 0.2]
#         self.p = [0.2, 0.2, 0.2, 0.2, 0.7]
        
#         self.learning_rate = 0.05
#         self.strategies = ['predict', 'random', 'counter', 'counter_copycat', 'counter_goldfish']
#         # self.strategies = ['predict', 'counter', 'random', 'counter_copycat', 'counter_goldfish', 'predict_goldfish', 'predict_markov']
#         self.last_strategy = None


#     def make_move(self) -> int:
#         self.round += 1

#         # Choose strategy
#         if self.opponent_last_move is not None:
#             if self.round <= 35:
#                 self.last_strategy = np.random.choice(self.strategies, p=self.init_p)
#             elif self.round > 35:
#                 self.last_strategy = np.random.choice(self.strategies, p=self.p)
#         else:
#             self.last_strategy = np.random.choice(['predict', 'random'], p=[0.7, 0.3])


#         strategy = self.last_strategy

#         if strategy == 'predict':
#             predicted_move = np.argmax(self.opponent_moves_history)
#             move = (predicted_move + 1) % self.n_moves

#         # elif strategy == 'counter':
#         #     move = (self.opponent_last_move + 1) % self.n_moves

#         # elif strategy == 'counter':
#         #     move = (self.opponent_last_move + 2) % self.n_moves

#         elif strategy == 'counter_copycat':
#             if self.opponent_last_move == self.opponent_second_last_move:
#                 move = (self.opponent_last_move + 2) % self.n_moves
#             else:
#                 move = (self.opponent_last_move + 1) % self.n_moves

#         elif strategy == 'counter_goldfish':
#             if self.opponent_second_last_move is not None:
#                 move = (self.opponent_second_last_move + 1) % self.n_moves
#             else:
#                 move = (self.opponent_last_move + 1) % self.n_moves

#         else:  # strategy == 'random'
#             move = np.random.randint(0, self.n_moves)

#         return move

#     def update_results(self, my_move, other_move):
#         self.opponent_moves_history[other_move] += 1
#         self.opponent_second_last_move = self.opponent_last_move
#         self.opponent_last_move = other_move

#         # Update probabilities based on the last strategy's performance
#         payoff = self.game_matrix[my_move, other_move]
#         strategy_index = self.strategies.index(self.last_strategy)
#         if payoff > 0:
#             self.p[strategy_index] += self.learning_rate
#         # elif payoff < 0 and self.p[strategy_index] - self.learning_rate > 0.0001:
#         #     self.p[strategy_index] -= self.learning_rate

#         self.p = [p / sum(self.p) for p in self.p]  # Normalize probabilities
#         print(self.p)



#     def reset(self):
#         self.opponent_moves_history = np.zeros(self.n_moves, dtype=int)
#         self.round = 0
#         self.opponent_last_move = None
#         self.opponent_second_last_move = None
#         self.p = [0.2, 0.2, 0.2, 0.2, 0.2]

    

#     # players = {"unif": uniform_player, "first": first_move_player, "nash":nash_player, "markov": markov_player, "copycat":copy_cat, "goldfish":goldfish_player}
#     # for player1 in players.keys():
#     #     for player2 in players.keys():
#     #         player1_score, player2_score = play_game(players[player1], players[player2], game_matrix)

#     #         print(f"{player1} player's score: {player1_score}")
#     #         print(f"{player2} player's score: {player2_score}")
#     #         print("\n\n")
def is_singleMovePlayer(arr):
    return all(elem == arr[0] for elem in arr)

def is_CopyCatPlayer(my_moves, their_moves):
    '''
    
    '''
    for i in range(1, len(my_moves)):
        if my_moves[i-1] != their_moves[i]:
            return False


    return True

def is_GoldfishPlayer(my_moves, their_moves):
    for i in range(1, len(my_moves)):
        if (my_moves[i-1] + 1) % 3 != their_moves[i]:
            return False

    return True


import numpy as np

# def detect_markov_strategy(opponent_moves, n_moves=3, threshold=0.7):
#     move_counts = np.zeros((n_moves, n_moves))

#     for i in range(1, len(opponent_moves)):
#         previous_move = opponent_moves[i-1]
#         current_move = opponent_moves[i]
#         move_counts[previous_move][current_move] += 1

#     # Normalize the move counts to get probabilities
#     row_sums = move_counts.sum(axis=1, keepdims=True)
#     move_probabilities = move_counts / row_sums

#     # Check if there is a strong correlation between the previous and current moves
#     max_correlations = np.max(move_probabilities, axis=1)
#     markov_detected = np.all(max_correlations >= threshold)

#     return markov_detected

class StudentAgent(IteratedGamePlayer):
    """
    YOUR DOCUMENTATION GOES HERE!

    our default play strategy for the first 5 moves is to play a uniform game.
    in the first 5 moves, we attempt to detect SingeMovePlayer, CopyCatPlayer, GoldfishPlayer (with 100% certainty)
        - if any of these players are detected, then we follow the specific strategy to play against such a player

    if it is unclear who we are playing after 5 moves, we keep playing the uniform strategy for everyone


    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.opponent_moves_history = np.zeros(self.n_moves, dtype=int)
        self.round = 0
        self.potentially_markov = 0
        self.cur_strategy = "Uniform"
        self.opponent_last_move = None
        

        self.first_five_moves = {"me":[], "others":[]}
    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        # YOUR CODE GOES HERE
        self.round += 1

        if self.opponent_last_move == None or self.cur_strategy == 'Uniform':
            return np.random.randint(0, self.n_moves)
        
        elif self.cur_strategy == "SingleMovePlayer":
            # return (self.opponent_last_move + 1) % self.n_moves
            predicted_move = np.argmax(self.opponent_moves_history)
            return (predicted_move + 1) % self.n_moves
    
        elif self.cur_strategy == "CopyCatPlayer":
            return (self.my_last_move + 1) % self.n_moves
            
        elif self.cur_strategy == "GoldfishPlayer":
            # if my last move won - then he will change to beat my last move
            # if my last move lost - then he will continue playing the hand
            if self.game_matrix[self.my_last_move, self.opponent_last_move] >= 0:
                # I won last match up
                # in anticipation of that - I play a move to counter his move
                return (self.my_last_move + 2) % self.n_moves
            elif self.game_matrix[self.my_last_move, self.opponent_last_move] < 0:
                # I lost last match up
                return (self.opponent_last_move + 1) % self.n_moves

        
        elif self.cur_strategy == "counterMarkov":
            predicted_move = np.argmax(self.opponent_moves_history)
            return (predicted_move + 1) % self.n_moves

        else: # uniform strategy against all other players
            return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.opponent_last_move = other_move
        self.my_last_move = my_move




        if self.round <= 5:
            self.first_five_moves["me"].append(my_move)
            self.first_five_moves["others"].append(other_move)

        if self.round == 6:
            if is_singleMovePlayer(self.first_five_moves["others"]):
                self.cur_strategy = "SingleMovePlayer"
            elif is_CopyCatPlayer(self.first_five_moves["me"], self.first_five_moves["others"]):
                self.cur_strategy = "CopyCatPlayer"
                print("hi")
            elif is_GoldfishPlayer(self.first_five_moves["me"], self.first_five_moves["others"]):
                self.cur_strategy = "GoldfishPlayer"
            print(self.first_five_moves["me"])
            print(self.first_five_moves["others"])


    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        # YOUR CODE GOES HERE
        self.opponent_moves_history = np.zeros(self.n_moves, dtype=int)
        self.round = 0
        self.potentially_markov = 0
        self.cur_strategy = "Uniform"
        self.opponent_last_move = None
        

        self.first_five_moves = {"me":[], "others":[]}


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    nash_player = NashAgent(game_matrix)
    goldfish_player = GoldfishAgent(game_matrix)
    markov_player = MarkovAgent(game_matrix)
    copy_cat = CopycatPlayer(game_matrix)



    print("first move")
    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score0, first_move_score = play_game(student_player, first_move_player, game_matrix)

    print("Your player's score: {:}".format(student_score0))
    print("First-move player's score: {:}".format(first_move_score))


    print("uniform")
    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score1, first_move_score = play_game(student_player, uniform_player, game_matrix)

    print("Your player's score: {:}".format(student_score1))
    print("Unifrom player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score2, first_move_score = play_game(student_player, nash_player, game_matrix)

    print("Your player's score: {:}".format(student_score2))
    print("Nash player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score3, first_move_score = play_game(student_player, goldfish_player, game_matrix)

    print("Your player's score: {:}".format(student_score3))
    print("Goldfish player's score: {:}".format(first_move_score))


    # Now try your agent

    student_player = StudentAgent(game_matrix)
    student_score4, first_move_score = play_game(student_player, markov_player, game_matrix)

    print("Your player's score: {:}".format(student_score4))
    print("Markov player's score: {:}".format(first_move_score))


    # Now try your agent
    student_player = StudentAgent(game_matrix)
    student_score5, first_move_score = play_game(student_player, copy_cat, game_matrix)

    print("Your player's score: {:}".format(student_score5))
    print("Copycat player's score: {:}".format(first_move_score))



    print(f"sum of scores = {student_score0 + student_score1 + student_score2 + student_score3 + student_score4 + student_score5}")

    


