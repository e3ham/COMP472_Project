Comp 472 Project Deliverable 2

Authors: Rohan Pulavarthy - 40087125; Hyeok Lee - 40130526; Eeham Khan - 40189248;

Professor: Leila Kosseim

Description: The core decision-making logic relies on the Minimax algorithm enhanced with Alpha-Beta pruning, allowing for more efficient move evaluations. This algorithm explores potential moves to a specified depth and returns the best possible move based on heuristic evaluations. The code offers three distinct heuristics: the first considers the types of units a player has, assigning weights to each unit type; the second emphasizes the total health points of each player's units; and the third focuses on the number of units a player has on the board.

Utility functions like get_valid_moves, get_state, and undo_move aid the Minimax function. They respectively retrieve all valid moves for the current player, store the current game state, and reverse a move. The main function acts as the game driver, allowing players to choose among different game modes, such as computer versus computer or human versus computer. Depending on the selected mode, the game loop calls either the human player or the computer AI to decide on moves. If the AI is involved, it makes a move based on the Minimax evaluation and potentially communicates with an external game broker.

