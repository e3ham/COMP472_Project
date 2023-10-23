from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar, Any
import random
import requests

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

# create the output file
with open('gameTrace-b-t-100.txt', 'w') as f:
    f.write("")


class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4


class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker


class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3


##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health: int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table: ClassVar[list[list[int]]] = [
        [3, 3, 3, 3, 1],  # AI
        [1, 1, 6, 1, 1],  # Tech
        [9, 6, 1, 6, 1],  # Virus
        [3, 3, 3, 3, 1],  # Program
        [1, 1, 1, 1, 1],  # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table: ClassVar[list[list[int]]] = [
        [0, 1, 1, 0, 0],  # AI
        [3, 0, 0, 3, 3],  # Tech
        [0, 0, 0, 0, 0],  # Virus
        [0, 0, 0, 0, 0],  # Program
        [0, 0, 0, 0, 0],  # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta: int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount


##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row: int = 0
    col: int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
            coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
            coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string() + self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row - dist, self.row + 1 + dist):
            for col in range(self.col - dist, self.col + 1 + dist):
                yield Coord(row, col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row - 1, self.col)
        yield Coord(self.row, self.col - 1)
        yield Coord(self.row + 1, self.col)
        yield Coord(self.row, self.col + 1)

    def iter_all_surrounding(self):
        """Generate all surrounding coordinates, including diagonals."""
        directions = [
            (-1, 0),  # Up
            (1, 0),  # Down
            (0, -1),  # Left
            (0, 1),  # Right
            (-1, -1),  # Up-left diagonal
            (-1, 1),  # Up-right diagonal
            (1, -1),  # Down-left diagonal
            (1, 1)  # Down-right diagonal
        ]
        for dr, dc in directions:
            yield Coord(self.row + dr, self.col + dc)

    @classmethod
    def from_string(cls, s: str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 2:
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src: Coord = field(default_factory=Coord)
    dst: Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string() + " " + self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row, self.dst.row + 1):
            for col in range(self.src.col, self.dst.col + 1):
                yield Coord(row, col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0, col0), Coord(row1, col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0, 0), Coord(dim - 1, dim - 1))

    @classmethod
    def from_string(cls, s: str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
            s = s.replace(sep, "")
        if len(s) == 4:
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None


##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth: int | None = 4
    min_depth: int | None = 2
    max_time: float | None = 5.0
    game_type: GameType = GameType.AttackerVsComp
    alpha_beta: bool = True
    max_turns: int | None = 100
    randomize_moves: bool = True
    broker: str | None = None


##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth: dict[int, int] = field(default_factory=dict)
    total_seconds: float = 0.0


##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played: int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai: bool = True
    _defender_has_ai: bool = True
    cumulative_evals = 0

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim - 1
        self.set(Coord(0, 0), Unit(player=Player.Defender, type=UnitType.AI))
        self.set(Coord(1, 0), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(0, 1), Unit(player=Player.Defender, type=UnitType.Tech))
        self.set(Coord(2, 0), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(0, 2), Unit(player=Player.Defender, type=UnitType.Firewall))
        self.set(Coord(1, 1), Unit(player=Player.Defender, type=UnitType.Program))
        self.set(Coord(md, md), Unit(player=Player.Attacker, type=UnitType.AI))
        self.set(Coord(md - 1, md), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md, md - 1), Unit(player=Player.Attacker, type=UnitType.Virus))
        self.set(Coord(md - 2, md), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md, md - 2), Unit(player=Player.Attacker, type=UnitType.Program))
        self.set(Coord(md - 1, md - 1), Unit(player=Player.Attacker, type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord: Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord: Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord: Coord, unit: Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord, None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord: Coord, health_delta: int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords: CoordPair) -> bool:
        """Validate a move expressed as a CoordPair."""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False

        # Check if there is a unit at the source coordinate
        src_unit = self.get(coords.src)
        if src_unit is None or src_unit.player != self.next_player:
            return False

        # Check if there is no unit at the destination coordinate
        dst_unit = self.get(coords.dst)
        if dst_unit is not None:
            return False

        # Check if AI, Firewall, or Program is engaged in combat
        if src_unit.type in [UnitType.AI, UnitType.Firewall, UnitType.Program]:
            for adj_coord in coords.src.iter_adjacent():
                adj_unit = self.get(adj_coord)
                if adj_unit and adj_unit.player != self.next_player:
                    return False  # Engaged in combat, can't move

        # Check movement restrictions for AI, Firewall, and Program
        if src_unit.type in [UnitType.AI, UnitType.Firewall, UnitType.Program]:
            if src_unit.player == Player.Attacker:
                if not (coords.src.row >= coords.dst.row and coords.src.col >= coords.dst.col):
                    return False
            else:  # Defender
                if not (coords.src.row <= coords.dst.row and coords.src.col <= coords.dst.col):
                    return False

        # Check for single step movement without diagonal
        row_diff = abs(coords.src.row - coords.dst.row)
        col_diff = abs(coords.src.col - coords.dst.col)
        if (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1):
            return True

        return False

    def perform_move(self, coords: CoordPair) -> Tuple[bool, str]:
        """Validate and perform a move expressed as a CoordPair."""

        print(f"Trying to perform action from {coords.src} to {coords.dst}")

        # If it's a valid move
        if self.is_valid_move(coords):
            print(f"Move from {coords.src} to {coords.dst} is valid")
            self.set(coords.dst, self.get(coords.src))
            self.set(coords.src, None)
            with open('gameTrace-b-t-100.txt', 'a') as f:
                f.write(f"Moved from {coords.src} to {coords.dst}")
            return (True, f"Moved from {coords.src} to {coords.dst}")

        src_unit = self.get(coords.src)
        dst_unit = self.get(coords.dst)

        # Check if there's an actual unit at the source
        if not src_unit:
            return (False, "invalid action: No unit at source")

        # If it's a self-destruct
        if coords.src == coords.dst:
            print(f"Attempting to self-destruct at {coords.src}")
            affected_units = self.self_destruct(coords.src)
            if affected_units and src_unit.player == self.next_player:
                affected_units_str = ', '.join([str(coord) for coord in affected_units])
                with open('gameTrace-b-t-100.txt', 'a') as f:
                    f.write(f"Self-destructed at {coords.src}. Affected units: {affected_units_str}")
                return (True, f"Self-destructed at {coords.src}. Affected units: {affected_units_str}")
            return (False, "Self-destruction failed")  # Explicitly specifying the failure reason

        # If it's an attack
        if dst_unit and src_unit.player != dst_unit.player:
            print(f"Attempting to attack from {coords.src} to {coords.dst}")
            success, message = self.attack(coords.src, coords.dst)
            if success:
                with open('gameTrace-b-t-100.txt', 'a') as f:
                    f.write(f"Attacked from {coords.src} to {coords.dst}")
                return (True, f"Attacked from {coords.src} to {coords.dst}")
            return (False, message)  # Explicitly returning the error from attack method

        # If it's a repair
        if dst_unit and src_unit.player == dst_unit.player:
            print(f"Attempting to repair {coords.dst} using {coords.src}")
            success, message = self.repair(coords.src, coords.dst)
            if success:
                with open('gameTrace-b-t-100.txt', 'a') as f:
                    f.write(f"Repaired {coords.dst} using {coords.src}")
                return (True, f"Repaired {coords.dst} using {coords.src}")
            return (False, message)  # Explicitly returning the error from repair method

        return (False, "invalid action")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        with open('gameTrace-b-t-100.txt', 'a') as f:
            f.write("\n")
            f.write("\n")
            f.write(f"{output}")
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_game_over(self) -> Tuple[bool, str]:
        if not self._attacker_has_ai:
            return (True, f"Game over! Defender wins in {self.turns_played} turns!")
        if not self._defender_has_ai:
            return (True, f"Game over! Attacker wins in {self.turns_played} turns!")
        if not self._attacker_has_ai and not self._defender_has_ai:
            return (True, f"Game over! Defender wins in {self.turns_played} turns!")
        if not self._defender_has_ai:
            return (True, f"Game over! Attacker wins in {self.turns_played} turns!")
        if self.turns_played >= 100:  # Assuming 100 as the limit
            return (True, "Game over! Defender wins due to move limit!")
        return (False, "")

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success, result) = self.perform_move(mv)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                print(f"Player {self.next_player.name}, enter your move or attack (e.g., 'A1 B2'): ", end='')
                move_input = input()
                if move_input.lower() == "exit":
                    exit()  # Allow the user to exit the game
                elif "attack" in move_input.lower():
                    attack_coords = move_input.split()[1:]
                    if len(attack_coords) == 2:
                        src_coord_str, dst_coord_str = attack_coords
                        src_coord = Coord.from_string(src_coord_str)
                        dst_coord = Coord.from_string(dst_coord_str)
                        if src_coord and dst_coord:
                            success, result = self.attack(src_coord, dst_coord)
                            if success:
                                print(result)
                                self.next_turn()
                                break
                            else:
                                print(result)
                        else:
                            print("Invalid coordinates provided for attack.")
                    else:
                        print("Invalid attack command format. Please provide coordinates in the format 'A1 B2'.")
                else:
                    mv = CoordPair.from_string(move_input)
                    (success, result) = self.perform_move(mv)
                    if success:
                        print(result)
                        self.next_turn()
                        break
                    else:
                        print("The move is not valid! Try again!")

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success, result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ", end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord, Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield coord, unit

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        elif self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        elif self._defender_has_ai:
            return Player.Defender
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()

        for (src, _) in self.player_units(self.next_player):
            move.src = src

            # Iterate through all adjacent cells
            for dst in src.iter_adjacent():
                move.dst = dst
                src_unit = self.get(src)
                dst_unit = self.get(dst)

                if self.is_valid_move(move):
                    yield move.clone()
                elif src_unit is not None and dst_unit is not None:
                    if src_unit.player != dst_unit.player:
                        yield move.clone()
                elif src_unit is not None and dst_unit is not None:
                    if src_unit.player == dst_unit.player:
                        yield move.clone()

    def suggest_move(self) -> CoordPair | None:
        start_time = datetime.now()
        score, move, nb_evals = self.minimax_alpha_beta(self.options.max_depth, True, float('-inf'), float('inf'))
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Heuristic score: {score}")
        print(f"Evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{self.options.max_depth - k}:{self.stats.evaluations_per_depth[k]} ", end='')
        print()
        total_evals = sum(self.stats.evaluations_per_depth.values())
        if self.stats.total_seconds > 0:
            print(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s")
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        print(f"Cumulative evaluations: {total_evals}")
        print(f"Cumulative % evals per depth: ", end='')
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{self.options.max_depth - k}:{self.stats.evaluations_per_depth[k]/total_evals*100:0.1f}% ", end='')
        print()
        branching = 0.0
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            branching += self.stats.evaluations_per_depth[k]/self.stats.evaluations_per_depth[k+1]
            if k == len(self.stats.evaluations_per_depth) - 2:
                break
        print(f"Average branching factor: {branching/(self.options.max_depth - 1):0.1f}")

        with open('gameTrace-b-t-100.txt', 'a') as f:
            f.write(f"Heuristic score: {score} \n")
            f.write(f"Evals per depth: ")
            for k in sorted(self.stats.evaluations_per_depth.keys()):
                f.write(f"{self.options.max_depth - k}:{self.stats.evaluations_per_depth[k]} ")
            f.write("\n")
            total_evals = sum(self.stats.evaluations_per_depth.values())
            if self.stats.total_seconds > 0:
                f.write(f"Eval perf.: {total_evals / self.stats.total_seconds / 1000:0.1f}k/s \n")
            f.write(f"Elapsed time: {elapsed_seconds:0.1f}s \n")
            f.write(f"Cumulative evaluations: {total_evals} \n")
            f.write(f"Cumulative % evals per depth: ")
            for k in sorted(self.stats.evaluations_per_depth.keys()):
                f.write(f"{self.options.max_depth - k}:{self.stats.evaluations_per_depth[k]/total_evals*100:0.1f}% ")
            f.write("\n")
            f.write(f"Average branching factor: {branching/(self.options.max_depth - 1):0.1f} \n")

        return move

    def minimax_alpha_beta(self, depth, maximizing_player, alpha, beta) -> tuple[float, CoordPair |
        None, float]:
        nb_evals = 0

        if depth == 0 or self.is_finished():
            # Calculate the heuristic score for this state
            score = self.heuristic_e0(self.next_player)
            current_nb_evals = self.stats.evaluations_per_depth.get(depth, 0)
            new_nb_evals = current_nb_evals + 1
            if depth not in self.stats.evaluations_per_depth:
                self.stats.evaluations_per_depth[depth] = 0
            self.stats.evaluations_per_depth[depth] = new_nb_evals

            return score, None, depth

        if maximizing_player:
            max_eval = float('-inf')
            best_move = None
            avg_depth = 0
            for move in self.move_candidates():
                prev_state = self.get_state()
                #self.perform_move(move)  # Make the move
                evaluate, _, avg_depth = self.minimax_alpha_beta(depth - 1, False, alpha, beta)  # Recurse
                #self.undo_move(move)  # Restore the previous state
                max_eval = max(max_eval, evaluate)
                if evaluate > alpha:
                    alpha = evaluate
                    best_move = move
                    current_nb_evals = self.stats.evaluations_per_depth.get(depth, 0)
                    new_nb_evals = current_nb_evals + 1
                    if depth not in self.stats.evaluations_per_depth:
                        self.stats.evaluations_per_depth[depth] = 0
                    self.stats.evaluations_per_depth[depth] = new_nb_evals

                if beta <= alpha:
                    break  # Pruning

            return max_eval, best_move, avg_depth
        else:
            min_eval = float('inf')
            best_move = None
            avg_depth = 0
            for move in self.move_candidates():
                prev_state = self.get_state()
                #self.perform_move(move)  # Make the move
                evaluate, _, avg_depth = self.minimax_alpha_beta(depth - 1, True, alpha, beta)  # Recurse
                #self.undo_move(move)  # Restore the previous state
                min_eval = min(min_eval, evaluate)
                if evaluate < beta:
                    beta = evaluate
                    best_move = move
                    current_nb_evals = self.stats.evaluations_per_depth.get(depth, 0)
                    new_nb_evals = current_nb_evals + 1
                    if depth not in self.stats.evaluations_per_depth:
                        self.stats.evaluations_per_depth[depth] = 0
                    self.stats.evaluations_per_depth[depth] = new_nb_evals

                if beta <= alpha:
                    break  # Pruning

            return min_eval, best_move, avg_depth

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played + 1:
                        move = CoordPair(
                            Coord(data['from']['row'], data['from']['col']),
                            Coord(data['to']['row'], data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

    def attack(self, src_coord, dst_coord):
        src_unit = self.get(src_coord)
        dst_unit = self.get(dst_coord)

        # Check if the source and destination coordinates are valid
        if not self.is_valid_coord(src_coord) or not self.is_valid_coord(dst_coord):
            return (False, "Invalid coordinates")

        # Check if there is a unit at the source coordinate, and it belongs to the current player
        if src_unit is None or src_unit.player != self.next_player:
            return (False, "Invalid source unit")

        # Check if there is a unit at the destination coordinate, and it belongs to the opposing player
        if dst_unit is None or dst_unit.player == self.next_player:
            return (False, "Invalid target unit")

        # Calculate the damage inflicted by the source unit on the destination unit
        damage = src_unit.damage_amount(dst_unit)

        # Modify the health of the target unit and the source unit based on the damage
        self.mod_health(dst_coord, -damage)
        self.mod_health(src_coord, -damage)

        return (True, f"Attacked {dst_coord.to_string()} with {src_coord.to_string()}, damage: {damage}")

    def repair(self, src_coord: Coord, dst_coord: Coord) -> Tuple[bool, str]:
        """Perform a repair from src_coord to dst_coord."""
        src_unit = self.get(src_coord)
        dst_unit = self.get(dst_coord)

        # Check if the source and destination coordinates are valid
        if not self.is_valid_coord(src_coord) or not self.is_valid_coord(dst_coord):
            return (False, "Invalid coordinates")

        # Check if there is a unit at the source coordinate and it belongs to the current player
        if src_unit is None or src_unit.player != self.next_player:
            return (False, "Invalid source unit")

        # Check if there is a unit at the destination coordinate and it belongs to the current player
        if dst_unit is None or dst_unit.player != self.next_player:
            return (False, "Invalid target unit")

        # Calculate the damage inflicted by the source unit on the destination unit
        repair = src_unit.repair_amount(dst_unit)

        # Check if source unit can repair destination unit
        if repair == 0:
            return (False, "Invalid target unit, can't repair this unit")

        # Check if destination unit is already full health
        if dst_unit.health == 9:
            return (False, "Invalid target unit, target already full health")

        # Modify the health of the target unit and the source unit based on the damage
        self.mod_health(dst_coord, +repair)

        return (True, f"Repaired {dst_coord.to_string()} with {src_coord.to_string()}, repaired: {repair}")

    def self_destruct(self, coord: Coord) -> Tuple[bool, str]:
        print(f"Trying to self destruct at {coord}")
        """Perform a self-destruct action on the unit at the given coordinate."""
        # Check if the coordinate is valid
        if not self.is_valid_coord(coord):
            return (False, "Invalid coordinate")

        # Check if there is a unit at the coordinate
        unit = self.get(coord)
        if unit is None or unit.player != self.next_player:
            return (False, "No unit or not your unit to self-destruct")

        if unit.type == UnitType.AI:
            if unit.player == Player.Attacker:
                self._attacker_has_ai = False
                return (True, f"Attacker's unit at {coord.row, coord.col} self-destructed! Game over! Defender wins!")
            elif unit.player == Player.Defender:
                self._defender_has_ai = False
                return (True, f"Defender's unit at {coord.row, coord.col} self-destructed! Game over! Attacker wins!")

        # Inflict damage to all surrounding units
        affected_units = []  # Keep track of affected units
        for adj_coord in coord.iter_all_surrounding():
            if self.is_valid_coord(adj_coord):
                target_unit = self.get(adj_coord)
                if target_unit:  # If there's a unit in the adjacent cell
                    self.mod_health(adj_coord, -2)  # Inflict 2 points of damage
                    affected_units.append(adj_coord)  # Add to the list of affected units

        # Remove the unit that self-destructed
        self.set(coord, None)

        # Create a message for affected units
        affected_units_str = ', '.join([str(c) for c in affected_units])
        return (True, f"Unit at {coord.row, coord.col} self-destructed! Affected units: {affected_units_str}")

    # DEMO ONLY
    def heuristic_e0(self, player):
        VPi = TPi = FPi = PPi = AIPi = 0

        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                if unit.type == UnitType.Virus:
                    VPi += 1
                elif unit.type == UnitType.Tech:
                    TPi += 1
                elif unit.type == UnitType.Firewall:
                    FPi += 1
                elif unit.type == UnitType.Program:
                    PPi += 1
                elif unit.type == UnitType.AI:
                    AIPi += 1

        e0 = (3 * VPi + 3 * TPi + 3 * FPi + 3 * PPi + 9999 * AIPi)

        return e0

    # focus on the total health points of each player's units,
    # favor the player with higher total health points among their units
    def heuristic_e1(self, player):
        # Initialize total health points for each player
        total_health_player = 0
        total_health_opponent = 0

        # Iterate through all units on the board
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None:
                if unit.player == player:
                    total_health_player += unit.health
                else:
                    total_health_opponent += unit.health

        # Calculate e1 using the difference in total health points
        e1 = total_health_player - total_health_opponent
        print(f"Player {player} Health: {total_health_player}, Opponent Health: {total_health_opponent}")
        return e1

    # focus on the number of units on the board for each player,
    # encourages having more units on the board compared to the opponent
    def heuristic_e2(self, player):
        unit_count_player = 0
        unit_count_opponent = 0

        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None:
                if unit.player == player:
                    unit_count_player += 1
                else:
                    unit_count_opponent += 1

        e2 = unit_count_player - unit_count_opponent
        print(f"Player {player} Unit Count: {unit_count_player}, Opponent Unit Count: {unit_count_opponent}")
        return e2

    def get_valid_moves(self):
        valid_moves = []

        # Iterate through the board to find valid moves
        for src_coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            src_unit = self.get(src_coord)
            # Check if there is a unit at the source coordinate
            if src_unit is not None and src_unit.player == self.next_player:
                for dst_coord in src_coord.iter_adjacent():
                    # Check if the destination coordinate is within bounds
                    if self.is_valid_coord(dst_coord):
                        dst_unit = self.get(dst_coord)

                        # Check if the destination coordinate is empty
                        if dst_unit is None:
                            # Append a move representing a move action
                            valid_moves.append((src_coord, dst_coord, "move"))

                        # Check if it's a valid attack action
                        elif dst_unit.player != self.next_player:
                            # Append a move representing an attack action
                            valid_moves.append((src_coord, dst_coord, "attack"))

        return valid_moves  # Make sure this is dedented correctly

    def get_state(self):
        # Create a deep copy of the board state
        return [list(row) for row in self.board]

    def undo_move(self, move: CoordPair):
        # Implement the reverse of a move to restore the previous state
        self.set(move.src, self.get(move.dst))
        self.set(move.dst, None)

##############################################################################################################

def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--attack', type=str, help='perform an attack in the format "A1 B2"')
    args = parser.parse_args()

    # Prompt user for game type directly
    print("Select game type:")
    print("auto (CompVsComp)")
    print("attacker (AttackerVsComp)")
    print("defender (CompVsDefender)")
    print("manual (AttackerVsDefender)")
    game_type_input = input("Enter your choice: ")

    valid_game_types = ["auto", "attacker", "defender", "manual"]
    if game_type_input not in valid_game_types:
        print(f"Invalid game type: {game_type_input}. Please choose from {', '.join(valid_game_types)}")
        exit(1)

    # parse the game type
    if game_type_input == "attacker":
        game_type = GameType.AttackerVsComp
    elif game_type_input == "defender":
        game_type = GameType.CompVsDefender
    elif game_type_input == "manual":
        game_type = GameType.AttackerVsDefender
    else:
        game_type = GameType.CompVsComp

    # set up game options
    options = Options(game_type=game_type)

    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth
    if args.max_time is not None:
        options.max_time = args.max_time
    if args.broker is not None:
        options.broker = args.broker

    # create a new game
    game = Game(options=options)

    with open('gameTrace-b-t-100.txt', 'a') as f:
        f.write("GAME PARAMETERS \n")
        f.write(f"Max number of turns: {game.options.max_turns} \n")
        f.write(f"Play mode: {game_type_input} \n")

        # the main game loop
        while True:
            print(game)
            winner = game.has_winner()
            if winner is not None:
                f.write(f"Game over! {winner.name} wins in {game.turns_played} turns.")
                print(f"Game over! {winner.name} wins in {game.turns_played} turns!")
                break

            if game.options.game_type == GameType.AttackerVsDefender:
                game.human_turn()
            elif game.options.game_type == GameType.AttackerVsComp:
                if game.next_player == Player.Attacker:
                    game.human_turn()
                else:
                    move = game.computer_turn()
                    if move is not None:
                        game.post_move_to_broker(move)
                    else:
                        print("Computer doesn't know what to do!!!")
                        exit(1)
            elif game.options.game_type == GameType.CompVsDefender:
                if game.next_player == Player.Defender:
                    game.human_turn()
                else:
                    move = game.computer_turn()
                    if move is not None:
                        game.post_move_to_broker(move)
                    else:
                        print("Computer doesn't know what to do!!!")
                        exit(1)
            elif game.options.game_type == GameType.CompVsComp:
                move = game.computer_turn()
                if move is not None:
                    game.post_move_to_broker(move)
                else:
                    print("Computer doesn't know what to do!!!")
                    exit(1)
##############################################################################################################

if __name__ == '__main__':
    main()