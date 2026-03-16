from board import HexBoard
from solution import SmartPlayer


board = HexBoard(size=5)
smart_player = SmartPlayer(player_id=1)

board.place_piece(0, 1, 2) 
r, c = smart_player.play(board)
print(f"SmartPlayer jugó en: ({r}, {c})")



