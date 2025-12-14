"""
Gomoku GUI - Tkinter Version
Supports Player vs AI and AI vs AI gameplay
Uses trained MuZero model
"""

import tkinter as tk
from tkinter import messagebox
import sys
import torch
import numpy as np
import importlib
import pathlib

# Import MuZero modules
import models
from games.gomoku import Game, Gomoku


class GomokuGUI:
    """Tkinter-based Gomoku GUI."""
    
    def __init__(self, root, model_path=None):
        """
        Initialize Gomoku GUI
        
        Args:
            root: Tkinter root window
            model_path: Model checkpoint path (optional)
        """
        self.root = root
        self.root.title("Gomoku AI Battle")
        self.root.resizable(False, False)
        
        # Board settings
        self.board_size = 9
        self.cell_size = 60
        self.margin = 20  # Reduced margin for better proportion
        self.canvas_size = self.board_size * self.cell_size + 2 * self.margin
        
        # Colors
        self.COLOR_BG = "#DCB35C"
        self.COLOR_LINE = "#000000"
        self.COLOR_BLACK = "#000000"
        self.COLOR_WHITE = "#FFFFFF"
        self.COLOR_LAST_MOVE = "#FF0000"
        self.COLOR_BUTTON = "#64C864"
        
        # Game state
        self.game = Game()
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.mode = "player_vs_ai"  # "player_vs_ai" or "ai_vs_ai"
        
        # Load model
        self.model = None
        self.config = None
        if model_path:
            self.load_model(model_path)
        
        # Setup GUI
        self.setup_gui()
        
    def load_model(self, model_path):
        """Load trained model."""
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Load config
            game_module = importlib.import_module("games.gomoku")
            self.config = game_module.MuZeroConfig()
            
            # Create model
            self.model = models.MuZeroNetwork(self.config)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and "weights" in checkpoint:
                # MuZero checkpoint format
                self.model.set_weights(checkpoint["weights"])
                print(f"✓ Model loaded successfully: {model_path}")
                print(f"  Training steps: {checkpoint.get('training_step', 'N/A')}")
                print(f"  Games played: {checkpoint.get('num_played_games', 'N/A')}")
            elif isinstance(checkpoint, dict):
                # Raw weights format (OrderedDict)
                self.model.set_weights(checkpoint)
                print(f"✓ Model loaded successfully (raw weights): {model_path}")
            else:
                raise ValueError("Unknown checkpoint format")
            
            self.model.eval()
            
            # Use CUDA if available
            if torch.cuda.is_available():
                self.model.to(torch.device("cuda"))
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
    
    def setup_gui(self):
        """Setup GUI components."""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        # Info frame
        info_frame = tk.Frame(main_frame)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # Status label
        self.status_label = tk.Label(info_frame, text="Turn: Black (X)", 
                                     font=("Arial", 14, "bold"))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Model status
        model_text = "Model: Loaded" if self.model else "Model: Not Loaded"
        self.model_label = tk.Label(info_frame, text=model_text, 
                                    font=("Arial", 10))
        self.model_label.pack(side=tk.RIGHT, padx=10)
        
        # Canvas for board
        self.canvas = tk.Canvas(main_frame, 
                               width=self.canvas_size, 
                               height=self.canvas_size,
                               bg=self.COLOR_BG, 
                               highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(side=tk.TOP, pady=(10, 0))
        
        # Buttons
        self.reset_button = tk.Button(button_frame, text="Reset", 
                                      command=self.reset_game,
                                      font=("Arial", 12),
                                      bg=self.COLOR_BUTTON,
                                      width=10)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        self.mode_button = tk.Button(button_frame, 
                                     text="Player vs AI",
                                     command=self.toggle_mode,
                                     font=("Arial", 12),
                                     bg=self.COLOR_BUTTON,
                                     width=15)
        self.mode_button.pack(side=tk.LEFT, padx=5)
        
        self.ai_move_button = tk.Button(button_frame, text="AI Move",
                                        command=self.ai_move,
                                        font=("Arial", 12),
                                        bg=self.COLOR_BUTTON,
                                        width=10)
        self.ai_move_button.pack(side=tk.LEFT, padx=5)
        
        # Initially hide AI move button
        if self.mode == "player_vs_ai":
            self.ai_move_button.pack_forget()
        
        # Draw initial board
        self.draw_board()
        self.update_status()
        
        # Center window after all widgets are created
        self.root.update_idletasks()
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"+{x}+{y}")
    
    def draw_board(self):
        """Draw the game board."""
        self.canvas.delete("all")
        
        # Calculate board size and center it on canvas
        board_pixel_size = (self.board_size - 1) * self.cell_size
        offset = (self.canvas_size - board_pixel_size) // 2
        
        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            x = offset + i * self.cell_size
            self.canvas.create_line(x, offset, 
                                   x, offset + board_pixel_size,
                                   fill=self.COLOR_LINE, width=2)
            # Horizontal lines
            y = offset + i * self.cell_size
            self.canvas.create_line(offset, y,
                                   offset + board_pixel_size, y,
                                   fill=self.COLOR_LINE, width=2)
        
        # Update margin for piece drawing
        self.actual_margin = offset
        
        # Draw star points
        star_positions = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        for row, col in star_positions:
            x = self.actual_margin + col * self.cell_size
            y = self.actual_margin + row * self.cell_size
            self.canvas.create_oval(x-4, y-4, x+4, y+4, 
                                   fill=self.COLOR_LINE)
        
        # Draw pieces
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.game.env.board[row][col]
                if piece != 0:
                    x = self.actual_margin + col * self.cell_size
                    y = self.actual_margin + row * self.cell_size
                    radius = self.cell_size // 2 - 3
                    
                    color = self.COLOR_BLACK if piece == 1 else self.COLOR_WHITE
                    self.canvas.create_oval(x-radius, y-radius, 
                                          x+radius, y+radius,
                                          fill=color, 
                                          outline=self.COLOR_LINE, 
                                          width=2)
                    
                    # Mark last move
                    if self.last_move and self.last_move == (row, col):
                        self.canvas.create_oval(x-6, y-6, x+6, y+6,
                                              fill=self.COLOR_LAST_MOVE)
    
    def update_status(self):
        """Update status label."""
        if not self.game_over:
            current_player = "Black (X)" if self.game.env.player == 1 else "White (O)"
            self.status_label.config(text=f"Turn: {current_player}")
        else:
            if self.winner:
                winner_text = "Black (X) Wins!" if self.winner == 1 else "White (O) Wins!"
            else:
                winner_text = "Draw!"
            self.status_label.config(text=winner_text)
    
    def get_board_position(self, x, y):
        """Convert canvas coordinates to board position."""
        margin = getattr(self, 'actual_margin', self.margin)
        col = round((x - margin) / self.cell_size)
        row = round((y - margin) / self.cell_size)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return row, col
        return None
    
    def make_move(self, row, col):
        """Make a move."""
        action = row * self.board_size + col
        
        # Check if legal action
        if action not in self.game.legal_actions():
            return False
        
        # Execute action
        observation, reward, done = self.game.step(action)
        self.last_move = (row, col)
        
        if done:
            self.game_over = True
            if reward > 0:
                # Current player (who just moved) wins
                self.winner = -self.game.env.player  # Player switched after step
        
        self.draw_board()
        self.update_status()
        return True
    
    def ai_move(self):
        """AI makes a move."""
        if self.game_over or not self.model:
            return
        
        try:
            # Get current observation
            observation = self.game.env.get_observation()
            
            # Convert to tensor
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            if torch.cuda.is_available():
                obs_tensor = obs_tensor.cuda()
            
            # Use model to predict
            with torch.no_grad():
                value, reward, policy_logits, encoded_state = self.model.initial_inference(obs_tensor)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            # Get legal actions
            legal_actions = self.game.legal_actions()
            
            # Choose best legal action
            legal_policy = {action: policy[action] for action in legal_actions}
            best_action = max(legal_policy, key=legal_policy.get)
            
            # Execute action
            row = best_action // self.board_size
            col = best_action % self.board_size
            self.make_move(row, col)
            
            print(f"AI move: ({chr(65+row)}{chr(65+col)}), confidence: {legal_policy[best_action]:.2%}")
            
        except Exception as e:
            print(f"✗ AI move error: {e}")
    
    def on_canvas_click(self, event):
        """Handle canvas click."""
        if self.game_over:
            return
        
        # Player vs AI mode - player's turn
        if self.mode == "player_vs_ai":
            pos = self.get_board_position(event.x, event.y)
            if pos:
                row, col = pos
                if self.make_move(row, col):
                    # AI's turn after player
                    if not self.game_over and self.model:
                        self.root.after(300, self.ai_move)
        
        # AI vs AI mode - click to advance
        elif self.mode == "ai_vs_ai":
            # In AI vs AI mode, use button instead
            pass
    
    def reset_game(self):
        """Reset game."""
        self.game = Game()
        self.game_over = False
        self.winner = None
        self.last_move = None
        self.draw_board()
        self.update_status()
    
    def toggle_mode(self):
        """Toggle game mode."""
        if self.mode == "player_vs_ai":
            self.mode = "ai_vs_ai"
            self.mode_button.config(text="AI vs AI")
            self.ai_move_button.pack(side=tk.LEFT, padx=5)
        else:
            self.mode = "player_vs_ai"
            self.mode_button.config(text="Player vs AI")
            self.ai_move_button.pack_forget()
        
        self.reset_game()


def main():
    """Main program."""
    print("=" * 50)
    print("Gomoku AI Battle System (Tkinter)")
    print("=" * 50)
    
    # Find latest model - use the most recent MuZero checkpoint
    model_path = "results/gomoku/2025-12-14--09-15-53/model.checkpoint"
    
    if pathlib.Path(model_path).exists():
        print(f"Found MuZero model: {model_path}")
    else:
        print("No trained MuZero model found")
        print("Please run 'python muzero.py' to train first")
        model_path = None
    
    print("\nControls:")
    print("- Click on board to place piece")
    print("- 'Reset' button to restart game")
    print("- 'Player vs AI' / 'AI vs AI' to switch mode")
    print("- In AI vs AI mode, click 'AI Move' to let AI play")
    print("\nStarting game...\n")
    
    # Create and run GUI
    root = tk.Tk()
    gui = GomokuGUI(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
