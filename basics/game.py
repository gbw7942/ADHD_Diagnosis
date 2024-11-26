import random
import json
import os

# Decorator to track game statistics (e.g., total games played)
def track_game_stats(func):
    def wrapper(self, *args, **kwargs):
        self.stats['games_played'] += 1
        return func(self, *args, **kwargs)
    return wrapper

# Generator for hints (one letter at a time)
def hint_generator(word):
    for letter in word:
        yield letter

class WordGuessingGame:
    def __init__(self, word_list, max_incorrect_guesses=6, save_file="game_state.json"):
        self.word_list = word_list
        self.max_incorrect_guesses = max_incorrect_guesses
        self.word = random.choice(word_list).lower()
        self.guesses = []
        self.incorrect_guesses = 0
        self.stats = {"games_played": 0, "games_won": 0, "games_lost": 0}
        self.save_file = save_file
        self.load_game()

    @track_game_stats
    def start(self):
        print("Welcome to the Word Guessing Game!")
        while not self.is_game_over():
            self.display_progress()
            try:
                guess = self.get_guess()
                self.process_guess(guess)
            except ValueError as e:
                print(e)
            self.save_game()
        
        self.display_result()

    def is_game_over(self):
        return self.incorrect_guesses >= self.max_incorrect_guesses or self.is_word_guessed()

    def is_word_guessed(self):
        return all(letter in self.guesses for letter in self.word)

    def get_guess(self):
        guess = input("Guess a letter: ").lower()
        if not guess.isalpha() or len(guess) != 1:
            raise ValueError("Invalid input! Please enter a single letter.")
        if guess in self.guesses:
            raise ValueError("You already guessed that letter!")
        return guess

    def process_guess(self, guess):
        self.guesses.append(guess)
        if guess in self.word:
            print(f"Good guess! '{guess}' is in the word.")
        else:
            self.incorrect_guesses += 1
            print(f"Sorry! '{guess}' is not in the word. Incorrect guesses: {self.incorrect_guesses}")

    def display_progress(self):
        displayed_word = ''.join([letter if letter in self.guesses else '_' for letter in self.word])
        print(f"Word: {displayed_word}")
        print(f"Incorrect guesses: {self.incorrect_guesses}/{self.max_incorrect_guesses}")

    def display_result(self):
        if self.is_word_guessed():
            self.stats['games_won'] += 1
            print(f"Congratulations! You guessed the word '{self.word}'")
        else:
            self.stats['games_lost'] += 1
            print(f"Game over! The word was '{self.word}'")

        print(f"Games played: {self.stats['games_played']}")
        print(f"Games won: {self.stats['games_won']}")
        print(f"Games lost: {self.stats['games_lost']}")

    def save_game(self):
        game_state = {
            'word': self.word,
            'guesses': self.guesses,
            'incorrect_guesses': self.incorrect_guesses,
            'stats': self.stats
        }
        with open(self.save_file, 'w') as file:
            json.dump(game_state, file)
        print("Game saved.")

    def load_game(self):
        if os.path.exists(self.save_file):
            with open(self.save_file, 'r') as file:
                game_state = json.load(file)
                self.word = game_state['word']
                self.guesses = game_state['guesses']
                self.incorrect_guesses = game_state['incorrect_guesses']
                self.stats = game_state['stats']
            print("Loaded previous game state.")

    def get_hint(self):
        hint_gen = hint_generator(self.word)
        print(f"Hint: The first letter is '{next(hint_gen)}'")

# Word list
words = ["python", "developer", "hangman", "programming", "algorithm"]

# Start the game
if __name__ == "__main__":
    game = WordGuessingGame(words)
    game.start()
