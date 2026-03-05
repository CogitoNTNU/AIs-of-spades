# AIs Of Spades  
## Our Goal  
The goal of our project is to create a neural network capable of reaching an intermediate-advanced level in the game of Texas Hold'em Poker.

## Environment  
The environment is built using and modifying the library [pokerenv](https://github.com/trouverun/pokeren).

## Network  

The network is a complex, purpose-built network.

Output:
* A soft-max distribution of the three game options (fold, check/call, bet).
* A vector consisting of two values ​​between 0 and 1 indicating the mean and variance of the bet (0 indicates betting the minimum, 1 indicates all-in).
* A vector consisting of N numbers that will be provided as input to the next iteration; it is not human-readable.

Input:
* The pre-flop, flop, turn, and river cards expressed as a 4x13x4 matrix (4 suits, 13 cards per suit, 4 game states).
* The matrix of previous bets expressed in 6 columns (6 players at the table, normalized for the current player).
* The encoding vector with the table state, a vector for the very first hand (immediately after state.reset). During training, the network will learn to encode information for the future (opponents' behavior).


