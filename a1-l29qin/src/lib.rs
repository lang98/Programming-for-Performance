// Library routines for reading and solving Sudoku puzzles

#![warn(clippy::all)]
pub mod verify;

use std::io::Read;
use std::num::NonZeroU8;
use std::ops::{Rem, Div, Deref};
use std::fs::read;

// Type definition for a 9x9 array that will represent a Sudoku puzzle.
// Entries with None represent unfilled positions in the puzzle.
type Sudoku = [[Option<NonZeroU8>; 9]; 9];

// This function is called by main. It calls solve() to recursively find the solution.
// The puzzle is modified in-place.
pub fn solve_puzzle(puzzle: &mut Sudoku) {
    solve(puzzle, 0, 0);
}

// Fills in the empty positions in the puzzle with the right values, using a
// recursive brute force approach. Modify the puzzle in place. Return true if
// solved successfully, false if unsuccessful. You may modify the function signature
// if you need/wish.
fn solve(puzzle: &mut Sudoku, mut row: usize, mut col: usize) -> bool {
    if row == 9 && col == 0 {
        return true;
    }

    if puzzle[row][col] != None {
        if col == 8 {
            if solve(puzzle, row + 1, 0) {
                return true;
            }
        } else {
            if solve(puzzle, row, col + 1) {
                return true;
            }
        }
        return false;
    }

    for num in 1..10 {
        let v = NonZeroU8::new(num as u8);
        if check_square(puzzle, row, col, v.unwrap()) {
            puzzle[row][col] = v;
            if col == 8 {
                if solve(puzzle, row + 1, 0) {
                    return true;
                }
            } else {
                if solve(puzzle, row, col + 1) {
                    return true;
                }
            }
            puzzle[row][col] = None;
        }
    }

    false
}

// Helper that checks if a specific square in the puzzle can take on
// a given value. Return true if that value is allowed in that square, false otherwise.
// You can choose not to use this if you prefer.
fn check_square(puzzle: &Sudoku, row: usize, col: usize, val: NonZeroU8) -> bool {
    for i in 0..9 {
        if puzzle[i][col] == Some(val) {
            return false;
        }
        if puzzle[row][i] == Some(val) {
            return false;
        }
    }

    // check the other 4 cells in the subgrid
    let r1 = (row + 2) % 3;
    let r2 = (row + 4) % 3;
    let c1 = (col + 2) % 3;
    let c2 = (col + 4) % 3;
    let index_r = 3 * (row / 3);
    let index_c = 3 * (col / 3);
    let checks = [
        puzzle[index_r + r1][index_c + c1],
        puzzle[index_r + r2][index_c + c1],
        puzzle[index_r + r1][index_c + c2],
        puzzle[index_r + r2][index_c + c2],
    ];
    if checks.iter().any(|&x| x == Some(val)) {
        return false;
    }
    true
}

// Helper for printing a sudoku puzzle to stdout for debugging.
pub fn print_puzzle(puzzle: &Sudoku) {
    for row in puzzle.iter() {
        for elem in row.iter() {
            print!("{}", elem.map(|e| (e.get() + b'0') as char).unwrap_or('.'));
        }
        print!("\n");
    }
    print!("\n");
}

// Read the input byte by byte until a complete Sudoku puzzle has been
// read or EOF is reached.  Assumes the input follows the correct format
// (i.e. matching the files in the input folder).
pub fn read_puzzle(reader: &mut impl Read) -> Option<Box<Sudoku>> {
    // Turn the input stream into an iterator of bytes
    let mut bytes = reader.bytes().map(|b| b.expect("input error")).peekable();
    let mut puzzle = Box::new([[None; 9]; 9]);
    let mut x = 0;
    let mut y = 0;

    // Fill in the puzzle matrix. Ignore the non-puzzle input bytes.
    // Go thru the input until we find a puzzle or EOF (None)
    loop {
        match bytes.peek() {
            Some(b'1'..=b'9') | Some(b'.') => {
                if let Some(b) = bytes.peek() {
                    if *b == b'.' {
                        puzzle[y][x] = None;
                    } else {
                        puzzle[y][x] = NonZeroU8::new(*b - 48);
                    }
                }
                bytes.next();
                if x == 8 && y == 8 {
                    break;
                }

                x += 1;
                if x == 9 {
                    x = 0;
                    y += 1;
                }
            },
            None => return None,
            _ => {
                bytes.next();
            }
        }
    }
    Some(puzzle)
}

// Do a simple check that the puzzle is valid.
// Returns true if it is valid, false if it is not.
// (The verifier server doesn't tell you what's wrong so this function can also help you track
// down an error if your puzzles are not being solved correctly.)
pub fn check_puzzle(puzzle: &Sudoku) -> bool {
    let mut row = Vec::new();
    let mut col = Vec::new();
    let mut correct = vec![];
    for i in 1..10 {
        correct.push(NonZeroU8::new(i as u8).unwrap());
    }

    // Checking row and column
    for c in 0..9 {
        row.clear();
        col.clear();
        for r in 0..9 {
            row.push(puzzle[c][r].expect("Cannot unwrap solution cell"));
            col.push(puzzle[r][c].expect("Cannot unwrap solution cell"));
        }
        row.sort();
        col.sort();
        row.dedup();
        col.dedup();
        if row != correct {
            return false;
        }
        if col != correct {
            return false;
        }
    }

    // Checking 9 sub-squares
    let mut sq = Vec::new();
    for c in (0..9).step_by(3) {
        for r in (0..9).step_by(3) {
            sq.clear();
            sq.push(puzzle[c][r].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c][r + 1].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c][r + 2].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c + 1][r].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c + 2][r].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c + 1][r + 1].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c + 1][r + 2].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c + 2][r + 1].expect("Cannot unwrap solution cell"));
            sq.push(puzzle[c + 2][r + 2].expect("Cannot unwrap solution cell"));
            sq.sort();
            sq.dedup();
            if sq != correct {
                return false;
            }
        }
    }
    true
}
