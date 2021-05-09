// Starter code for ECE 459 Lab 1, Winter 2021


#![warn(clippy::all)]
use lab1::*;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

// Either solve input puzzles or verify complete puzzles,
// depending on the argument passed on the command line

fn main() {
    let args: Vec<String> = env::args().collect();  // get command-line args
    if args.len() < 3 {
        eprintln!("Usage: executable [mode] [filename]");
        return;
    }

    let mode = &args[1];  // this will be "solve" or "verify"
    let file_name = &args[2];  // name of file (in inputfiles and solutions)

    // for either solve or verify, we'll need to read the solution file
    let solution_file = File::open::<PathBuf>(["solved", file_name].iter().collect()).expect("cannot open solution file");
    let mut solution_reader = BufReader::new(solution_file);

    if mode == "solve" {
        let mut puzzle_number = 0;

        // set up to read puzzles from the input file
        let puzzle_file = File::open::<PathBuf>(["inputfiles", file_name].iter().collect()).expect("cannot open input file");
        let mut puzzle_reader = BufReader::new(puzzle_file);

        // Read a puzzle, read the solution, solve the puzzle, compare
        while let Some(mut puzzle) = read_puzzle(&mut puzzle_reader) {
            let sln = read_puzzle(&mut solution_reader).unwrap();
            solve_puzzle(&mut puzzle);
            if puzzle != sln {  // see if the solved puzzle matches the solution
                println!("Puzzle {} does not match solution:", puzzle_number);
                print_puzzle(&puzzle);
            } else if !check_puzzle(&puzzle) {  // verify the solved puzzle
                println!("Puzzle {} does not pass verifier:", puzzle_number);
                print_puzzle(&puzzle);
            }
            puzzle_number += 1;
        }
    } else if mode == "verify" {  // verify all the solutions
        // read all the solutions from the file into a vector
        let mut slns = vec![];
        while let Some(puzzle) = read_puzzle(&mut solution_reader) {
            slns.push(puzzle);
        }
        verify::verify_puzzles(slns.into_iter(), 3);
    } else {
        eprintln!("1st argument should be \"solve\" or \"verify\"");
    }
}
