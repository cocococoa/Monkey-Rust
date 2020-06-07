use monkey_rust::parser::Parser;
use std::io;

fn prompt(s: &str) -> io::Result<()> {
    use std::io::{stdout, Write};
    let stdout = stdout();
    let mut stdout = stdout.lock();
    stdout.write(s.as_bytes())?;
    stdout.flush()
}

fn main() {
    use std::io::{stdin, BufRead, BufReader};

    let stdin = stdin();
    let stdin = stdin.lock();
    let stdin = BufReader::new(stdin);
    let mut lines = stdin.lines();

    loop {
        prompt(">>> ").unwrap();
        if let Some(Ok(line)) = lines.next() {
            let mut parser = Parser::new(&line);
            parser.parse_program();
            if parser.is_err() {
                println!("{:?}", parser.errors());
            } else {
                for stmt in parser.statements() {
                    println!("{:?}", stmt);
                }
            }
        } else {
            break;
        }
    }
}
