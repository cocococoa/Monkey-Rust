/************** Common **************/
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Loc(usize, usize);
impl Loc {
    pub fn merge(&self, other: &Loc) -> Loc {
        use std::cmp::{max, min};
        Loc(min(self.0, other.0), max(self.1, other.1))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Annot<T> {
    value: T,
    loc: Loc,
}
impl<T> Annot<T> {
    pub fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }
}

/************** Lexer **************/
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenKind {
    EOF,

    // identifier
    IDENT(Vec<u8>),
    NUM(u64),

    // operator
    ASSIGN,
    PLUS,
    MINUS,
    BANG,
    ASTERISK,
    SLASH,

    // compare
    LT,
    GT,
    EQ,
    NOTEQ,

    // delimiter
    COMMA,
    SEMICOLON,
    LPAREN,
    RPAREN,
    LBRACE,
    RBRACE,

    // keyword
    FUNCTION,
    LET,
    TRUE,
    FALSE,
    IF,
    ELSE,
    RETURN,
}
type Token = Annot<TokenKind>;
impl Token {
    pub fn eof(loc: Loc) -> Self {
        Token::new(TokenKind::EOF, loc)
    }
    // identifier
    pub fn ident(ident: &[u8], loc: Loc) -> Self {
        Token::new(TokenKind::IDENT(ident.to_vec()), loc)
    }
    pub fn number(n: u64, loc: Loc) -> Self {
        Token::new(TokenKind::NUM(n), loc)
    }
    // operator
    pub fn assign(loc: Loc) -> Self {
        Token::new(TokenKind::ASSIGN, loc)
    }
    pub fn plus(loc: Loc) -> Self {
        Token::new(TokenKind::PLUS, loc)
    }
    pub fn minus(loc: Loc) -> Self {
        Token::new(TokenKind::MINUS, loc)
    }
    pub fn bang(loc: Loc) -> Self {
        Token::new(TokenKind::BANG, loc)
    }
    pub fn asterisk(loc: Loc) -> Self {
        Token::new(TokenKind::ASTERISK, loc)
    }
    pub fn slash(loc: Loc) -> Self {
        Token::new(TokenKind::SLASH, loc)
    }
    // compare
    pub fn lower_than(loc: Loc) -> Self {
        Token::new(TokenKind::LT, loc)
    }
    pub fn greater_than(loc: Loc) -> Self {
        Token::new(TokenKind::GT, loc)
    }
    pub fn equal(loc: Loc) -> Self {
        Token::new(TokenKind::EQ, loc)
    }
    pub fn not_equal(loc: Loc) -> Self {
        Token::new(TokenKind::NOTEQ, loc)
    }
    // delimiter
    pub fn comma(loc: Loc) -> Self {
        Token::new(TokenKind::COMMA, loc)
    }
    pub fn semicolon(loc: Loc) -> Self {
        Token::new(TokenKind::SEMICOLON, loc)
    }
    pub fn lparen(loc: Loc) -> Self {
        Token::new(TokenKind::LPAREN, loc)
    }
    pub fn rparen(loc: Loc) -> Self {
        Token::new(TokenKind::RPAREN, loc)
    }
    pub fn lbrace(loc: Loc) -> Self {
        Token::new(TokenKind::LBRACE, loc)
    }
    pub fn rbrace(loc: Loc) -> Self {
        Token::new(TokenKind::RBRACE, loc)
    }
    // keyword
    pub fn function(loc: Loc) -> Self {
        Token::new(TokenKind::FUNCTION, loc)
    }
    pub fn let_token(loc: Loc) -> Self {
        Token::new(TokenKind::LET, loc)
    }
    pub fn true_token(loc: Loc) -> Self {
        Token::new(TokenKind::TRUE, loc)
    }
    pub fn false_token(loc: Loc) -> Self {
        Token::new(TokenKind::FALSE, loc)
    }
    pub fn if_token(loc: Loc) -> Self {
        Token::new(TokenKind::IF, loc)
    }
    pub fn else_token(loc: Loc) -> Self {
        Token::new(TokenKind::ELSE, loc)
    }
    pub fn return_token(loc: Loc) -> Self {
        Token::new(TokenKind::RETURN, loc)
    }

    // utility
    pub fn create_ident_or_keyword(s: &[u8], loc: Loc) -> Self {
        match s {
            &[b'f', b'n'] => Token::function(loc),
            &[b'l', b'e', b't'] => Token::let_token(loc),
            &[b't', b'r', b'u', b'e'] => Token::true_token(loc),
            &[b'f', b'a', b'l', b's', b'e'] => Token::false_token(loc),
            &[b'i', b'f'] => Token::if_token(loc),
            &[b'e', b'l', b's', b'e'] => Token::else_token(loc),
            &[b'r', b'e', b't', b'u', b'r', b'n'] => Token::return_token(loc),
            _ => Token::ident(s, loc),
        }
    }
}

// LexError
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LexErrorKind {
    InvalidChar(char),
}
type LexError = Annot<LexErrorKind>;
impl LexError {
    fn invalid_char(c: char, loc: Loc) -> Self {
        LexError::new(LexErrorKind::InvalidChar(c), loc)
    }
}

// Lexer
pub struct Lexer {
    input: Vec<u8>,
    current_position: usize,
    read_position: usize,
    ch: u8,
}
impl Lexer {
    pub fn new(input: &str) -> Self {
        let mut lexer = Lexer {
            input: input.as_bytes().to_vec(),
            current_position: 0,
            read_position: 0, // (if initialized) read_position == current_position + 1
            ch: b'\0',        // (if initialized) ch == input[current_position]
        };
        lexer.read_char(); // initialize lexer

        lexer
    }
    pub fn next_token(&mut self) -> Result<Token, LexError> {
        while b" \n\t".contains(&self.ch) {
            // skip white space
            self.read_char();
        }
        match self.ch {
            b'\0' => Ok(Token::eof(Loc(
                self.current_position,
                self.current_position,
            ))),
            b'0'..=b'9' => Ok(self.read_int()),
            b'a'..=b'z' | b'A'..=b'Z' => Ok(self.read_identifier()),
            // TODO: マクロ化
            b'+' => {
                self.read_char();
                Ok(Token::plus(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'-' => {
                self.read_char();
                Ok(Token::minus(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'*' => {
                self.read_char();
                Ok(Token::asterisk(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'/' => {
                self.read_char();
                Ok(Token::slash(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'<' => {
                self.read_char();
                Ok(Token::lower_than(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'>' => {
                self.read_char();
                Ok(Token::greater_than(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b',' => {
                self.read_char();
                Ok(Token::comma(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b';' => {
                self.read_char();
                Ok(Token::semicolon(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'(' => {
                self.read_char();
                Ok(Token::lparen(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b')' => {
                self.read_char();
                Ok(Token::rparen(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'{' => {
                self.read_char();
                Ok(Token::lbrace(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'}' => {
                self.read_char();
                Ok(Token::rbrace(Loc(
                    self.current_position - 1,
                    self.current_position,
                )))
            }
            b'=' => {
                self.read_char();
                if self.ch == b'=' {
                    self.read_char();
                    Ok(Token::equal(Loc(
                        self.current_position - 2,
                        self.current_position,
                    )))
                } else {
                    Ok(Token::assign(Loc(
                        self.current_position - 1,
                        self.current_position,
                    )))
                }
            }
            b'!' => {
                self.read_char();
                if self.ch == b'=' {
                    self.read_char();
                    Ok(Token::not_equal(Loc(
                        self.current_position - 2,
                        self.current_position,
                    )))
                } else {
                    Ok(Token::bang(Loc(
                        self.current_position - 1,
                        self.current_position,
                    )))
                }
            }
            _ => {
                self.read_char();
                Err(LexError::invalid_char(
                    self.ch as char,
                    Loc(self.current_position - 1, self.current_position),
                ))
            }
        }
    }
    fn read_char(&mut self) {
        self.ch = if self.read_position >= self.input.len() {
            b'\0'
        } else {
            self.input[self.read_position]
        };
        self.current_position = self.read_position;
        self.read_position += 1;
    }
    fn read_int(&mut self) -> Token {
        use std::str::from_utf8;

        let start = self.current_position;
        while b"0123456789".contains(&self.ch) {
            self.read_char();
        }
        let n = from_utf8(&self.input[start..self.current_position])
            .unwrap()
            .parse()
            .unwrap();

        Token::number(n, Loc(start, self.current_position))
    }
    fn read_identifier(&mut self) -> Token {
        let start = self.current_position;
        while (b'a' <= self.ch && self.ch <= b'z')
            || (b'A' <= self.ch && self.ch <= b'Z')
            || (b'0' <= self.ch && self.ch <= b'9')
            || self.ch == b'_'
        {
            self.read_char();
        }
        Token::create_ident_or_keyword(
            &self.input[start..self.current_position],
            Loc(start, self.current_position),
        )
    }
}

/************** Parser **************/
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Statement {
    Expression(Box<Ast>), // 式文
    Let { ident: Box<Ast>, exp: Box<Ast> },
    Return(Box<Ast>),
}
impl Statement {
    fn exp_stmt(exp: Ast) -> Self {
        Statement::Expression(Box::new(exp))
    }
    fn let_stmt(ident: Ast, exp: Ast) -> Self {
        Statement::Let {
            ident: Box::new(ident),
            exp: Box::new(exp),
        }
    }
    fn return_stmt(exp: Ast) -> Self {
        Statement::Return(Box::new(exp))
    }
    pub fn loc(&self) -> Loc {
        match self {
            Statement::Expression(exp) => exp.loc.clone(),
            Statement::Let { ident: _, exp } => exp.loc.clone(),
            Statement::Return(exp) => exp.loc.clone(),
        }
    }
}

type Statements = Vec<Statement>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AstKind {
    Num(u64),
    Ident(Vec<u8>),
    Boolean(bool),
    Argument(Vec<Ast>),
    UniOp {
        op: UniOp,
        e: Box<Ast>,
    },
    BinOp {
        op: BinOp,
        l: Box<Ast>,
        r: Box<Ast>,
    },
    If {
        cond: Box<Ast>,
        consequence: Statements,
        alternative: Statements,
    },
    Function {
        parameters: Box<Vec<Vec<u8>>>,
        body: Statements,
    },
    Call {
        function: Box<Ast>,
        args: Box<Ast>,
    },
}
type Ast = Annot<AstKind>;
impl Ast {
    fn num(n: u64, loc: Loc) -> Self {
        Self::new(AstKind::Num(n), loc)
    }
    fn ident(ident: &Vec<u8>, loc: Loc) -> Self {
        Self::new(AstKind::Ident(ident.clone()), loc)
    }
    fn boolean(b: bool, loc: Loc) -> Self {
        Self::new(AstKind::Boolean(b), loc)
    }
    fn argument(arg: Vec<Ast>, loc: Loc) -> Self {
        Self::new(AstKind::Argument(arg), loc)
    }
    fn uniop(op: UniOp, e: Ast, loc: Loc) -> Self {
        Self::new(AstKind::UniOp { op, e: Box::new(e) }, loc)
    }
    fn binop(op: BinOp, l: Ast, r: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::BinOp {
                op,
                l: Box::new(l),
                r: Box::new(r),
            },
            loc,
        )
    }
    fn if_ast(cond: Ast, consequence: Statements, alternative: Statements, loc: Loc) -> Self {
        Self::new(
            AstKind::If {
                cond: Box::new(cond),
                consequence,
                alternative,
            },
            loc,
        )
    }
    fn function(parameters: &Vec<Vec<u8>>, body: Statements, loc: Loc) -> Self {
        Self::new(
            AstKind::Function {
                parameters: Box::new(parameters.clone()),
                body: body,
            },
            loc,
        )
    }
    fn call(function: Ast, args: Ast, loc: Loc) -> Self {
        Self::new(
            AstKind::Call {
                function: Box::new(function),
                args: Box::new(args),
            },
            loc,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UniOpKind {
    // PLUS,
    MINUS,
    BANG,
}
type UniOp = Annot<UniOpKind>;
impl UniOp {
    // fn plus(loc: Loc) -> Self {
    //     Self::new(UniOpKind::PLUS, loc)
    // }
    fn minus(loc: Loc) -> Self {
        Self::new(UniOpKind::MINUS, loc)
    }
    fn bang(loc: Loc) -> Self {
        Self::new(UniOpKind::BANG, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BinOpKind {
    ADD,
    SUB,
    MULT,
    DIV,
    LT,
    GT,
    EQ,
    NOTEQ,
}
type BinOp = Annot<BinOpKind>;
impl BinOp {
    fn add(loc: Loc) -> Self {
        Self::new(BinOpKind::ADD, loc)
    }
    fn sub(loc: Loc) -> Self {
        Self::new(BinOpKind::SUB, loc)
    }
    fn mult(loc: Loc) -> Self {
        Self::new(BinOpKind::MULT, loc)
    }
    fn div(loc: Loc) -> Self {
        Self::new(BinOpKind::DIV, loc)
    }
    fn lt(loc: Loc) -> Self {
        Self::new(BinOpKind::LT, loc)
    }
    fn gt(loc: Loc) -> Self {
        Self::new(BinOpKind::GT, loc)
    }
    fn eq(loc: Loc) -> Self {
        Self::new(BinOpKind::EQ, loc)
    }
    fn not_eq(loc: Loc) -> Self {
        Self::new(BinOpKind::NOTEQ, loc)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
enum Precedance {
    LOWEST,
    EQUALS,
    LTGT,
    SUM,
    PRODUCT,
    PREFIX,
    CALL,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ParseError {
    UnexpectedToken(Token),
    NotExpression(Token),
    NotOperator(Token),
    UnclosedOpenParen(Token),
    RedundantExpression(Token),
    LexError(LexError), // 最悪
    SeriousError,       // 最悪
    Eof,
}

// pub enum Error {
//     ParseError(ParseError),
//     LexError(LexError),
// }

pub struct Parser {
    lexer: Lexer,
    errors: Vec<ParseError>,
    statements: Statements,
    cur_token: Token,
    peek_token: Token,
}
// TODO: if式中など、ネストの深いところでエラーが起きるともうダメ
impl Parser {
    pub fn new(input: &str) -> Self {
        let mut parser = Parser {
            lexer: Lexer::new(input),
            errors: vec![],
            statements: vec![],
            cur_token: Token::eof(Loc(0, 0)),
            peek_token: Token::eof(Loc(0, 0)),
        };
        parser.initialize_parser();
        parser
    }
    pub fn is_err(&self) -> bool {
        self.errors.len() != 0
    }
    pub fn errors(&self) -> &Vec<ParseError> {
        &self.errors
    }
    fn initialize_parser(&mut self) {
        let first = self.lexer.next_token();
        if first.is_err() {
            self.errors.push(ParseError::LexError(first.unwrap_err()));
            self.skip_until_next_semicolon();
            self.initialize_parser();
        } else {
            let second = self.lexer.next_token();
            if second.is_err() {
                self.errors.push(ParseError::LexError(first.unwrap_err()));
                self.skip_until_next_semicolon();
                self.initialize_parser();
            } else {
                // finish initialization
                self.cur_token = first.unwrap();
                self.peek_token = second.unwrap();
            }
        }
    }
    fn skip_until_next_semicolon(&mut self) {
        loop {
            match self.lexer.next_token() {
                Ok(token) => {
                    if token.value == TokenKind::SEMICOLON || token.value == TokenKind::EOF {
                        break;
                    }
                }
                Err(err) => {
                    self.errors.push(ParseError::LexError(err));
                }
            }
        }
    }
    fn next_token(&mut self) {
        let past_peek_token =
            std::mem::replace(&mut self.peek_token, self.lexer.next_token().unwrap());
        let _past_token = std::mem::replace(&mut self.cur_token, past_peek_token);
    }
    fn token_precedance(token: &TokenKind) -> Precedance {
        match token {
            TokenKind::EQ => Precedance::EQUALS,
            TokenKind::NOTEQ => Precedance::EQUALS,
            TokenKind::LT => Precedance::LTGT,
            TokenKind::GT => Precedance::LTGT,
            TokenKind::PLUS => Precedance::SUM,
            TokenKind::MINUS => Precedance::SUM,
            TokenKind::SLASH => Precedance::PRODUCT,
            TokenKind::ASTERISK => Precedance::PRODUCT,
            TokenKind::LPAREN => Precedance::CALL,
            _ => Precedance::LOWEST,
        }
    }
    fn peek_precedance(&self) -> Precedance {
        Self::token_precedance(&self.peek_token.value)
    }
    fn cur_precedance(&self) -> Precedance {
        Self::token_precedance(&self.cur_token.value)
    }
    fn consume_peek(&mut self, tok: TokenKind) -> Result<(), ParseError> {
        if self.peek_token.value == tok {
            self.next_token();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(self.peek_token.clone()))
        }
    }
    fn consume_current(&mut self, tok: TokenKind) -> Result<(), ParseError> {
        if self.cur_token.value == tok {
            self.next_token();
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(self.cur_token.clone()))
        }
    }
    // fn expect_peek(&mut self, tok: TokenKind) -> Result<(), ParseError> {
    //     if self.peek_token.value == tok {
    //         Ok(())
    //     } else {
    //         Err(ParseError::UnexpectedToken(self.peek_token.clone()))
    //     }
    // }
    fn expect_current(&mut self, tok: TokenKind) -> Result<(), ParseError> {
        if self.cur_token.value == tok {
            Ok(())
        } else {
            Err(ParseError::UnexpectedToken(self.cur_token.clone()))
        }
    }
    pub fn parse_program(&mut self) {
        match self.parse_statements() {
            Ok(statements) => self.statements = statements,
            Err(error) => {
                self.errors.push(error);
            }
        }
    }
    pub fn parse_statements(&mut self) -> Result<Statements, ParseError> {
        let mut ret = vec![];
        loop {
            if self.cur_token.value == TokenKind::EOF {
                break;
            } else {
                match self.parse_statement() {
                    Ok(stmt) => ret.push(stmt),
                    Err(_) => return Err(ParseError::SeriousError),
                }
            }
        }
        Ok(ret)
    }
    pub fn parse_blocked_statements(&mut self) -> Result<Statements, ParseError> {
        let mut ret = vec![];
        loop {
            if self.peek_token.value == TokenKind::EOF {
                return Err(ParseError::SeriousError);
            } else {
                match self.parse_statement() {
                    Ok(stmt) => ret.push(stmt),
                    Err(_) => return Err(ParseError::SeriousError),
                }
            }
            if self.cur_token.value == TokenKind::RBRACE {
                break;
            }
        }
        Ok(ret)
    }
    pub fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.cur_token.value {
            TokenKind::LET => self.parse_let_statement(),
            TokenKind::RETURN => self.parse_return_statement(),
            _ => self.parse_expression_statement(),
        }
    }
    fn parse_let_statement(&mut self) -> Result<Statement, ParseError> {
        self.consume_current(TokenKind::LET)?;
        let ident = self.parse_ident()?;
        self.next_token();
        self.consume_current(TokenKind::ASSIGN)?;
        let exp = self.parse_expression(Precedance::LOWEST)?;
        self.consume_peek(TokenKind::SEMICOLON)?;
        self.next_token();
        Ok(Statement::let_stmt(ident, exp))
    }
    fn parse_return_statement(&mut self) -> Result<Statement, ParseError> {
        self.consume_current(TokenKind::RETURN)?;
        let exp = self.parse_expression(Precedance::LOWEST)?;
        self.consume_peek(TokenKind::SEMICOLON)?;
        self.next_token();
        Ok(Statement::return_stmt(exp))
    }
    fn parse_expression_statement(&mut self) -> Result<Statement, ParseError> {
        let exp = self.parse_expression(Precedance::LOWEST);
        if self.peek_token.value == TokenKind::SEMICOLON {
            self.next_token();
        }
        self.next_token();
        exp.and_then(|ast| Ok(Statement::exp_stmt(ast)))
    }
    fn parse_expression(&mut self, predcedance: Precedance) -> Result<Ast, ParseError> {
        // println!("cur: {:?}, peek: {:?}", self.cur_token, self.peek_token);
        let mut left_exp = match &self.cur_token.value {
            TokenKind::IDENT(_) => self.parse_ident(),
            TokenKind::NUM(_) => self.parse_num(),
            TokenKind::TRUE | TokenKind::FALSE => self.parse_bool(),
            TokenKind::MINUS | TokenKind::BANG => self.parse_prefix_expression(),
            TokenKind::LPAREN => self.parse_ground_expression(),
            TokenKind::IF => self.parse_if_expression(),
            TokenKind::FUNCTION => self.parse_fn_expression(),
            _ => unreachable!(),
        }?;
        while self.peek_token.value != TokenKind::SEMICOLON && predcedance < self.peek_precedance()
        {
            self.next_token();
            let loc = left_exp.loc.clone();
            left_exp = if self.cur_token.value == TokenKind::LPAREN {
                self.parse_call_expression(left_exp)
            } else {
                self.parse_infix_expression(left_exp)
            }?;
            left_exp.loc = left_exp.loc.merge(&loc);
        }

        Ok(left_exp)
    }
    fn parse_ident(&mut self) -> Result<Ast, ParseError> {
        match &self.cur_token.value {
            TokenKind::IDENT(ident) => Ok(Ast::ident(ident, self.cur_token.loc.clone())),
            _ => Err(ParseError::UnexpectedToken(self.cur_token.clone())),
        }
    }
    fn parse_num(&mut self) -> Result<Ast, ParseError> {
        match &self.cur_token.value {
            TokenKind::NUM(n) => Ok(Ast::num(n.clone(), self.cur_token.loc.clone())),
            _ => Err(ParseError::UnexpectedToken(self.cur_token.clone())),
        }
    }
    fn parse_bool(&mut self) -> Result<Ast, ParseError> {
        match &self.cur_token.value {
            TokenKind::TRUE => Ok(Ast::boolean(true, self.cur_token.loc.clone())),
            TokenKind::FALSE => Ok(Ast::boolean(true, self.cur_token.loc.clone())),
            _ => Err(ParseError::UnexpectedToken(self.cur_token.clone())),
        }
    }
    fn parse_prefix_expression(&mut self) -> Result<Ast, ParseError> {
        let uniop = match &self.cur_token.value {
            TokenKind::BANG => UniOp::bang(self.cur_token.loc.clone()),
            TokenKind::MINUS => UniOp::minus(self.cur_token.loc.clone()),
            _ => unreachable!(),
        };
        self.next_token();
        let exp = self.parse_expression(Precedance::PREFIX)?;
        let loc = uniop.loc.clone().merge(&exp.loc);

        Ok(Ast::uniop(uniop, exp, loc))
    }
    fn parse_infix_expression(&mut self, left_exp: Ast) -> Result<Ast, ParseError> {
        let binop = match &self.cur_token.value {
            TokenKind::EQ => BinOp::eq(self.cur_token.loc.clone()),
            TokenKind::NOTEQ => BinOp::not_eq(self.cur_token.loc.clone()),
            TokenKind::LT => BinOp::lt(self.cur_token.loc.clone()),
            TokenKind::GT => BinOp::gt(self.cur_token.loc.clone()),
            TokenKind::PLUS => BinOp::add(self.cur_token.loc.clone()),
            TokenKind::MINUS => BinOp::sub(self.cur_token.loc.clone()),
            TokenKind::SLASH => BinOp::div(self.cur_token.loc.clone()),
            TokenKind::ASTERISK => BinOp::mult(self.cur_token.loc.clone()),
            _ => unreachable!(),
        };
        let cur_precedance = self.cur_precedance();
        self.next_token();
        let right_exp = self.parse_expression(cur_precedance)?;
        let loc = binop.loc.clone().merge(&right_exp.loc);

        Ok(Ast::binop(binop, left_exp, right_exp, loc))
    }
    fn parse_call_expression(&mut self, left_exp: Ast) -> Result<Ast, ParseError> {
        let mut ret = vec![];
        let mut loc = self.cur_token.loc.clone();
        self.consume_current(TokenKind::LPAREN)?;
        while self.cur_token.value != TokenKind::RPAREN {
            let exp = self.parse_expression(Precedance::LOWEST)?;
            loc = loc.merge(&exp.loc);
            ret.push(exp);
            self.next_token();
            match self.cur_token.value {
                TokenKind::COMMA => self.next_token(),
                TokenKind::RPAREN => (),
                _ => return Err(ParseError::UnexpectedToken(self.cur_token.clone())),
            }
        }
        let total_loc = left_exp.loc.clone().merge(&loc);
        Ok(Ast::call(left_exp, Ast::argument(ret, loc), total_loc))
    }
    fn parse_ground_expression(&mut self) -> Result<Ast, ParseError> {
        self.next_token();
        let exp = self.parse_expression(Precedance::LOWEST)?;
        if self.peek_token.value != TokenKind::RPAREN {
            Err(ParseError::UnclosedOpenParen(self.peek_token.clone()))
        } else {
            self.next_token();
            Ok(exp)
        }
    }
    fn parse_if_expression(&mut self) -> Result<Ast, ParseError> {
        let if_loc = self.cur_token.loc.clone();
        self.next_token();
        let exp = self.parse_expression(Precedance::LOWEST)?;
        self.consume_current(TokenKind::RPAREN)?;
        self.consume_current(TokenKind::LBRACE)?;
        let stmt1 = self.parse_blocked_statements()?;
        self.expect_current(TokenKind::RBRACE)?;

        match self.peek_token.value {
            TokenKind::ELSE => {
                self.next_token();
                self.consume_current(TokenKind::ELSE)?;
                self.consume_current(TokenKind::LBRACE)?;
                let stmt2 = self.parse_blocked_statements()?;
                self.expect_current(TokenKind::RBRACE)?;
                let loc = if_loc.merge(&exp.loc).merge(&self.cur_token.loc);
                Ok(Ast::if_ast(exp, stmt1, stmt2, loc))
            }
            _ => {
                let loc = if_loc.merge(&exp.loc).merge(&self.cur_token.loc);
                Ok(Ast::if_ast(exp, stmt1, vec![], loc))
            }
        }
    }
    fn parse_fn_expression(&mut self) -> Result<Ast, ParseError> {
        let fn_loc = self.cur_token.loc.clone();
        self.consume_current(TokenKind::FUNCTION)?;
        self.consume_current(TokenKind::LPAREN)?;
        let parameters = self.parse_parameters()?;
        self.consume_current(TokenKind::RPAREN)?;
        self.consume_current(TokenKind::LBRACE)?;
        let stmt1 = self.parse_blocked_statements()?;
        self.expect_current(TokenKind::RBRACE)?;
        Ok(Ast::function(
            &parameters,
            stmt1,
            fn_loc.merge(&self.cur_token.loc),
        ))
    }
    fn parse_parameters(&mut self) -> Result<Vec<Vec<u8>>, ParseError> {
        let mut ret = vec![];
        while self.cur_token.value != TokenKind::RPAREN {
            let ident = match &self.cur_token.value {
                TokenKind::IDENT(ident) => Ok(ident.clone()),
                _ => Err(ParseError::UnexpectedToken(self.cur_token.clone())),
            }?;
            ret.push(ident);
            self.next_token();
            match self.cur_token.value {
                TokenKind::COMMA => {
                    self.next_token();
                }
                TokenKind::RPAREN => {
                    ();
                }
                _ => {
                    return Err(ParseError::UnexpectedToken(self.cur_token.clone()));
                }
            }
        }
        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lexer1() {
        let input = "1 + 2 * (10 / 3 - 3) - 5";
        let expected = vec![
            Token::number(1, Loc(0, 1)),
            Token::plus(Loc(2, 3)),
            Token::number(2, Loc(4, 5)),
            Token::asterisk(Loc(6, 7)),
            Token::lparen(Loc(8, 9)),
            Token::number(10, Loc(9, 11)),
            Token::slash(Loc(12, 13)),
            Token::number(3, Loc(14, 15)),
            Token::minus(Loc(16, 17)),
            Token::number(3, Loc(18, 19)),
            Token::rparen(Loc(19, 20)),
            Token::minus(Loc(21, 22)),
            Token::number(5, Loc(23, 24)),
            Token::eof(Loc(24, 24)),
        ];
        let mut lexer = Lexer::new(input);
        for i in 0..expected.len() {
            let tok = lexer.next_token().unwrap();
            assert_eq!(tok, expected[i]);
        }
    }
    #[test]
    fn test_lexer2() {
        use TokenKind::*;
        let input = "=+(){},;";
        let expected = vec![
            ASSIGN, PLUS, LPAREN, RPAREN, LBRACE, RBRACE, COMMA, SEMICOLON, EOF,
        ];

        let mut lexer = Lexer::new(input);
        for i in 0..expected.len() {
            let tok = lexer.next_token().unwrap().value;
            assert_eq!(tok, expected[i]);
        }
    }
    #[test]
    fn test_lexer3() {
        use TokenKind::*;
        let input = "let five = 5;
let ten = 10;
!-/*5;
5 < 10 > 5;

if (5 < 10) {
    return true;
} else {
    return false;
}

10 == 10;
10 != 9;

let add = fn(x, y) {
    x + y;
};

let result = add(five, ten);
";
        let expected = vec![
            LET,
            IDENT(vec![b'f', b'i', b'v', b'e']),
            ASSIGN,
            NUM(5),
            SEMICOLON,
            LET,
            IDENT(vec![b't', b'e', b'n']),
            ASSIGN,
            NUM(10),
            SEMICOLON,
            BANG,
            MINUS,
            SLASH,
            ASTERISK,
            NUM(5),
            SEMICOLON,
            NUM(5),
            LT,
            NUM(10),
            GT,
            NUM(5),
            SEMICOLON,
            IF,
            LPAREN,
            NUM(5),
            LT,
            NUM(10),
            RPAREN,
            LBRACE,
            RETURN,
            TRUE,
            SEMICOLON,
            RBRACE,
            ELSE,
            LBRACE,
            RETURN,
            FALSE,
            SEMICOLON,
            RBRACE,
            NUM(10),
            EQ,
            NUM(10),
            SEMICOLON,
            NUM(10),
            NOTEQ,
            NUM(9),
            SEMICOLON,
            LET,
            IDENT(vec![b'a', b'd', b'd']),
            ASSIGN,
            FUNCTION,
            LPAREN,
            IDENT(vec![b'x']),
            COMMA,
            IDENT(vec![b'y']),
            RPAREN,
            LBRACE,
            IDENT(vec![b'x']),
            PLUS,
            IDENT(vec![b'y']),
            SEMICOLON,
            RBRACE,
            SEMICOLON,
            LET,
            IDENT(vec![b'r', b'e', b's', b'u', b'l', b't']),
            ASSIGN,
            IDENT(vec![b'a', b'd', b'd']),
            LPAREN,
            IDENT(vec![b'f', b'i', b'v', b'e']),
            COMMA,
            IDENT(vec![b't', b'e', b'n']),
            RPAREN,
            SEMICOLON,
            EOF,
        ];

        let mut lexer = Lexer::new(input);
        for i in 0..expected.len() {
            let tok = lexer.next_token().unwrap().value;
            assert_eq!(tok, expected[i]);
        }
    }

    // parserテスト用の補助関数
    fn compare_ast_with_expected(code: &str, expected: Ast) {
        let mut parser = Parser::new(code);
        match parser.parse_statement().unwrap() {
            Statement::Expression(exp) => {
                assert_eq!(*exp, expected);
            }
            _ => unreachable!(),
        }
    }
    fn _compare_ast_without_loc_impl(lhs: &Ast, rhs: &Ast) -> bool {
        {
            match (&lhs.value, &rhs.value) {
                (AstKind::Num(n), AstKind::Num(m)) => n == m,
                (AstKind::Ident(n), AstKind::Ident(m)) => n == m,
                (AstKind::UniOp { op: lhsop, e: lhse }, AstKind::UniOp { op: rhsop, e: rhse }) => {
                    lhsop.value == rhsop.value && _compare_ast_without_loc_impl(&lhse, &rhse)
                }
                (
                    AstKind::BinOp {
                        op: op1,
                        l: lhs1,
                        r: rhs1,
                    },
                    AstKind::BinOp {
                        op: op2,
                        l: lhs2,
                        r: rhs2,
                    },
                ) => {
                    op1.value == op2.value
                        && _compare_ast_without_loc_impl(&lhs1, &lhs2)
                        && _compare_ast_without_loc_impl(&rhs1, &rhs2)
                }
                // TODO: If, Functionも追加する
                _ => false,
            }
        }
    }
    fn compare_ast_without_loc(code1: &str, code2: &str) {
        let mut parser1 = Parser::new(code1);
        let mut parser2 = Parser::new(code2);
        match (
            parser1.parse_statement().unwrap(),
            parser2.parse_statement().unwrap(),
        ) {
            (Statement::Expression(exp1), Statement::Expression(exp2)) => {
                assert!(_compare_ast_without_loc_impl(&exp1, &exp2));
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn test_parser1() {
        let input = "!hoge10 - 100;";
        let expected = Ast::binop(
            BinOp::sub(
                // -
                Loc(8, 9),
            ),
            // !hoge10
            Ast::uniop(
                // !
                UniOp::bang(Loc(0, 1)),
                // hoge10
                Ast::ident(&vec![b'h', b'o', b'g', b'e', b'1', b'0'], Loc(1, 7)),
                Loc(0, 7),
            ),
            // 100
            Ast::num(100, Loc(10, 13)),
            Loc(0, 13),
        );
        compare_ast_with_expected(input, expected);
    }
    #[test]
    fn test_parser2() {
        let input = "5+5";
        let expected = Ast::binop(
            BinOp::add(Loc(1, 2)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(2, 3)),
            Loc(0, 3),
        );
        compare_ast_with_expected(input, expected);
        let input = "5-5";
        let expected = Ast::binop(
            BinOp::sub(Loc(1, 2)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(2, 3)),
            Loc(0, 3),
        );
        compare_ast_with_expected(input, expected);
        let input = "5*5";
        let expected = Ast::binop(
            BinOp::mult(Loc(1, 2)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(2, 3)),
            Loc(0, 3),
        );
        compare_ast_with_expected(input, expected);
        let input = "5/5";
        let expected = Ast::binop(
            BinOp::div(Loc(1, 2)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(2, 3)),
            Loc(0, 3),
        );
        compare_ast_with_expected(input, expected);
        let input = "5>5";
        let expected = Ast::binop(
            BinOp::gt(Loc(1, 2)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(2, 3)),
            Loc(0, 3),
        );
        compare_ast_with_expected(input, expected);
        let input = "5<5";
        let expected = Ast::binop(
            BinOp::lt(Loc(1, 2)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(2, 3)),
            Loc(0, 3),
        );
        compare_ast_with_expected(input, expected);
        let input = "5==5";
        let expected = Ast::binop(
            BinOp::eq(Loc(1, 3)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(3, 4)),
            Loc(0, 4),
        );
        compare_ast_with_expected(input, expected);
        let input = "5!=5";
        let expected = Ast::binop(
            BinOp::not_eq(Loc(1, 3)),
            Ast::num(5, Loc(0, 1)),
            Ast::num(5, Loc(3, 4)),
            Loc(0, 4),
        );
        compare_ast_with_expected(input, expected);
    }
    #[test]
    fn test_parser3() {
        let input = "5!=1+2*3";
        let right = Ast::binop(
            BinOp::add(Loc(4, 5)),
            Ast::num(1, Loc(3, 4)),
            Ast::binop(
                BinOp::mult(Loc(6, 7)),
                Ast::num(2, Loc(5, 6)),
                Ast::num(3, Loc(7, 8)),
                Loc(5, 8),
            ),
            Loc(3, 8),
        );
        let expected = Ast::binop(
            BinOp::not_eq(Loc(1, 3)),
            Ast::num(5, Loc(0, 1)),
            right,
            Loc(0, 8),
        );
        compare_ast_with_expected(input, expected);
    }
    #[test]
    fn test_parser4() {
        let input1 = "-a*(b)";
        let input2 = "-a*b";
        compare_ast_without_loc(input1, input2);

        let input1 = "5 * 2 + 3 * 4";
        let input2 = "(5*2)+(3*4)";
        compare_ast_without_loc(input1, input2);

        let input1 = "5 * (2 + 3 * 4)";
        let input2 = "((((5*(2+(3*4))))))";
        compare_ast_without_loc(input1, input2);
    }
    #[test]
    fn test_parser5() {
        let input = "if (x<y) {x}";
        let expected = Ast::if_ast(
            Ast::binop(
                BinOp::lt(Loc(5, 6)),
                Ast::ident(&vec![b'x'], Loc(4, 5)),
                Ast::ident(&vec![b'y'], Loc(6, 7)),
                Loc(4, 7),
            ),
            vec![Statement::exp_stmt(Ast::ident(&vec![b'x'], Loc(10, 11)))],
            vec![],
            Loc(0, 12),
        );
        compare_ast_with_expected(input, expected);

        let input = "if (x<y) {x} else {y} + 20";
        let expected = Ast::if_ast(
            Ast::binop(
                BinOp::lt(Loc(5, 6)),
                Ast::ident(&vec![b'x'], Loc(4, 5)),
                Ast::ident(&vec![b'y'], Loc(6, 7)),
                Loc(4, 7),
            ),
            vec![Statement::exp_stmt(Ast::ident(&vec![b'x'], Loc(10, 11)))],
            vec![Statement::exp_stmt(Ast::ident(&vec![b'y'], Loc(19, 20)))],
            Loc(0, 21),
        );
        let expected = Ast::binop(
            BinOp::add(Loc(22, 23)),
            expected,
            Ast::num(20, Loc(24, 26)),
            Loc(0, 26),
        );
        compare_ast_with_expected(input, expected);
    }
    #[test]
    fn test_parser6() {
        let input = "fn (x, y, z) { x * x + y * y }";
        let expected = Ast::function(
            &vec![vec![b'x'], vec![b'y'], vec![b'z']],
            vec![Statement::Expression(Box::new(Ast::binop(
                BinOp::add(Loc(21, 22)),
                Ast::binop(
                    BinOp::mult(Loc(17, 18)),
                    Ast::ident(&vec![b'x'], Loc(15, 16)),
                    Ast::ident(&vec![b'x'], Loc(19, 20)),
                    Loc(15, 20),
                ),
                Ast::binop(
                    BinOp::mult(Loc(25, 26)),
                    Ast::ident(&vec![b'y'], Loc(23, 24)),
                    Ast::ident(&vec![b'y'], Loc(27, 28)),
                    Loc(23, 28),
                ),
                Loc(15, 28),
            )))],
            Loc(0, 30),
        );
        compare_ast_with_expected(input, expected);
    }
}
