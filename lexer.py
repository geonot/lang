import re

# Token types
EOF = 'EOF'
NEWLINE = 'NEWLINE'
INDENT = 'INDENT'
DEDENT = 'DEDENT'
IDENTIFIER = 'IDENTIFIER'
INTEGER_LITERAL = 'INTEGER_LITERAL'
FLOAT_LITERAL = 'FLOAT_LITERAL'
STRING_LITERAL = 'STRING_LITERAL'
DOLLAR_PARAM = 'DOLLAR_PARAM'
ERROR = 'ERROR' # New Error token

# Keywords - Using "KW_" prefix for clarity in token type
KW_IS = 'KW_IS'
KW_FN = 'KW_FN'
KW_TRUE = 'KW_TRUE'
KW_NO = 'KW_NO'
KW_IF = 'KW_IF'
KW_ELSE = 'KW_ELSE'
KW_UNLESS = 'KW_UNLESS'
KW_WHILE = 'KW_WHILE'
KW_UNTIL = 'KW_UNTIL'
KW_ITERATE = 'KW_ITERATE'
KW_RETURN = 'KW_RETURN'
KW_MOD = 'KW_MOD' # Keyword 'mod'
KW_USE = 'KW_USE'
KW_OBJECT = 'KW_OBJECT'
KW_STORE = 'KW_STORE'
KW_ACTOR = 'KW_ACTOR'
KW_EMPTY = 'KW_EMPTY'
KW_NOW = 'KW_NOW'
KW_OR = 'KW_OR'
KW_AND = 'KW_AND'
KW_NOT = 'KW_NOT'
KW_EQUALS = 'KW_EQUALS' # Keyword 'equals'
KW_AS = 'KW_AS'
KW_FOR = 'KW_FOR'
KW_ERR = 'KW_ERR'
KW_ACROSS = 'KW_ACROSS'

KEYWORDS = {
    'is': KW_IS, 'fn': KW_FN, 'true': KW_TRUE, 'no': KW_NO, 'if': KW_IF, 'else': KW_ELSE,
    'unless': KW_UNLESS, 'while': KW_WHILE, 'until': KW_UNTIL, 'iterate': KW_ITERATE,
    'return': KW_RETURN, 'mod': KW_MOD, 'use': KW_USE, 'object': KW_OBJECT, 'store': KW_STORE,
    'actor': KW_ACTOR, 'empty': KW_EMPTY, 'now': KW_NOW, 'or': KW_OR, 'and': KW_AND,
    'not': KW_NOT, 'equals': KW_EQUALS, 'as': KW_AS, 'for': KW_FOR, 'err': KW_ERR,
    'across': KW_ACROSS
}

# Symbols - Using "SYM_" prefix for clarity
SYM_DOT = 'SYM_DOT'
SYM_LPAREN = 'SYM_LPAREN'
SYM_RPAREN = 'SYM_RPAREN'
SYM_COMMA = 'SYM_COMMA'
SYM_COLON = 'SYM_COLON'
SYM_PLUS = 'SYM_PLUS'
SYM_MINUS = 'SYM_MINUS'
SYM_MUL = 'SYM_MUL'
SYM_DIV = 'SYM_DIV'
SYM_OPERATOR_MOD = 'SYM_OPERATOR_MOD' # Operator '%'
SYM_ASSIGN = 'SYM_ASSIGN' # Assignment '='
SYM_GT = 'SYM_GT'
SYM_GTE = 'SYM_GTE'
SYM_LT = 'SYM_LT'
SYM_LTE = 'SYM_LTE'
SYM_QUESTION = 'SYM_QUESTION'
SYM_EXCLAMATION = 'SYM_EXCLAMATION'
SYM_AMPERSAND = 'SYM_AMPERSAND'
SYM_AT = 'SYM_AT'


SYMBOLS = {
    '.': SYM_DOT, '(': SYM_LPAREN, ')': SYM_RPAREN, ',': SYM_COMMA, ':': SYM_COLON,
    '+': SYM_PLUS, '-': SYM_MINUS, '*': SYM_MUL, '/': SYM_DIV, '%': SYM_OPERATOR_MOD,
    '=': SYM_ASSIGN, '>': SYM_GT, '>=': SYM_GTE, '<': SYM_LT, '<=': SYM_LTE,
    '?': SYM_QUESTION, '!': SYM_EXCLAMATION, '&': SYM_AMPERSAND, '@': SYM_AT
    # Note: '=' is assign. 'equals' is the keyword for comparison.
}


class Token:
    def __init__(self, type, value, line, column):
        self.type = type
        self.value = value
        self.line = line
        self.column = column

    def __repr__(self):
        return f"Token({self.type!r}, {self.value!r}, line={self.line}, col={self.column})"

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.type == other.type and
                self.value == other.value and
                self.line == other.line and
                self.column == other.column)


class Lexer:
    def __init__(self, text):
        self.text = text
        self.pos = 0
        self.current_char = self.text[self.pos] if self.pos < len(self.text) else None
        self.line = 1
        self.column = 1
        self.indent_stack = [0]
        self.token_queue = []
        self.keywords = KEYWORDS
        self.symbols = SYMBOLS
        # Sort by length descending to match longer symbols first (e.g., ">=" before ">")
        self.sorted_symbol_keys = sorted(self.symbols.keys(), key=len, reverse=True)

    def advance(self):
        if self.current_char == '\n':
            self.line += 1
            self.column = 1
        elif self.current_char is not None: # Don't advance column for EOF
            self.column += 1

        self.pos += 1
        if self.pos < len(self.text):
            self.current_char = self.text[self.pos]
        else:
            self.current_char = None

    def peek(self, n=1):
        peek_pos = self.pos + n
        if peek_pos < len(self.text):
            return self.text[peek_pos]
        return None

    def skip_whitespace(self):
        while self.current_char is not None and self.current_char in ' \t\r':
            self.advance()

    def skip_comment(self):
        # Comments are // in Coral, not #
        if self.current_char == '/' and self.peek() == '/':
            while self.current_char is not None and self.current_char != '\n':
                self.advance()

    def _identifier(self):
        start_line = self.line
        start_column = self.column
        value = ''
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
            value += self.current_char
            self.advance()

        token_type = self.keywords.get(value, IDENTIFIER)
        return Token(token_type, value, start_line, start_column)

    def _number(self):
        start_line = self.line
        start_column = self.column
        num_str = ''
        is_float = False

        # Check for hex (0x) or binary (0b) prefixes
        if self.current_char == '0':
            num_str += self.current_char
            self.advance()
            if self.current_char in ('x', 'X'):
                num_str += self.current_char
                self.advance()
                while self.current_char is not None and self.current_char.isalnum(): # More specific hex check needed
                    if not self.current_char.lower() in '0123456789abcdef':
                        # Malformed hex
                        val_so_far = num_str + self.current_char
                        self.advance() # consume offending char
                        return Token(ERROR, f"Malformed hexadecimal literal: {val_so_far}", start_line, start_column)
                    num_str += self.current_char
                    self.advance()
                if len(num_str) == 2: # Just "0x"
                    return Token(ERROR, f"Incomplete hexadecimal literal: {num_str}", start_line, start_column)
                return Token(INTEGER_LITERAL, int(num_str, 16), start_line, start_column)
            elif self.current_char in ('b', 'B'):
                num_str += self.current_char
                self.advance()
                while self.current_char is not None and self.current_char in '01':
                    num_str += self.current_char
                    self.advance()
                if len(num_str) == 2: # Just "0b"
                     return Token(ERROR, f"Incomplete binary literal: {num_str}", start_line, start_column)
                # Check if there's any non-binary digit after "0b" part
                if self.current_char is not None and self.current_char.isdigit():
                    val_so_far = num_str + self.current_char
                    self.advance()
                    return Token(ERROR, f"Malformed binary literal (non-binary digit): {val_so_far}", start_line, start_column)
                return Token(INTEGER_LITERAL, int(num_str, 2), start_line, start_column)
            # Fall through for numbers starting with 0 that are not hex or bin (e.g. 0, 0.5)
            # No octal support explicitly mentioned, so 0 followed by other digits is decimal.

        # Regular decimal integers or floats
        while self.current_char is not None and self.current_char.isdigit():
            num_str += self.current_char
            self.advance()

        if self.current_char == '.':
            # Check if it's "1.method" or "1.2"
            if self.peek() is None or not self.peek().isdigit():
                # This is likely an integer followed by a dot operator.
                # Let the main loop handle the dot.
                if not num_str: # Case like ".5" - leading dot for float
                    num_str = '0' # Interpret as "0.5"
                else: # Integer part already consumed
                    return Token(INTEGER_LITERAL, int(num_str), start_line, start_column)

            # It's a float
            is_float = True
            num_str += self.current_char  # Add the dot
            self.advance()
            while self.current_char is not None and self.current_char.isdigit():
                num_str += self.current_char
                self.advance()
            # Check for multiple decimal points e.g. 1.2.3
            if self.current_char == '.':
                num_str += self.current_char
                self.advance() # consume the extra dot
                return Token(ERROR, f"Malformed float literal (multiple .): {num_str}", start_line, start_column)


        if is_float:
            if num_str == '.': # only a dot was found, not valid
                 return Token(ERROR, "Isolated '.' is not a valid number", start_line, start_column)
            return Token(FLOAT_LITERAL, float(num_str), start_line, start_column)
        else:
            if not num_str: # Should not happen if called correctly
                return Token(ERROR, "Empty number string", start_line, start_column)
            return Token(INTEGER_LITERAL, int(num_str), start_line, start_column)

    def _string(self, quote_char):
        start_line = self.line
        start_column = self.column
        self.advance()  # Consume opening quote
        value = ''
        while self.current_char is not None and self.current_char != quote_char:
            if self.current_char == '\n': # Unterminated string due to newline
                # Do not consume newline, let handle_newline deal with it.
                # Error token should point to where string started.
                return Token(ERROR, f"Unterminated string literal (ends at newline)", start_line, start_column)
            if self.current_char == '\\':
                self.advance() # Consume backslash
                if self.current_char is None: # Unterminated string due to EOF after backslash
                    return Token(ERROR, "Unterminated string literal (EOF after escape)", start_line, start_column)
                if self.current_char == quote_char: value += quote_char
                elif self.current_char == 'n': value += '\n'
                elif self.current_char == 't': value += '\t'
                elif self.current_char == '\\': value += '\\'
                else: # Unknown escape sequence
                    value += '\\' + self.current_char # Keep it as is or error? Coral spec needed. For now, keep.
            else:
                value += self.current_char
            self.advance()

        if self.current_char == quote_char:
            self.advance()  # Consume closing quote
            return Token(STRING_LITERAL, value, start_line, start_column)
        else: # Unterminated string (EOF)
            return Token(ERROR, "Unterminated string literal (EOF)", start_line, start_column)

    def _dollar_param(self):
        start_line = self.line
        start_column = self.column
        self.advance()  # Consume '$'
        value = '$'

        # Check for $0, $1 etc.
        if self.current_char is not None and self.current_char.isdigit():
            while self.current_char is not None and self.current_char.isdigit():
                value += self.current_char
                self.advance()
            return Token(DOLLAR_PARAM, value, start_line, start_column)
        # Check for $name, $variable_name etc.
        elif self.current_char is not None and (self.current_char.isalpha() or self.current_char == '_'):
            while self.current_char is not None and (self.current_char.isalnum() or self.current_char == '_'):
                value += self.current_char
                self.advance()
            return Token(DOLLAR_PARAM, value, start_line, start_column)
        else:
            # $ followed by something invalid (e.g. space, symbol, EOF)
            return Token(ERROR, f"Invalid character after $: '{self.current_char}'", start_line, start_column)


    def handle_newline(self):
        start_line = self.line
        start_col = self.column
        self.token_queue.append(Token(NEWLINE, '\n', start_line, start_col))
        self.advance()  # Consume '\n'

        current_indent = 0
        # Calculate indent on the new line
        while self.current_char is not None and self.current_char == ' ': # Only spaces for indent
            current_indent += 1
            self.advance()

        # Check if the line is effectively blank (only spaces then newline, EOF, or comment)
        is_effective_content = True
        if self.current_char is None or self.current_char == '\n':
            is_effective_content = False
        elif self.current_char == '/' and self.peek() == '/': # Start of a comment
            is_effective_content = False

        if is_effective_content:
            if current_indent > self.indent_stack[-1]:
                self.indent_stack.append(current_indent)
                self.token_queue.append(Token(INDENT, current_indent, self.line, 1))
            elif current_indent < self.indent_stack[-1]:
                while current_indent < self.indent_stack[-1]:
                    self.indent_stack.pop()
                    self.token_queue.append(Token(DEDENT, self.indent_stack[-1], self.line, 1))
                if current_indent != self.indent_stack[-1]:
                    self.token_queue.append(Token(ERROR, "Indentation error: unaligned indent", self.line, 1))
        # If not is_effective_content, this line doesn't affect indentation levels.

    def tokens(self):
        while True:
            if self.token_queue:
                yield self.token_queue.pop(0)
                continue

            if self.current_char is None: # EOF
                # Emit DEDENTs for any remaining open indents
                current_pos_line = self.line # Use current line, might be last content line or next
                current_pos_col = self.column if self.column > 1 else 1 # Use current column or 1
                while self.indent_stack[-1] > 0:
                    self.indent_stack.pop()
                    # DEDENT value is the level returned to.
                    yield Token(DEDENT, self.indent_stack[-1] if self.indent_stack else 0, current_pos_line, 1)
                yield Token(EOF, None, current_pos_line, current_pos_col)
                break

            start_line = self.line
            start_column = self.column

            if self.current_char in ' \t\r': # \r for windows line endings if not handled by text input
                self.skip_whitespace()
                continue

            if self.current_char == '/' and self.peek() == '/':
                self.skip_comment()
                continue

            # After skip_whitespace/comment, current_char could be None if file ends with them
            if self.current_char is None:
                continue # Let the EOF logic at the top of the loop handle this

            if self.current_char == '\n':
                self.handle_newline() # Adds to token_queue, advances
                continue

            if self.current_char.isalpha() or self.current_char == '_':
                yield self._identifier() # Advances
                continue

            # Need to be careful with '.' for floats vs method calls
            if self.current_char.isdigit() or \
               (self.current_char == '.' and self.peek() is not None and self.peek().isdigit()):
                yield self._number() # Advances
                continue

            if self.current_char == '$':
                yield self._dollar_param() # Advances
                continue

            if self.current_char in ('"', "'"):
                yield self._string(self.current_char) # Advances
                continue

            # Match symbols
            symbol_matched = False
            for s_key in self.sorted_symbol_keys:
                if self.text.startswith(s_key, self.pos):
                    token_val = s_key
                    token_type = self.symbols[s_key]
                    # Capture position before advancing for the symbol
                    sym_start_line, sym_start_col = self.line, self.column
                    for _ in range(len(s_key)):
                        self.advance()
                    yield Token(token_type, token_val, sym_start_line, sym_start_col)
                    symbol_matched = True
                    break

            if symbol_matched:
                continue

            # If no token matched
            err_char = self.current_char
            err_line = self.line
            err_col = self.column
            self.advance() # Consume the unexpected character
            yield Token(ERROR, f"Unexpected character: '{err_char}'", err_line, err_col)
```
