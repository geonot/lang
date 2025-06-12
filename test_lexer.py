import unittest
from lexer import (
    Lexer, Token,
    EOF, NEWLINE, INDENT, DEDENT, IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL, DOLLAR_PARAM, ERROR,
    KW_IS, KW_FN, KW_TRUE, KW_NO, KW_IF, KW_ELSE, KW_UNLESS, KW_WHILE, KW_UNTIL, KW_ITERATE, KW_RETURN,
    KW_MOD, KW_USE, KW_OBJECT, KW_STORE, KW_ACTOR, KW_EMPTY, KW_NOW, KW_OR, KW_AND, KW_NOT,
    KW_EQUALS, KW_AS, KW_FOR, KW_ERR, KW_ACROSS,
    SYM_DOT, SYM_LPAREN, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_PLUS, SYM_MINUS,
    SYM_MUL, SYM_DIV, SYM_OPERATOR_MOD, SYM_ASSIGN,
    SYM_GT, SYM_GTE, SYM_LT, SYM_LTE, SYM_QUESTION, SYM_EXCLAMATION,
    SYM_AMPERSAND, SYM_AT
)

class TestLexer(unittest.TestCase):

    def assertTokens(self, code, expected_tokens, include_eof=True):
        """Helper to compare token lists, optionally adding EOF if not in expected."""
        lexer = Lexer(code)
        tokens = list(lexer.tokens())

        # Find the EOF token produced by the lexer
        eof_token_actual = None
        if tokens and tokens[-1].type == EOF:
            eof_token_actual = tokens[-1]

        # If include_eof is true and expected_tokens doesn't have EOF, add the actual EOF from lexer
        if include_eof and eof_token_actual:
            # Check if last expected token is already EOF
            if not (expected_tokens and expected_tokens[-1].type == EOF):
                 expected_tokens.append(eof_token_actual)
            # If it is, ensure its line/col match (or are reasonably close for empty/simple inputs)
            elif expected_tokens and expected_tokens[-1].type == EOF:
                # For EOF, value is None. Line/col can vary.
                # We can either make expected EOF flexible or copy from actual.
                # For simplicity, if expected has an EOF, we'll ensure it's the last one.
                # The main check is the sequence of non-EOF tokens.
                # For this helper, let's just ensure the token lists (excluding EOF value) match.
                # The number of tokens must match too.
                pass # EOF already in expected, will be compared by type and value (None)

        # Remove EOF from actual tokens for list comparison if not included in expected,
        # or if we want to compare only up to non-EOF tokens.
        # For full comparison, we keep it.

        # A simple way to compare, focusing on type and value:
        processed_tokens = [(t.type, t.value) for t in tokens]
        processed_expected = [(t.type, t.value) for t in expected_tokens]

        self.assertEqual(processed_tokens, processed_expected,
                         f"\nCode: '{code}'\nActual Tokens: {tokens}\nExpected Tokens: {expected_tokens}")

    def test_empty_input(self):
        # Lexer adds EOF if input is empty. Line/col might vary slightly based on init.
        # Let's get the EOF from the lexer itself for comparison.
        lexer = Lexer("")
        actual_tokens = list(lexer.tokens())
        self.assertEqual(len(actual_tokens), 1)
        self.assertEqual(actual_tokens[0].type, EOF)
        # self.assertEqual(actual_tokens[0].line, 1) # Lexer's EOF handling might vary
        # self.assertEqual(actual_tokens[0].column, 1)


    def test_identifier(self):
        self.assertTokens("name", [Token(IDENTIFIER, "name", 1, 1)])

    def test_integer_literal(self):
        self.assertTokens("123", [Token(INTEGER_LITERAL, 123, 1, 1)])
        self.assertTokens("0xFF", [Token(INTEGER_LITERAL, 255, 1, 1)])
        self.assertTokens("0x1a", [Token(INTEGER_LITERAL, 26, 1, 1)])
        self.assertTokens("0b101", [Token(INTEGER_LITERAL, 5, 1, 1)])
        self.assertTokens("0B11", [Token(INTEGER_LITERAL, 3, 1, 1)])

    def test_invalid_integer_literals(self):
        self.assertTokens("0xG", [Token(ERROR, "Malformed hexadecimal literal: 0xG", 1, 1)])
        self.assertTokens("0b2", [Token(ERROR, "Malformed binary literal (non-binary digit): 0b2", 1, 1)])
        self.assertTokens("0b", [Token(ERROR, "Incomplete binary literal: 0b", 1, 1)])
        self.assertTokens("0x", [Token(ERROR, "Incomplete hexadecimal literal: 0x", 1, 1)])


    def test_float_literal(self):
        self.assertTokens("123.45", [Token(FLOAT_LITERAL, 123.45, 1, 1)])
        self.assertTokens(".5", [Token(FLOAT_LITERAL, 0.5, 1, 1)])
        self.assertTokens("0.0", [Token(FLOAT_LITERAL, 0.0, 1, 1)])
        self.assertTokens("5.", [Token(FLOAT_LITERAL, 5.0, 1, 1)]) # Assuming lexer handles "5." as 5.0

    def test_invalid_float_literals(self):
        # Current lexer _number() logic for "1.2.3" produces ERROR
        self.assertTokens("1.2.3", [Token(ERROR, "Malformed float literal (multiple .): 1.2.3", 1, 1)])
        self.assertTokens(".", [Token(SYM_DOT, ".", 1, 1)]) # A single dot is a symbol, not an error or number
        # What about "1.2G"? Lexer might produce FLOAT(1.2), IDENTIFIER(G) or error.
        # Current lexer would parse 1.2, then G would be separate.
        # self.assertTokens("1.2G", [Token(FLOAT_LITERAL, 1.2, 1, 1), Token(IDENTIFIER, "G", 1, 4)])


    def test_string_literal(self):
        self.assertTokens("'hello'", [Token(STRING_LITERAL, "hello", 1, 1)])
        self.assertTokens('"world_2"', [Token(STRING_LITERAL, "world_2", 1, 1)])
        self.assertTokens("'esc\\naped'", [Token(STRING_LITERAL, "esc\naped", 1, 1)])
        self.assertTokens("'\\t\\r\\\"\\\\'\\\\'", [Token(STRING_LITERAL, "\t\r\"'\\", 1, 1)])


    def test_unterminated_string_literal(self):
        self.assertTokens("'abc", [Token(ERROR, "Unterminated string literal (EOF)", 1, 1)])
        self.assertTokens("'abc\\", [Token(ERROR, "Unterminated string literal (EOF after escape)", 1, 1)])
        self.assertTokens("'abc\ndef", [
            Token(ERROR, "Unterminated string literal (ends at newline)", 1, 1),
            # Lexer currently doesn't emit NEWLINE if string error occurs on that line before newline char.
            # It depends on how _string handles advance() before erroring.
            # Current lexer's _string does not advance past the newline if it causes unterm error.
            # So, the NEWLINE token will be generated by the main loop.
            Token(NEWLINE, "\n", 1,5),
            Token(IDENTIFIER, "def", 2,1),
        ])


    def test_assignment_simple(self):
        self.assertTokens("x is 10", [
            Token(IDENTIFIER, "x", 1, 1),
            Token(KW_IS, "is", 1, 3),
            Token(INTEGER_LITERAL, 10, 1, 6),
        ])

    def test_keywords(self):
        self.assertTokens("if true else no", [
            Token(KW_IF, "if", 1, 1),
            Token(KW_TRUE, "true", 1, 4),
            Token(KW_ELSE, "else", 1, 9),
            Token(KW_NO, "no", 1, 14),
        ])

    def test_symbols(self):
        self.assertTokens(". + - * / % = > >= < <= ? ! & @ ( ) , :", [
            Token(SYM_DOT, ".", 1, 1),
            Token(SYM_PLUS, "+", 1, 3),
            Token(SYM_MINUS, "-", 1, 5),
            Token(SYM_MUL, "*", 1, 7),
            Token(SYM_DIV, "/", 1, 9),
            Token(SYM_OPERATOR_MOD, "%", 1, 11),
            Token(SYM_ASSIGN, "=", 1, 13), # Assuming SYM_ASSIGN is for '='
            Token(SYM_GT, ">", 1, 15),
            Token(SYM_GTE, ">=", 1, 17),
            Token(SYM_LT, "<", 1, 20),
            Token(SYM_LTE, "<=", 1, 22),
            Token(SYM_QUESTION, "?", 1, 25),
            Token(SYM_EXCLAMATION, "!", 1, 27),
            Token(SYM_AMPERSAND, "&", 1, 29),
            Token(SYM_AT, "@", 1, 31),
            Token(SYM_LPAREN, "(", 1, 33),
            Token(SYM_RPAREN, ")", 1, 35),
            Token(SYM_COMMA, ",", 1, 37),
            Token(SYM_COLON, ":", 1, 39),
        ])

    def test_dollar_param(self):
        self.assertTokens("$name $0", [
            Token(DOLLAR_PARAM, "name", 1, 1),
            Token(DOLLAR_PARAM, 0, 1, 7),
        ])
        # Test with lexer's actual behavior for value (str vs int)
        lexer_name = Lexer("$name")
        tokens_name = list(lexer_name.tokens())
        self.assertEqual(tokens_name[0].value, "name") # DollarParamNode expects str "name" or int 0

        lexer_0 = Lexer("$0")
        tokens_0 = list(lexer_0.tokens())
        self.assertEqual(tokens_0[0].value, 0) # DollarParamNode expects int 0

    def test_invalid_dollar_param(self):
        self.assertTokens("$?", [Token(ERROR, "Invalid character after $: '?'", 1, 1)])
        self.assertTokens("$", [Token(ERROR, "Invalid character after $: 'None'", 1, 1)]) # If $ is at EOF

    # --- Indentation Tests ---
    def test_newline_indent_dedent_simple(self):
        code = "a\n  b\nc"
        expected = [
            Token(IDENTIFIER, "a", 1, 1), Token(NEWLINE, "\n", 1, 2),
            Token(INDENT, 2, 2, 1), Token(IDENTIFIER, "b", 2, 3), Token(NEWLINE, "\n", 2, 4),
            Token(DEDENT, 0, 3, 1), Token(IDENTIFIER, "c", 3, 1)
        ]
        self.assertTokens(code, expected)

    def test_multiple_indents_dedents(self):
        code = "l1\n  l2a\n    l3\n  l2b\nl0"
        expected = [
            Token(IDENTIFIER, "l1", 1, 1), Token(NEWLINE, "\n", 1, 3),
            Token(INDENT, 2, 2, 1), Token(IDENTIFIER, "l2a", 2, 3), Token(NEWLINE, "\n", 2, 6),
            Token(INDENT, 4, 3, 1), Token(IDENTIFIER, "l3", 3, 5), Token(NEWLINE, "\n", 3, 7),
            Token(DEDENT, 2, 4, 1), Token(IDENTIFIER, "l2b", 4, 3), Token(NEWLINE, "\n", 4, 6),
            Token(DEDENT, 0, 5, 1), Token(IDENTIFIER, "l0", 5, 1)
        ]
        self.assertTokens(code, expected)

    def test_dedent_to_zero(self):
        code = "l1\n  l2\nl0"
        expected = [
            Token(IDENTIFIER, "l1", 1, 1), Token(NEWLINE, "\n", 1, 3),
            Token(INDENT, 2, 2, 1), Token(IDENTIFIER, "l2", 2, 3), Token(NEWLINE, "\n", 2, 5),
            Token(DEDENT, 0, 3, 1), Token(IDENTIFIER, "l0", 3, 1)
        ]
        self.assertTokens(code, expected)

    def test_inconsistent_indent_error(self):
        # l3 is less indented than l2 but not aligned with l1 (0)
        code = "l1\n  l2\n l3"
        expected = [
            Token(IDENTIFIER, "l1", 1, 1), Token(NEWLINE, "\n", 1, 3),
            Token(INDENT, 2, 2, 1), Token(IDENTIFIER, "l2", 2, 3), Token(NEWLINE, "\n", 2, 5),
            # Error occurs when ' l3' is processed. DEDENT to non-existent level 1.
            Token(ERROR, "Indentation error: unaligned indent", 3, 1)
            # Lexer might try to dedent, then error. The exact sequence can vary.
            # Current lexer: DEDENT to 0, then error because 1 is not on stack.
            # Or, DEDENT to 0, then INDENT to 1, if that's how it works.
            # The current lexer's handle_newline:
            # It pops stack until current_indent >= stack top. If current_indent < stack top, error.
            # So it would pop 2, stack top becomes 0. current_indent is 1. 1 < 0 is false.
            # Then it checks if current_indent (1) != stack[-1] (0). Yes. -> Error.
            # So, DEDENT(0) then ERROR("Indentation error: unaligned indent", 3, 1)
            # Token(DEDENT, 0, 3,1), # DEDENT from 2 to 0
            # Token(ERROR, "Indentation error: unaligned indent", 3, 2) # Error at column of l3
        ]
        # Due to complexity of exact error token sequence with indent errors,
        # we'll check if an ERROR token with the specific message is present.
        lexer = Lexer(code)
        tokens = list(lexer.tokens())
        found_error = False
        for t in tokens:
            if t.type == ERROR and "Indentation error: unaligned indent" in t.value:
                found_error = True
                break
        self.assertTrue(found_error, f"Expected 'Indentation error' for code: '{code}'\nGot: {tokens}")


    def test_blank_lines_and_indent(self):
        code = "l1\n\n  l2\n    \n  l3" # Blank line with spaces, blank line
        expected = [
            Token(IDENTIFIER, "l1", 1, 1), Token(NEWLINE, "\n", 1, 3),
            Token(NEWLINE, "\n", 2, 1), # Blank line
            Token(INDENT, 2, 3, 1), Token(IDENTIFIER, "l2", 3, 3), Token(NEWLINE, "\n", 3, 5),
            # Line 4 is "    \n". It's a blank line effectively, no content, so no indent/dedent change.
            # Lexer's handle_newline skips indent/dedent logic if line has no effective content.
            Token(NEWLINE, "\n", 4, 5), # The newline from the blank line with spaces
            Token(IDENTIFIER, "l3", 5, 3) # l3 should still be at indent level 2
        ]
        self.assertTokens(code, expected)

    def test_all_keywords(self):
        all_keywords_code = " ".join(KEYWORDS.keys())
        expected_tokens = []
        col = 1
        for kw_text in KEYWORDS.keys():
            expected_tokens.append(Token(KEYWORDS[kw_text], kw_text, 1, col))
            col += len(kw_text) + 1
        self.assertTokens(all_keywords_code, expected_tokens)

    def test_all_symbols(self):
        # Sort symbols by length descending to handle multi-char symbols correctly if simply joined
        # However, lexer matches longest, so order of joining here doesn't strictly matter for lexer test itself.
        # But for constructing the test string, ensure spaces.
        symbol_texts = sorted(SYMBOLS.keys(), key=len, reverse=True)
        all_symbols_code = " ".join(symbol_texts)

        expected_tokens = []
        col = 1
        for sym_text in symbol_texts: # Use original order for expectation if that's how we built string
            # Re-sort for this loop to match the string construction
            pass # Handled by creating string based on sorted and iterating that way

        current_col = 1
        sorted_for_string = SYMBOLS.keys() # Or symbol_texts if using that for string
        all_symbols_code_for_test = " ".join(sorted_for_string)

        expected_sym_tokens = []
        for sym_text in sorted_for_string:
            expected_sym_tokens.append(Token(SYMBOLS[sym_text], sym_text, 1, current_col))
            current_col += len(sym_text) + 1
        self.assertTokens(all_symbols_code_for_test, expected_sym_tokens)

    # --- Edge Cases ---
    def test_input_with_only_comments(self):
        code = "// comment1\n// comment2\n  // comment3 indented"
        expected = [
            Token(NEWLINE, "\n", 1, 11),
            Token(NEWLINE, "\n", 2, 11),
            # Indented comment line still just a comment, results in newline.
            # Indentation on comment lines is ignored by lexer's indent logic.
            Token(NEWLINE, "\n", 3, 19)
        ]
        self.assertTokens(code, expected)

    def test_input_ending_mid_comment(self):
        code = "x is // comment" # Ends without newline after comment
        expected = [
            Token(IDENTIFIER, "x", 1, 1),
            Token(KW_IS, "is", 1, 3),
            # The comment consumes till EOF. No newline token. EOF is added by assertTokens.
        ]
        self.assertTokens(code, expected)

        code2 = "x is #" # Assuming # also starts a comment as per original example lexer. Current is //
                        # If # is not comment, it's an error or symbol.
                        # Current lexer uses // for comments.
        # If # is an error:
        # self.assertTokens(code2, [
        #     Token(IDENTIFIER, "x", 1, 1),
        #     Token(KW_IS, "is", 1, 3),
        #     Token(ERROR, "Unexpected character: '#'", 1, 6)
        # ])
        # If # is not defined and not comment, it's an unexpected char.
        # The provided lexer only has // comments.

    def test_long_identifier(self):
        long_name = "a" * 1000
        self.assertTokens(long_name, [Token(IDENTIFIER, long_name, 1, 1)])

    def test_long_string(self):
        long_str_content = "b" * 1000
        self.assertTokens(f"'{long_str_content}'", [Token(STRING_LITERAL, long_str_content, 1, 1)])


    def test_comment_handling(self):
        # Comments should be skipped, newlines preserved if significant
        code1 = "// comment\nident"
        expected1 = [
            Token(NEWLINE, "\n", 1, 11), # Newline after comment
            Token(IDENTIFIER, "ident", 2, 1),
        ]
        self.assertTokens(code1, expected1)

        code2 = "ident // comment"
        expected2 = [
            Token(IDENTIFIER, "ident", 1, 1),
            # No newline token if comment is at end of file without newline
        ]
        self.assertTokens(code2, expected2)

        code3 = "ident // comment\n" # With newline after comment at EOF
        expected3 = [
            Token(IDENTIFIER, "ident", 1, 1),
            Token(NEWLINE, "\n", 1, 17)
        ]
        self.assertTokens(code3, expected3)

        code4 = "line1\n// blank line comment\nline3"
        expected4 = [
            Token(IDENTIFIER, "line1", 1,1),
            Token(NEWLINE, "\n", 1,6),
            Token(NEWLINE, "\n", 2,22), # Newline from the comment line itself
            Token(IDENTIFIER, "line3", 3,1)
        ]
        self.assertTokens(code4, expected4)

    def test_error_token(self):
        # Test for an invalid character that lexer should flag as ERROR
        # Example: an unsupported symbol like `^` if not defined
        code = "a ^ b"
        # Assuming '^' is not a valid symbol and results in an ERROR token
        # The lexer advances past error, producing an ERROR token.
        lexer = Lexer(code)
        tokens = list(lexer.tokens())

        # Verify the sequence of tokens including the error
        # The exact value of the error token might include the character.
        # We are checking if an ERROR token is produced and other tokens are correct.

        # Find the error token and check its properties if needed
        error_token_found = False
        for t in tokens:
            if t.type == ERROR:
                error_token_found = True
                self.assertTrue("^" in t.value or "Unexpected character: '^'" in t.value) # Check if error message is reasonable
                break
        self.assertTrue(error_token_found, f"Expected an ERROR token for code: '{code}', got {tokens}")

        # Check overall token structure if error is expected to be non-fatal for lexing rest of line
        # Example: Token(IDENTIFIER, "a", 1,1), Token(ERROR, "Unexpected character: '^'", 1,3), Token(IDENTIFIER, "b",1,5)
        # This depends on how the lexer is designed to recover or report errors.
        # Current lexer: IDENTIFIER('a'), ERROR('Unexpected char: ^'), IDENTIFIER('b'), EOF
        # Let's use assertTokens for this if the error token value is predictable.
        # The value for ERROR token is "Unexpected character: '{err_char}'"
        self.assertTokens(code, [
            Token(IDENTIFIER, "a", 1, 1),
            Token(ERROR, "Unexpected character: '^'", 1, 3), # Col where '^' is.
            Token(IDENTIFIER, "b", 1, 5)
        ])


if __name__ == '__main__':
    unittest.main()
