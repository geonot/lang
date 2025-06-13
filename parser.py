from lexer import (
    Token, EOF, NEWLINE, INDENT, DEDENT, IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL, DOLLAR_PARAM,
    KW_IS, KW_FN, KW_TRUE, KW_NO, KW_IF, KW_ELSE, KW_UNLESS, KW_WHILE, KW_UNTIL, KW_ITERATE, KW_RETURN,
    KW_MOD, # This is the module keyword from EBNF `mod`
    KW_USE, KW_OBJECT, KW_STORE, KW_ACTOR, KW_EMPTY, KW_NOW, KW_OR, KW_AND, KW_NOT,
    KW_EQUALS, KW_AS, KW_FOR, KW_ERR, KW_ACROSS,
    SYM_DOT, SYM_LPAREN, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_PLUS, SYM_MINUS,
    SYM_MUL, SYM_DIV, SYM_OPERATOR_MOD, # Corrected symbol names
    SYM_ASSIGN, # Corrected: '=' for assignment (used with 'is' keyword typically, or direct for map blocks etc)
    SYM_GT, SYM_GTE, SYM_LT, SYM_LTE, SYM_QUESTION, SYM_EXCLAMATION,
    SYM_AMPERSAND, SYM_AT, ERROR
)

# Import all AST node classes
from ast_nodes import *

class ParseError(Exception):
    def __init__(self, message, token):
        super().__init__(message)
        self.token = token
        self.line = token.line if token else -1
        self.column = token.column if token else -1

    def __str__(self):
        if self.token and self.token.type != EOF:
            return f"ParseError at Line {self.line}, Column {self.column} (Token: {self.token.type} '{self.token.value}'): {super().__str__()}"
        elif self.token and self.token.type == EOF:
            return f"ParseError at end of file: {super().__str__()}"
        else:
            return f"ParseError: {super().__str__()}"


class Parser:
    def __init__(self, tokens): # tokens is a list of Token objects
        self.tokens = tokens
        self.pos = 0
        if not self.tokens: # Handle empty token list by adding EOF
            self.tokens.append(Token(EOF, None, 1, 1)) # Default to line 1, col 1
        self.current_token = self.tokens[self.pos]
        if self.current_token.type == ERROR: # Check initial token
            raise ParseError(f"Lexer error: {self.current_token.value}", self.current_token)

    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
            if self.current_token.type == ERROR: # Check for lexer error on advance
                raise ParseError(f"Lexer error: {self.current_token.value}", self.current_token)
        else:
            # Simplified EOF token generation using the last valid token's line/col
            last_valid_token = self.tokens[-1] if self.tokens else Token(EOF, None, 1,1) # Guard against empty list after init
            line = last_valid_token.line
            col = last_valid_token.column + (len(str(last_valid_token.value)) if last_valid_token.value else 1) \
                  if last_valid_token.type != EOF else last_valid_token.column
            self.current_token = Token(EOF, None, line, col)


    def peek(self, offset=1):
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]

        last_token = self.tokens[-1] if self.tokens else None
        line = last_token.line if last_token and last_token.type != EOF else (self.current_token.line if self.current_token else 1)
        col = last_token.column if last_token and last_token.type != EOF else (self.current_token.column if self.current_token else 1)
        if last_token and last_token.type != EOF and last_token.value:
             col += len(str(last_token.value))
        return Token(EOF, None, line, col)

    def consume(self, expected_token_type, error_message=None):
        token = self.current_token
        if token.type == expected_token_type:
            self.advance()
            return token
        else:
            msg = error_message or f"Expected {expected_token_type} but got {token.type} ('{token.value}')"
            raise ParseError(msg, token)

    def consume_if(self, expected_token_type):
        if self.current_token.type == expected_token_type:
            token = self.current_token
            self.advance()
            return token
        return None

    def expect_newline_or_eof(self, message="Expected newline or end of statement"):
        if self.current_token.type == NEWLINE:
            return self.consume(NEWLINE)
        elif self.current_token.type == EOF:
            return self.current_token
        elif self.current_token.type == DEDENT:
            return self.current_token
        else:
            raise ParseError(message, self.current_token)

    def parse_program(self):
        location_info = (self.current_token.line, self.current_token.column)
        statements = []
        while self.current_token.type != EOF:
            if self.current_token.type == NEWLINE: # Handle multiple newlines by parsing them as EmptyStatements
                statements.append(self.parse_top_level_statement()) # Will parse NEWLINE as EmptyStatement
            elif self.current_token.type != EOF:
                stmt = self.parse_top_level_statement()
                statements.append(stmt)
            else:
                break

        # Do not consume EOF here, let the caller (e.g. test runner) verify final token
        # self.consume(EOF, "Expected end of file.")
        return ProgramNode(body=statements, location_info=location_info)

    def parse_top_level_statement(self):
        loc = (self.current_token.line, self.current_token.column)
        if self.current_token.type == NEWLINE:
            self.advance()
            return EmptyStatementNode(location_info=loc)

        # KW_MOD is for 'module' keyword as per EBNF
        if self.current_token.type == KW_FN: return self.parse_function_definition()
        if self.current_token.type == KW_OBJECT: return self.parse_object_definition()
        if self.current_token.type == KW_STORE: return self.parse_store_definition()
        if self.current_token.type == KW_MOD: return self.parse_module_definition()

        return self.parse_statement()

    def parse_statement(self):
        loc = (self.current_token.line, self.current_token.column)

        if self.current_token.type == KW_RETURN: return self.parse_return_statement()
        if self.current_token.type == KW_USE: return self.parse_use_statement()
        if self.current_token.type == KW_IF: return self.parse_if_then_else_statement()

        # New handling for prefix unless: if a statement starts with KW_UNLESS
        if self.current_token.type == KW_UNLESS:
            return self.parse_prefix_unless_statement()

        if self.current_token.type == KW_WHILE: return self.parse_while_loop()
        if self.current_token.type == KW_UNTIL: return self.parse_until_loop()
        if self.current_token.type == KW_ITERATE: return self.parse_iterate_loop()

        # Try parsing an expression first (for expression statements, assignments, or postfix unless)
        expr_node = self.parse_expression()

        # Check for postfix unless
        if self.consume_if(KW_UNLESS):
            condition_expr = self.parse_expression()
            stmt_loc = expr_node.location_info if hasattr(expr_node, 'location_info') else loc

            # Create FullExpressionNode and ExpressionStatementNode for the PostfixUnlessStatementNode
            full_expr_node = FullExpressionNode(expression=expr_node, error_handler=None, location_info=expr_node.location_info if hasattr(expr_node, 'location_info') else stmt_loc)
            expr_stmt_node = ExpressionStatementNode(expression=full_expr_node, location_info=stmt_loc)

            self.expect_newline_or_eof("Expected newline or EOF after postfix unless statement.")
            return PostfixUnlessStatementNode(expression_statement=expr_stmt_node, condition=condition_expr, location_info=stmt_loc)

        # Handle assignment or regular expression statement
        if self.consume_if(KW_IS):
            assign_loc = expr_node.location_info if hasattr(expr_node, 'location_info') else loc
            if not isinstance(expr_node, (IdentifierNode, PropertyAccessNode, ListElementAccessNode, DollarParamNode)): # Added DollarParamNode
                raise ParseError("Invalid target for assignment.", expr_node.location_info if hasattr(expr_node, 'location_info') else self.current_token)

            rhs_value_node = None
            if self.current_token.type == NEWLINE and self.peek().type == INDENT:
                rhs_value_node = self.parse_map_block_assignment_rhs()
            else:
                rhs_value_node = self.parse_expression()

            self.expect_newline_or_eof("Expected newline or EOF after assignment statement.")
            return AssignmentNode(target=expr_node, value=rhs_value_node, location_info=assign_loc)
        else:
            expr_loc = expr_node.location_info if hasattr(expr_node, 'location_info') else loc
            error_handler_node = None
            if self.current_token.type == KW_ERR:
                error_handler_node = self.parse_error_handler_suffix()

            full_expr_node = FullExpressionNode(expression=expr_node, error_handler=error_handler_node, location_info=expr_loc)
            self.expect_newline_or_eof("Expected newline or EOF after expression statement.")
            return ExpressionStatementNode(expression=full_expr_node, location_info=expr_loc)

    def parse_expression(self):
        return self.parse_ternary_conditional_expression()

    def parse_ternary_conditional_expression(self):
        loc = (self.current_token.line, self.current_token.column) # Initial loc
        expr = self.parse_logical_or_expression()
        if self.consume_if(SYM_QUESTION):
            true_expr = self.parse_expression()
            self.consume(SYM_EXCLAMATION, "Expected '!' in ternary expression.")
            false_expr = self.parse_expression()
            # Ternary node should use the location of '?'
            return TernaryConditionalExpressionNode(condition=expr, true_expr=true_expr, false_expr=false_expr, location_info=loc) # loc was from start of condition
        return expr

    def _parse_binary_expression(self, parse_higher_precedence_operand, operator_token_types):
        # loc of the first operand
        left_loc = (self.current_token.line, self.current_token.column)
        left = parse_higher_precedence_operand()

        while self.current_token.type in operator_token_types:
            op_token = self.current_token
            self.consume(op_token.type)
            right = parse_higher_precedence_operand()
            # BinaryOpNode location should ideally be the operator's location
            left = BinaryOpNode(left=left, operator=op_token.value, right=right, location_info=(op_token.line, op_token.column))
        return left

    def parse_logical_or_expression(self):
        return self._parse_binary_expression(self.parse_logical_and_expression, [KW_OR])

    def parse_logical_and_expression(self):
        return self._parse_binary_expression(self.parse_comparison_expression, [KW_AND])

    def parse_comparison_expression(self):
        # EBNF: ( '=' | 'equals' | '>' | '>=' | '<' | '<=' )
        # Assuming '=' for comparison is NOT SYM_ASSIGN. If lexer has SYM_EQ for this, use it.
        # For now, using KW_EQUALS for 'equals' and other distinct symbols.
        # If '=' as comparison is needed, lexer must provide a token like SYM_COMPARISON_EQ.
        comparison_ops = [KW_EQUALS, SYM_GT, SYM_GTE, SYM_LT, SYM_LTE]
        return self._parse_binary_expression(self.parse_additive_expression, comparison_ops)

    def parse_additive_expression(self):
        return self._parse_binary_expression(self.parse_multiplicative_expression, [SYM_PLUS, SYM_MINUS])

    def parse_multiplicative_expression(self):
        # SYM_OPERATOR_MOD is '%'
        return self._parse_binary_expression(self.parse_unary_expression, [SYM_MUL, SYM_DIV, SYM_OPERATOR_MOD])

    def parse_unary_expression(self):
        loc = (self.current_token.line, self.current_token.column)
        operator = None
        op_val = None
        if self.current_token.type == SYM_MINUS:
            op_token = self.consume(SYM_MINUS)
            operator = op_token.value
            loc = (op_token.line, op_token.column) # operator loc
        elif self.current_token.type == KW_NOT:
            op_token = self.consume(KW_NOT)
            operator = op_token.value
            loc = (op_token.line, op_token.column) # operator loc

        operand = self.parse_value_or_invocation()

        if operator:
            return UnaryOpNode(operator=operator, operand=operand, location_info=loc)
        return operand

    def parse_value_or_invocation(self):
        base_expr_loc = (self.current_token.line, self.current_token.column)
        base_expr = self.parse_primary_expression_base()

        # Loop for property access or list element access
        while True:
            current_op_loc = (self.current_token.line, self.current_token.column)
            if self.current_token.type == SYM_DOT and self.peek().type == IDENTIFIER:
                self.consume(SYM_DOT)
                prop_token = self.consume(IDENTIFIER)
                prop_ident_node = IdentifierNode(prop_token.value, location_info=(prop_token.line, prop_token.column))
                base_expr = PropertyAccessNode(base_expr=base_expr, property_name=prop_ident_node, location_info=current_op_loc) # loc of '.'
            elif self.current_token.type == SYM_LPAREN:
                # Try to distinguish list access `base_expr(index_expr)` from a call `base_expr(args...)`
                # EBNF implies suffixes (like list access) are parsed before attempting call_operation.
                # Lookahead to see if it's NOT a list access.
                # If it looks like a call (e.g. named args, empty args, multiple args), break to parse as call.
                peek_token = self.peek()
                peek_after_peek_token = self.peek(2)

                if peek_token.type == SYM_RPAREN: # e.g. base() -> call, not list access
                    break
                if peek_token.type == IDENTIFIER and peek_after_peek_token.type == SYM_COLON: # e.g. base(name: val) -> call
                    break
                # If we see `expr ,`, it's likely a call with multiple arguments.
                # This requires parsing an expression first to see if a comma follows.
                # For simplicity, if it's not an obvious call starter, attempt to parse as list access.
                # A more robust method would involve more sophisticated lookahead or tentative parsing.

                # Try to parse as list_element_access: ( expression )
                # If this structure is not strictly met, it will either error or (ideally) be handled by call parsing.
                # This is a point where grammar ambiguity `a(b)` (call or access) is tricky without type info or stricter syntax.
                # Assuming list access is `(one_expression_then_rparen)`
                # We will consume LPAREN, parse one expression, then expect RPAREN.
                # If a comma appears after the expression, it's a call, and we should have broken.
                # This path is taken if it's not `()` or `(ident: val)`.
                # It could be `(expr)` or `(expr, ...)`. We want to parse `(expr)` as list access.
                # If after parsing `expr`, the next token is COMMA, it's a call.

                # Let's check if the expression inside LPAREN is followed by COMMA
                # This is still tricky. For now, let's assume if it's not clearly a multi-arg or named-arg call,
                # we attempt list access. The ambiguity of `a(b)` will default to list access if `b` is a simple expression.
                # If `parse_expression()` consumes tokens and then a comma is found, we can't easily backtrack here.

                # Simplified approach: if it wasn't `()` or `(ident: val)`, try list access.
                # If `parse_expression` inside list access fails or `)` isn't found, it's a syntax error for list access.
                self.consume(SYM_LPAREN)
                index_expr = self.parse_expression()
                # If after parsing index_expr, the current token is a comma, then it was actually a call.
                # This is hard to backtrack from.
                # The current EBNF implies `list_element_access_suffix` is `( expression )`.
                # So, if there's a comma, it means it wasn't a valid list_element_access_suffix.
                if self.current_token.type == SYM_COMMA:
                    # This indicates it was actually a call like `foo(arg1, arg2)`.
                    # Backtracking is needed here. This is a limitation of the current single-pass parser.
                    # For now, this will lead to a parse error expecting ')' if a comma is found.
                    # This is an area for future improvement (e.g. tentative parsing or more lookahead).
                    # To make progress, we will assume that if it's not `()` or `(ident: val)`,
                    # and we parse an expression, the next token *must* be `)` for it to be a list access.
                    # If it's a comma, then `consume(SYM_RPAREN)` below will fail.
                    pass # Let consume RPAREN handle it.

                self.consume(SYM_RPAREN, "Expected ')' after list element access index.")
                base_expr = ListElementAccessNode(base_expr=base_expr, index_expr=index_expr, location_info=current_op_loc) # loc of '('
            else:
                break

        if self.is_start_of_call_operation(base_expr):
            return self.parse_call_operation(base_expr)

        return base_expr

    # Removed is_start_of_call_operation_args as its logic is being integrated or simplified
    # def is_start_of_call_operation_args(self, callee_expr): ...

    def is_start_of_call_operation(self, callee_expr):
        if self.current_token.type == SYM_DOT and (self.peek().type == IDENTIFIER or self.peek().type == KW_ACROSS):
            return True

        # If callee is an Identifier and next token can start arguments (paren, or a potential first arg for no-paren call)
        if isinstance(callee_expr, IdentifierNode):
            # Potential start of no_paren_space_separated_argument_list or paren_argument_list
            if self.current_token.type == SYM_LPAREN: return True # LPAREN always means call if it reaches here
            # Check for tokens that can start an expression but are not binary operators that would bind tighter
            # This is to allow `func arg1 arg2`
            if self.current_token.type in [IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL, KW_TRUE, KW_NO, KW_EMPTY, KW_NOW, DOLLAR_PARAM] and \
               self.current_token.type not in self.get_binary_operator_tokens() + [KW_IS, KW_ERR, KW_UNLESS, NEWLINE, EOF, DEDENT, SYM_QUESTION, SYM_EXCLAMATION, SYM_RPAREN, SYM_COMMA, SYM_COLON]:
                return True

        # If callee_expr is some other callable expression, e.g. (get_func()) arg1 ...
        # This typically requires parentheses for the arguments: (get_func())(arg1)
        # If LPAREN is current after suffixes, it must be a call.
        if self.current_token.type == SYM_LPAREN: # Handles cases like (expr)(args) or list[i](args)
             return True

        return False


    def parse_primary_expression_base(self):
        loc = (self.current_token.line, self.current_token.column)
        token = self.current_token

        if token.type == IDENTIFIER:
            self.advance()
            return IdentifierNode(token.value, location_info=loc)
        elif token.type == INTEGER_LITERAL:
            self.advance()
            # Lexer should provide actual int/float value if possible
            return LiteralNode(int(token.value), 'INTEGER', location_info=loc)
        elif token.type == FLOAT_LITERAL:
            self.advance()
            return LiteralNode(float(token.value), 'FLOAT', location_info=loc)
        elif token.type == STRING_LITERAL:
            self.advance()
            return LiteralNode(token.value, 'STRING', location_info=loc)
        elif token.type == KW_TRUE:
            self.advance()
            return LiteralNode(True, 'BOOLEAN', location_info=loc)
        elif token.type == KW_NO:
            self.advance()
            return LiteralNode(False, 'BOOLEAN', location_info=loc)
        elif token.type == KW_EMPTY:
            self.advance()
            return LiteralNode(None, 'EMPTY', location_info=loc)
        elif token.type == KW_NOW:
            self.advance()
            return LiteralNode("now", 'NOW', location_info=loc)
        elif token.type == DOLLAR_PARAM:
            self.advance()
            val = token.value
            if val.startswith('$') and val[1:].isdigit(): val = int(val[1:])
            elif val.startswith('$'): val = val[1:]
            return DollarParamNode(val, location_info=loc)
        elif token.type == SYM_LPAREN:
            self.advance()

            if self.current_token.type == SYM_RPAREN: # Empty list ()
                self.advance()
                return ListLiteralNode(elements=[], location_info=loc)

            # Distinguish map vs list/grouped expr
            # Check for map_entry: IDENTIFIER ':'
            is_map = False
            if self.current_token.type == IDENTIFIER and self.peek().type == SYM_COLON:
                is_map = True

            if is_map:
                entries = []
                entries.append(self.parse_map_entry())
                while self.consume_if(SYM_COMMA):
                    if self.current_token.type == SYM_RPAREN: break # Trailing comma
                    entries.append(self.parse_map_entry())
                self.consume(SYM_RPAREN, "Expected ')' after map literal.")
                return MapLiteralNode(entries=entries, location_info=loc)
            else:
                first_expr = self.parse_expression()
                if self.consume_if(SYM_COMMA): # List literal
                    elements = [first_expr]
                    if self.current_token.type != SYM_RPAREN:
                         elements.append(self.parse_expression())
                         while self.consume_if(SYM_COMMA):
                            if self.current_token.type == SYM_RPAREN: break # Trailing comma
                            elements.append(self.parse_expression())
                    self.consume(SYM_RPAREN, "Expected ')' after list literal.")
                    return ListLiteralNode(elements=elements, location_info=loc)
                else: # Grouped expression
                    self.consume(SYM_RPAREN, "Expected ')' after grouped expression.")
                    # Return the inner expression, location of original LPAREN might be useful for source mapping.
                    # For AST simplicity, just return first_expr. If GroupedExpressionNode is needed, wrap it.
                    return first_expr
        else:
            expected_things = "a literal (integer, float, string), an identifier, '$parameter', '(', or keywords like 'true', 'no', 'empty', 'now'"
            raise ParseError(f"Unexpected token {token.type} ('{token.value}') when expecting {expected_things}.", token)

    def parse_map_entry(self):
        loc = (self.current_token.line, self.current_token.column)
        key_token = self.consume(IDENTIFIER)
        key_node = IdentifierNode(key_token.value, location_info=(key_token.line, key_token.column))
        self.consume(SYM_COLON, "Expected ':' in map entry.")
        value_expr = self.parse_expression()
        return MapEntryNode(key=key_node, value=value_expr, location_info=loc)

    def parse_function_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_FN)
        name_token = self.consume(IDENTIFIER)
        name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))

        params = []
        if self.consume_if(SYM_LPAREN):
            if self.current_token.type != SYM_RPAREN:
                params.append(self.parse_parameter_definition())
                while self.consume_if(SYM_COMMA):
                    if self.current_token.type == SYM_RPAREN: break # Trailing comma
                    params.append(self.parse_parameter_definition())
            self.consume(SYM_RPAREN, "Expected ')' after function parameters.")

        body_node = self.parse_function_body_content(name_node.name)
        return FunctionDefinitionNode(name=name_node, params=params, body=body_node, location_info=loc)

    def parse_parameter_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        name_token = self.consume(IDENTIFIER)
        name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))
        default_value = None
        if self.consume_if(SYM_COLON):
            default_value = self.parse_expression()
        return ParameterNode(name=name_node, default_value=default_value, location_info=loc)

    def parse_object_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_OBJECT)
        name_token = self.consume(IDENTIFIER)
        name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))
        self.consume(NEWLINE)
        self.consume(INDENT)
        members = []
        while self.current_token.type != DEDENT and self.current_token.type != EOF:
            # Skip any blank lines
            while self.current_token.type == NEWLINE:
                self.advance()
            # If after skipping newlines we are at DEDENT or EOF, break the loop
            if self.current_token.type == DEDENT or self.current_token.type == EOF:
                break
            members.append(self._parse_field_or_method_member()) # Use helper
        self.consume(DEDENT)
        return ObjectDefinitionNode(name=name_node, members=members, location_info=loc)

    def _parse_field_or_method_member(self, is_store_context=False):
        # Helper to parse field or method, can be used by object and store
        member_loc = (self.current_token.line, self.current_token.column)

        # Store-specific members first if in that context
        if is_store_context:
            if self.current_token.type == SYM_AMPERSAND: return self.parse_relation_definition()
            if self.current_token.type == KW_AS: return self.parse_cast_definition()
            if self.current_token.type == SYM_AT: return self.parse_receive_handler()

        # Common field/method parsing (IDENTIFIER must be current token)
        member_name_token = self.consume(IDENTIFIER, "Expected identifier for field or method name.")
        member_name_node = IdentifierNode(member_name_token.value, location_info=(member_name_token.line, member_name_token.column))

        # Refined EBNF-based distinction:
        # method_definition = IDENTIFIER , [ function_parameters ] , function_body ;
        # field_definition  = IDENTIFIER , [ '?' , expression ] , NEWLINE ;

        # 1. Check for `IDENTIFIER (` (Method with parameters)
        if self.current_token.type == SYM_LPAREN:
            params = []
            self.consume(SYM_LPAREN) # Consume the LPAREN
            if self.current_token.type != SYM_RPAREN:
                params.append(self.parse_parameter_definition())
                while self.consume_if(SYM_COMMA):
                    if self.current_token.type == SYM_RPAREN: break # Trailing comma
                    params.append(self.parse_parameter_definition())
            self.consume(SYM_RPAREN, "Expected ')' after method parameters.")
            body_node = self.parse_function_body_content(member_name_node.name)
            return MethodDefinitionNode(name=member_name_node, params=params, body=body_node, location_info=member_loc)

        # 2. Check for `IDENTIFIER ?` (Field with default value)
        elif self.current_token.type == SYM_QUESTION:
            self.consume(SYM_QUESTION) # Consume '?'
            default_value = self.parse_expression()
            self.consume(NEWLINE, f"Expected newline after field definition for '{member_name_node.name}'.")
            return FieldDefinitionNode(name=member_name_node, default_value=default_value, location_info=member_loc)

        # 3. Check for `IDENTIFIER NEWLINE`
        elif self.current_token.type == NEWLINE:
            # Peek ahead to distinguish `IDENTIFIER NEWLINE INDENT` (method) from `IDENTIFIER NEWLINE` (field)
            if self.peek().type == INDENT: # Method with block body, no params
                # `parse_function_body_content` expects to be called when NEWLINE is current and INDENT follows
                body_node = self.parse_function_body_content(member_name_node.name)
                return MethodDefinitionNode(name=member_name_node, params=[], body=body_node, location_info=member_loc)
            else: # Field `IDENTIFIER NEWLINE` (no default value)
                self.consume(NEWLINE) # Consume the NEWLINE
                return FieldDefinitionNode(name=member_name_node, default_value=None, location_info=member_loc)

        # 4. `IDENTIFIER expression NEWLINE` (Method with single expression body, no params)
        # This is the case if current_token is not LPAREN, QUESTION, or NEWLINE after IDENTIFIER.
        else:
            # This implies a method with no parameters and a single expression body.
            # `parse_function_body_content` handles the `expression NEWLINE` case.
            params = [] # No parameters
            body_node = self.parse_function_body_content(member_name_node.name)
            return MethodDefinitionNode(name=member_name_node, params=params, body=body_node, location_info=member_loc)


    def parse_function_body_content(self, func_name_for_error="function"):
        # EBNF for function_body:
        # ( expression , NEWLINE )
        # | ( NEWLINE , INDENT , { statement } , [ [ 'return' ] , expression , NEWLINE ] , DEDENT )
        # | ( NEWLINE , INDENT , { statement } , DEDENT )
        body_node = None
        if self.current_token.type == NEWLINE and self.peek().type == INDENT: # Block body
            self.consume(NEWLINE)
            self.consume(INDENT)
            body_statements = []
            # Parse statements
            while self.current_token.type != DEDENT and self.current_token.type != EOF:
                # The EBNF `[ [ 'return' ] , expression , NEWLINE ]` before DEDENT
                # is handled naturally by self.parse_statement().
                # If 'return expression NEWLINE' is last, it's a ReturnStatementNode.
                # If 'expression NEWLINE' is last, it's an ExpressionStatementNode.
                # Semantic analysis can determine if the last ExpressionStatementNode
                # should be treated as an implicit return.
                body_statements.append(self.parse_statement())

            # After loop, if last statement was an expression that should be an implicit return,
            # it's already in body_statements.
            self.consume(DEDENT, f"Expected DEDENT at end of block for {func_name_for_error}")
            body_node = body_statements
        else: # Single expression body: expression , NEWLINE
            expr_loc = (self.current_token.line, self.current_token.column)
            body_expr = self.parse_expression()
            self.expect_newline_or_eof(f"Expected newline or EOF after single expression body for {func_name_for_error}.")
            # body_expr itself is the node (e.g. ExpressionNode)
            body_node = body_expr

        return body_node

    # Removed peek_is_dedent_after_possible_return as its logic is subsumed by the main statement parsing loop.
    # The EBNF's optional final return expression is handled by parse_statement() if present.

    def parse_store_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_STORE)
        is_actor = bool(self.consume_if(KW_ACTOR))
        name_token = self.consume(IDENTIFIER, "Expected identifier for store name.")
        name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))

        for_target_node = None
        if self.consume_if(KW_FOR):
            for_token = self.consume(IDENTIFIER, "Expected identifier for store 'for' target.")
            for_target_node = IdentifierNode(for_token.value, location_info=(for_token.line, for_token.column))

        self.consume(NEWLINE, "Expected newline after store declaration.")
        self.consume(INDENT, "Expected indent for store body.")

        members = []
        while self.current_token.type != DEDENT and self.current_token.type != EOF:
            members.append(self.parse_store_member())

        self.consume(DEDENT, "Expected dedent after store body.")
        return StoreDefinitionNode(name=name_node, is_actor=is_actor, for_target=for_target_node, members=members, location_info=loc)

    def parse_store_member(self):
        # store_member = field_definition | method_definition | relation_definition | cast_definition | receive_handler ;
        # Use the _parse_field_or_method_member helper, which now includes store-specific checks.
        return self._parse_field_or_method_member(is_store_context=True)

    def parse_relation_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(SYM_AMPERSAND, "Expected '&' for relation definition.")
        name_token = self.consume(IDENTIFIER, "Expected identifier for relation name.")
        name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))
        self.consume(NEWLINE, "Expected newline after relation definition.")
        return RelationDefinitionNode(name=name_node, location_info=loc)

    def parse_cast_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_AS, "Expected 'as' for cast definition.")

        type_token = self.consume(IDENTIFIER, "Expected 'string', 'map', or 'list' for cast type.")
        cast_to_type = type_token.value
        if cast_to_type not in ['string', 'map', 'list']:
            raise ParseError(f"Invalid cast type '{cast_to_type}'. Expected 'string', 'map', or 'list'.", type_token)

        # cast_body = ( expression , NEWLINE ) | ( NEWLINE , INDENT , { ( expression | assignment ) , NEWLINE } , DEDENT ) ;
        cast_body_node = None
        if self.current_token.type == NEWLINE and self.peek().type == INDENT: # Block form
            self.consume(NEWLINE)
            self.consume(INDENT)
            body_items = []
            while self.current_token.type != DEDENT and self.current_token.type != EOF:
                # Try parsing assignment first (IDENTIFIER 'is' expr)
                # This requires more lookahead or a structure that allows parsing either.
                # For now, assume expression, as assignment is also an expression (though statement here).
                # The EBNF `(expression | assignment)` means we try one, then the other, or parse expression and check if it was assignment.
                # Let's parse as statement, which can be assignment or expression statement
                stmt = self.parse_statement() # This will consume its own NEWLINE.
                body_items.append(stmt)
                # self.consume(NEWLINE, "Newline expected after each item in cast block body.") # parse_statement handles its newline
            self.consume(DEDENT, "Expected dedent at end of cast block body.")
            cast_body_node = body_items # List of statements (AssignmentNode or ExpressionStatementNode)
        else: # Single expression form
            cast_body_node = self.parse_expression()
            self.consume(NEWLINE, "Expected newline after single expression cast body.")

        return CastDefinitionNode(cast_to_type=cast_to_type, body=cast_body_node, location_info=loc)

    def parse_receive_handler(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(SYM_AT, "Expected '@' for receive handler.")
        message_name_token = self.consume(IDENTIFIER, "Expected identifier for receive message name.")
        message_name_node = IdentifierNode(message_name_token.value, location_info=(message_name_token.line, message_name_token.column))

        # function_body is parsed next
        body_node = self.parse_function_body_content(f"receive handler '{message_name_node.name}'")
        return ReceiveHandlerNode(message_name=message_name_node, body=body_node, location_info=loc)

    def parse_module_definition(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_MOD) # 'mod' keyword
        name_token = self.consume(IDENTIFIER)
        name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))

        # Module body: NEWLINE INDENT { top_level_statement } DEDENT
        self.consume(NEWLINE, "Expected newline after module declaration.")
        self.consume(INDENT, "Expected indent for module body.")

        body_statements = []
        while self.current_token.type != DEDENT and self.current_token.type != EOF:
            body_statements.append(self.parse_top_level_statement())

        self.consume(DEDENT, "Expected dedent after module body.")
        return ModuleDefinitionNode(name=name_node, body=body_statements, location_info=loc)


    def parse_return_statement(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_RETURN)
        expr = None
        if self.current_token.type not in [NEWLINE, EOF, DEDENT]:
            expr = self.parse_expression()
        self.expect_newline_or_eof("Expected newline or EOF after return statement.")
        return ReturnStatementNode(value=expr, location_info=loc)

    def parse_use_statement(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_USE)
        qid = self.parse_qualified_identifier()
        self.expect_newline_or_eof("Expected newline or EOF after use statement.")
        return UseStatementNode(qualified_identifier=qid, location_info=loc)

    def parse_qualified_identifier(self):
        loc = (self.current_token.line, self.current_token.column)
        parts = []
        id_token = self.consume(IDENTIFIER)
        parts.append(IdentifierNode(id_token.value, location_info=(id_token.line, id_token.column)))
        while self.consume_if(SYM_DOT):
            id_token = self.consume(IDENTIFIER)
            parts.append(IdentifierNode(id_token.value, location_info=(id_token.line, id_token.column)))
        return QualifiedIdentifierNode(parts=parts, location_info=loc)

    def parse_map_block_assignment_rhs(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(NEWLINE) # Already consumed by 'is' if line broken, or current if same line
        self.consume(INDENT)
        entries = []
        while self.current_token.type != DEDENT and self.current_token.type != EOF:
            entries.append(self.parse_map_block_entry())
        self.consume(DEDENT)
        # Trailing newline after DEDENT is handled by expect_newline_or_eof in assignment
        return MapBlockAssignmentRHSNode(entries=entries, location_info=loc)

    def parse_map_block_entry(self):
        loc = (self.current_token.line, self.current_token.column)
        key_token = self.consume(IDENTIFIER)
        key_node = IdentifierNode(key_token.value, location_info=(key_token.line, key_token.column))
        self.consume(KW_IS) # EBNF: map_block_entry = IDENTIFIER , 'is' , expression , NEWLINE ;
        value_expr = self.parse_expression()
        self.consume(NEWLINE, "Expected newline after map block entry.")
        return MapBlockEntryNode(key=key_node, value=value_expr, location_info=loc)

    def parse_error_handler_suffix(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_ERR)
        action_node = None
        if self.current_token.type == KW_RETURN:
            ret_loc = (self.current_token.line, self.current_token.column)
            self.consume(KW_RETURN)
            expr = None
            if self.current_token.type not in [NEWLINE, EOF, DEDENT]:
                expr = self.parse_expression()
            action_node = ReturnStatementNode(value=expr, location_info=ret_loc)
        elif self.current_token.type == NEWLINE and self.peek().type == INDENT:
            action_node = self.parse_statement_or_block_content_for_error_handler()
        else:
            action_node = self.parse_expression()
        return ErrorHandlerSuffixNode(action=action_node, location_info=loc)

    def parse_statement_or_block_content_for_error_handler(self):
        self.consume(NEWLINE)
        self.consume(INDENT)
        statements = []
        while self.current_token.type != DEDENT and self.current_token.type != EOF:
            statements.append(self.parse_statement())
        self.consume(DEDENT)
        return statements

    def parse_call_operation(self, callee_expr):
        call_loc = callee_expr.location_info # Use callee's location as start of call operation
        actual_callee = callee_expr
        arguments = []
        # call_style can be simplified or made more granular if needed by AST consumers or for debugging.
        # For now, it primarily distinguishes dot vs direct and paren vs no-paren.
        call_style = ""

        # Path 1: Dot Call (e.g., callee_expr.method_or_across ...)
        if self.current_token.type == SYM_DOT and (self.peek().type == IDENTIFIER or self.peek().type == KW_ACROSS):
            dot_loc = (self.current_token.line, self.current_token.column)
            self.consume(SYM_DOT)
            method_or_across_token = self.current_token
            self.advance() # Consume IDENTIFIER or KW_ACROSS

            prop_name_val = method_or_across_token.value
            prop_loc = (method_or_across_token.line, method_or_across_token.column)
            prop_ident_node = IdentifierNode(prop_name_val, location_info=prop_loc)
            actual_callee = PropertyAccessNode(base_expr=callee_expr, property_name=prop_ident_node, location_info=dot_loc)

            method_type = "dot_method" if method_or_across_token.type == IDENTIFIER else "dot_across"

            # Arguments for dot call
            if self.current_token.type == SYM_LPAREN:
                arguments, _ = self.parse_paren_argument_list() # parse_paren_argument_list returns (args, style_string)
                call_style = f"{method_type}_paren"
            # Check for no_paren_space_separated_argument_list
            elif self.current_token.type not in [NEWLINE, EOF, DEDENT, KW_IS, KW_ERR, KW_UNLESS, SYM_DOT, SYM_LPAREN, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_QUESTION, SYM_EXCLAMATION] + list(self.get_binary_operator_tokens()):
                arguments, _ = self.parse_no_paren_space_separated_argument_list()
                call_style = f"{method_type}_no_paren_space"
            else: # No arguments
                arguments = []
                call_style = f"{method_type}_empty"

        # Path 2: Direct Call (e.g., callee_expr(args...) or callee_expr arg1 arg2 ...)
        # Current token is NOT SYM_DOT. callee_expr is the function.
        else:
            if self.current_token.type == SYM_LPAREN:
                arguments, _ = self.parse_paren_argument_list()
                call_style = "direct_paren"
            # Check for no_paren_space_separated_argument_list.
            # This form is generally applicable whether callee_expr is an IdentifierNode or a more complex expression like (get_func()).
            # The condition checks if the current token can start an expression for an argument.
            elif self.current_token.type not in [NEWLINE, EOF, DEDENT, KW_IS, KW_ERR, KW_UNLESS, SYM_DOT, SYM_LPAREN, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_QUESTION, SYM_EXCLAMATION] + list(self.get_binary_operator_tokens()):
                arguments, _ = self.parse_no_paren_space_separated_argument_list()
                call_style = "direct_no_paren_space"
                # Could add detail: if isinstance(actual_callee, IdentifierNode): call_style += "_identifier_callee"
            else: # No arguments, or not a recognized argument pattern for a direct call
                arguments = []
                call_style = "direct_empty"

        return CallOperationNode(callee=actual_callee, arguments=arguments, call_style=call_style, location_info=call_loc)

    # Removed peek_has_comma_before_terminator as no_paren_comma_separated_argument_list is removed for simplification.
    # def peek_has_comma_before_terminator(self): ...

    # Removed parse_no_paren_comma_separated_argument_list to simplify argument parsing.
    # No-paren calls will rely on space separation. Comma separation is for within parens.
    # def parse_no_paren_comma_separated_argument_list(self): ...

    def parse_paren_argument_list(self):
        self.consume(SYM_LPAREN)
        args = []
        if self.current_token.type != SYM_RPAREN:
            args.append(self.parse_argument())
            while self.consume_if(SYM_COMMA):
                if self.current_token.type == SYM_RPAREN: break
                args.append(self.parse_argument())
        self.consume(SYM_RPAREN, "Expected ')' after parenthesized argument list.")
        return args, "paren_args"

    def parse_no_paren_space_separated_argument_list(self):
        args = []
        # Check if the current token can start an argument and is not a terminating token or operator
        while self.current_token.type not in [NEWLINE, EOF, DEDENT, KW_IS, KW_ERR, KW_UNLESS, SYM_DOT, SYM_LPAREN, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_QUESTION, SYM_EXCLAMATION] + list(self.get_binary_operator_tokens()):
            args.append(self.parse_expression_for_no_paren_call()) # Arguments are expressions themselves
        return args, "no_paren_space_args"

    def get_binary_operator_tokens(self):
        return [KW_OR, KW_AND, KW_EQUALS, SYM_GT, SYM_GTE, SYM_LT, SYM_LTE,
                SYM_PLUS, SYM_MINUS, SYM_MUL, SYM_DIV, SYM_OPERATOR_MOD]

    def parse_expression_for_no_paren_call(self):
        # EBNF: expression_for_no_paren_call = primary_expression_base | additive_expression ;
        # This means it will not parse comparison, logical_or, logical_and if they are compound,
        # as those are higher precedence than additive_expression.
        # `parse_additive_expression` correctly handles this: it will parse a primary_expression_base
        # if no additive operators (+, -) are found at its precedence level.
        return self.parse_additive_expression()

    def parse_argument(self):
        loc = (self.current_token.line, self.current_token.column)
        name_node = None
        if self.current_token.type == IDENTIFIER and self.peek().type == SYM_COLON:
            name_token = self.consume(IDENTIFIER)
            name_node = IdentifierNode(name_token.value, location_info=(name_token.line, name_token.column))
            self.consume(SYM_COLON)

        value_expr = self.parse_expression()
        return ArgumentNode(value=value_expr, name=name_node, location_info=loc)

    def parse_if_then_else_statement(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_IF)
        condition = self.parse_expression()
        if_block = self.parse_statement_or_block()

        else_if_clauses = []
        while self.current_token.type == KW_ELSE and self.peek().type == KW_IF:
            self.consume(KW_ELSE)
            else_if_loc = (self.current_token.line, self.current_token.column)
            self.consume(KW_IF)
            else_if_cond = self.parse_expression()
            else_if_block_content = self.parse_statement_or_block()
            else_if_clauses.append({'condition': else_if_cond, 'block': else_if_block_content, 'location_info': else_if_loc})

        else_block_content = None
        if self.consume_if(KW_ELSE):
            else_block_content = self.parse_statement_or_block()

        return IfThenElseStatementNode(condition=condition, if_block=if_block, else_if_clauses=else_if_clauses, else_block=else_block_content, location_info=loc)

    def parse_prefix_unless_statement(self): # Renamed for clarity
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_UNLESS)
        condition = self.parse_expression()
        block = self.parse_statement_or_block()
        return UnlessStatementNode(condition=condition, block=block, location_info=loc)

    def parse_statement_or_block(self):
        if self.current_token.type == NEWLINE and self.peek().type == INDENT:
            self.consume(NEWLINE)
            self.consume(INDENT)
            statements = []
            while self.current_token.type != DEDENT and self.current_token.type != EOF:
                statements.append(self.parse_statement())
            self.consume(DEDENT)
            return statements
        else:
            # A single statement. It should end with a NEWLINE (or EOF/DEDENT)
            # which is typically consumed by the caller or expect_newline_or_eof.
            # Here we just parse the statement itself.
            # `parse_statement()` is responsible for consuming its own terminating newline.
            stmt = self.parse_statement()
            # The call to expect_newline_or_eof here was redundant because parse_statement handles it.
            return stmt


    def parse_while_loop(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_WHILE)
        condition = self.parse_expression()
        body = self.parse_statement_or_block()
        return WhileLoopNode(condition=condition, body=body, location_info=loc)

    def parse_until_loop(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_UNTIL)
        condition = self.parse_expression()
        body = self.parse_statement_or_block()
        return UntilLoopNode(condition=condition, body=body, location_info=loc)

    def parse_iterate_loop(self):
        loc = (self.current_token.line, self.current_token.column)
        self.consume(KW_ITERATE)
        iterable_expr = self.parse_expression()

        loop_variable_node = None
        if self.consume_if(SYM_LPAREN):
            var_token = self.consume(IDENTIFIER)
            loop_variable_node = IdentifierNode(var_token.value, location_info=(var_token.line, var_token.column))
            self.consume(SYM_RPAREN, "Expected ')' after iterate loop variable.")

        body = self.parse_statement_or_block()
        return IterateLoopNode(iterable=iterable_expr, loop_variable=loop_variable_node, body=body, location_info=loc)
```
