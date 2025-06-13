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
        # Postfix unless is handled after expression parsing. This is for prefix unless.
        if self.current_token.type == KW_UNLESS and self.peek().type != NEWLINE and self.peek().type != EOF :
            # Check if it's prefix unless: `unless expr block` vs `expr unless expr`
            # This is a heuristic: if `unless` is not immediately followed by something that looks like the end of `expr unless cond`,
            # then it's a prefix unless.
            # A more robust way would be to parse expr, then check for `unless`. If not, and current was `unless`, backtrack or structure differently.
            # For now, assume if we are here, it's a prefix `unless condition block`.
            # The `parse_statement` logic for postfix unless handles the other case.
            # This means `parse_unless_statement` will only be called for prefix form.
            return self.parse_prefix_unless_statement()

        if self.current_token.type == KW_WHILE: return self.parse_while_loop()
        if self.current_token.type == KW_UNTIL: return self.parse_until_loop()
        if self.current_token.type == KW_ITERATE: return self.parse_iterate_loop()

        expr_node = self.parse_expression()

        if self.consume_if(KW_UNLESS):
            condition_expr = self.parse_expression()
            # Original expr_node location might be more accurate for the statement itself
            stmt_loc = expr_node.location_info if hasattr(expr_node, 'location_info') else loc
            full_expr_node = FullExpressionNode(expression=expr_node, error_handler=None, location_info=expr_node.location_info if hasattr(expr_node, 'location_info') else stmt_loc)
            stmt_node = ExpressionStatementNode(expression=full_expr_node, location_info=stmt_loc)
            self.expect_newline_or_eof("Expected newline or EOF after postfix unless statement.")
            return PostfixUnlessStatementNode(expression_statement=stmt_node, condition=condition_expr, location_info=stmt_loc)

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
            elif self.current_token.type == SYM_LPAREN and not self.is_start_of_call_operation_args(base_expr):
                 # This check `is_start_of_call_operation_args` is tricky.
                 # EBNF: value_or_invocation = primary_expression_base , { property_access_suffix | list_element_access_suffix } , [ call_operation ] ;
                 # This means suffixes are parsed first. Then call_operation.
                 # So if LPAREN is for list access, it's consumed here. If for call, it's handled later.
                 # Heuristic: if what's inside LPAREN looks like a simple expression for index, it's access.
                 # This can be ambiguous: `a(b)` could be call or access. Context or grammar rules needed.
                 # For now, assume if not clearly a call's arg structure, it could be access.
                 # This is a known simplification point.
                self.consume(SYM_LPAREN)
                index_expr = self.parse_expression()
                self.consume(SYM_RPAREN, "Expected ')' after list element access index.")
                base_expr = ListElementAccessNode(base_expr=base_expr, index_expr=index_expr, location_info=current_op_loc) # loc of '('
            else:
                break

        if self.is_start_of_call_operation(base_expr):
            return self.parse_call_operation(base_expr)

        return base_expr

    def is_start_of_call_operation_args(self, callee_expr):
        # Simplified check if LPAREN starts arguments for a call vs. list access or grouping
        if self.current_token.type == SYM_LPAREN:
            # If the callee is an identifier or property access, LPAREN is more likely for a call.
            if isinstance(callee_expr, (IdentifierNode, PropertyAccessNode)):
                # Further check: list access `obj(index)` vs call `func(arg)`.
                # If `callee_expr` cannot be called (e.g., a non-function variable), then `(..)` must be list access.
                # This requires type system info not available at parse time.
                # A common syntactic distinction: `a (b)` (call with space) vs `a(b)` (list access or call).
                # Coral EBNF doesn't show space sensitivity here typically.
                # For now, assume `is_start_of_call_operation` will make the main decision.
                # This helper is mostly to prevent list access from consuming call's LPAREN.
                # If `value_or_invocation` parses suffixes first, then `call_operation`, then list access should be greedily matched if it's `base(index_expr)`.
                # This means `is_start_of_call_operation_args` should probably return False more often to let list access try first.
                # Let's assume `value_or_invocation` structure handles this: suffixes are tried, then call.
                # So if `LPAREN` is encountered in the suffix loop, it must be list access.
                # The check `and not self.is_start_of_call_operation_args(base_expr)` in `parse_value_or_invocation` loop for LPAREN
                # is trying to distinguish. If it *is* call args, then the suffix loop should break.
                # This is circular. A better way:
                # In `parse_value_or_invocation` loop:
                # if SYM_DOT: property access
                # if SYM_LPAREN and LOOKAHEAD indicates list access (e.g. simple expr then RPAREN, not full arg list): list access
                # else: break from suffix loop.
                # Then, after suffix loop, try `parse_call_operation`.
                # For now, the existing structure is kept. This is a point of fragility.
                return True # Overly simple: assume LPAREN after a potential callee is args start.
        return False

    def is_start_of_call_operation(self, callee_expr):
        if self.current_token.type == SYM_DOT and (self.peek().type == IDENTIFIER or self.peek().type == KW_ACROSS):
            return True

        # If callee is an Identifier and next token can start arguments (paren, or a potential first arg for no-paren call)
        if isinstance(callee_expr, IdentifierNode):
            # Potential start of no_paren_space_separated_argument_list or paren_argument_list
            if self.current_token.type == SYM_LPAREN: return True
            # Check for tokens that can start an expression but are not binary operators that would bind tighter
            # This is to allow `func arg1 arg2`
            if self.current_token.type in [IDENTIFIER, INTEGER_LITERAL, FLOAT_LITERAL, STRING_LITERAL, KW_TRUE, KW_NO, KW_EMPTY, KW_NOW, DOLLAR_PARAM] and \
               self.current_token.type not in self.get_binary_operator_tokens() + [KW_IS, KW_ERR, KW_UNLESS, NEWLINE, EOF, DEDENT, SYM_QUESTION, SYM_EXCLAMATION, SYM_RPAREN, SYM_COMMA, SYM_COLON]:
                return True

        # If callee_expr is some other callable expression, e.g. (get_func()) arg1 ...
        # This typically requires parentheses for the arguments: (get_func())(arg1)
        if self.current_token.type == SYM_LPAREN and not isinstance(callee_expr, (IdentifierNode, PropertyAccessNode)): # Avoid double-counting if already handled
             # If callee_expr is complex, LPAREN usually means a call.
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
            raise ParseError(f"Unexpected token {token.type} ('{token.value}') when expecting a primary expression.", token)

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

        # Lookahead for method indicators:
        # 1. `IDENTIFIER (` -> params, definitely method
        # 2. `IDENTIFIER :` -> type annotation or start of expression body (not in EBNF for method body start, but could be param type)
        #    If `IDENTIFIER : type_expr` for params, then method.
        #    If `IDENTIFIER : expr NL` for body -> method (not in current EBNF, but possible lang feature)
        # 3. `IDENTIFIER NL INDENT` -> block body, method with no params
        # 4. `IDENTIFIER expr NL` -> single expr body, method with no params

        # Default to field if none of the above strong method indicators are met.
        is_method = False
        if self.current_token.type == SYM_LPAREN: # Case 1
            is_method = True
        elif self.current_token.type == NEWLINE and self.peek().type == INDENT: # Case 3
            is_method = True
        elif self.current_token.type not in [SYM_QUESTION, NEWLINE, EOF, DEDENT]:
            # This is potentially `IDENTIFIER expr NL` (method) vs `IDENTIFIER ? expr NL` (field) or `IDENTIFIER NL` (field)
            # If it's not '?' or NEWLINE, it might be the start of an expression for a single-line method body.
            # This is the most ambiguous. A field is simply `IDENTIFIER NEWLINE` or `IDENTIFIER ? expr NEWLINE`.
            # If current token can start an expression and is not '?', it's likely method.
            # This heuristic is imperfect. e.g. `my_var some_other_identifier` could be start of method `my_var some_other_identifier NL`
            # or `my_var` is a field, and `some_other_identifier` starts next statement (if newline was missing).
            # Given `function_body = (expression, NEWLINE)`, if we see an expression starting token, assume method.
            if self.current_token.type not in [KW_IS, KW_ERR, KW_UNLESS, SYM_DOT, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_QUESTION, SYM_EXCLAMATION] + list(self.get_binary_operator_tokens()):
                 # Check if the expression is followed by a NEWLINE, which is characteristic of a single-expression method body.
                 # This requires more lookahead or a trial parse.
                 # For now, if it's not '?' and not directly a NEWLINE, assume it *could* be a method's expression body.
                 # This part needs to be cautious not to misinterpret a field followed by a new statement on the same line (if grammar allowed).
                 # However, Coral is newline-sensitive for statements.
                 is_method = True # Tentative: if it looks like an expression, assume method body.

        if is_method:
            params = []
            if self.consume_if(SYM_LPAREN): # Handles `IDENTIFIER (`
                if self.current_token.type != SYM_RPAREN:
                    params.append(self.parse_parameter_definition())
                    while self.consume_if(SYM_COMMA):
                        if self.current_token.type == SYM_RPAREN: break
                        params.append(self.parse_parameter_definition())
                self.consume(SYM_RPAREN, "Expected ')' after method parameters.")

            body_node = self.parse_function_body_content(member_name_node.name)
            return MethodDefinitionNode(name=member_name_node, params=params, body=body_node, location_info=member_loc)
        else: # Field
            default_value = None
            if self.consume_if(SYM_QUESTION):
                default_value = self.parse_expression()
            self.consume(NEWLINE, f"Expected newline after field definition for '{member_name_node.name}'.")
            return FieldDefinitionNode(name=member_name_node, default_value=default_value, location_info=member_loc)


    def parse_function_body_content(self, func_name_for_error="function"):
        body_node = None
        # EBNF: ( expression , NEWLINE )
        #     | ( NEWLINE , INDENT , { statement } , [ [ 'return' ] , expression , NEWLINE ] , DEDENT )
        #     | ( NEWLINE , INDENT , { statement } , DEDENT )

        if self.current_token.type == NEWLINE and self.peek().type == INDENT:
            self.consume(NEWLINE)
            self.consume(INDENT)
            body_statements = []
            # Parse statements
            while self.current_token.type != DEDENT and self.current_token.type != EOF:
                # Check for optional final return part: [ [ 'return' ] , expression , NEWLINE ]
                # This check should be done if the *next* token after this potential part is DEDENT.
                if self.peek_is_dedent_after_possible_return():
                    # If current is KW_RETURN, or can start an expression AND is followed by NEWLINE then DEDENT
                    is_explicit_return = self.current_token.type == KW_RETURN
                    # Try to parse `[['return'] expression NEWLINE]`
                    # This is complex. For now, rely on parsing a ReturnStatement or an ExpressionStatement.
                    # The EBNF implies that `expression NEWLINE` at the end of a block acts as a return.
                    # This can be handled by the last statement in `body_statements` if it's an ExpressionStatement.
                    # No special parsing here, semantic analysis or codegen can interpret last ExpressionStatement.
                    pass # Fall through to parse_statement

                body_statements.append(self.parse_statement())

            # After loop, if last statement was an expression that should be an implicit return,
            # it's already in body_statements.
            self.consume(DEDENT, f"Expected DEDENT at end of block for {func_name_for_error}")
            body_node = body_statements
        else: # Single expression body: expression , NEWLINE
            expr_loc = (self.current_token.line, self.current_token.column)
            body_expr = self.parse_expression()
            self.expect_newline_or_eof(f"Expected newline or EOF after single expression body for {func_name_for_error}.")
            body_node = body_expr

        return body_node

    def peek_is_dedent_after_possible_return(self):
        # Helper to check if `DEDENT` follows a potential `[['return'] expression NEWLINE]`
        # This is a rough check.
        pos_after_expr_and_newline = 0
        if self.current_token.type == KW_RETURN:
            #Approximate length of 'return' + ' ' + minimal_expr + NEWLINE
            pos_after_expr_and_newline = self.pos + 3
        elif self.current_token.type not in [NEWLINE, EOF, DEDENT]: # Potential start of expression
            #Approximate length of minimal_expr + NEWLINE
            pos_after_expr_and_newline = self.pos + 2
        else:
            return False

        if pos_after_expr_and_newline < len(self.tokens) -1 : # Need at least one token for DEDENT
             #This is extremely simplified. A real lookahead would parse the expression.
             #For now, this heuristic is not robust enough to use reliably.
             #return self.tokens[pos_after_expr_and_newline].type == DEDENT
             pass
        return False # Disable for now

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

        call_style = ""
        arguments = []
        actual_callee = callee_expr # This might change if it's a method call like obj.method

        if self.current_token.type == SYM_DOT and (self.peek().type == IDENTIFIER or self.peek().type == KW_ACROSS):
            dot_loc = (self.current_token.line, self.current_token.column)
            self.consume(SYM_DOT)
            method_or_across_token = self.current_token

            prop_name_val = method_or_across_token.value
            prop_loc = (method_or_across_token.line, method_or_across_token.column)
            self.advance()

            prop_ident_node = IdentifierNode(prop_name_val, location_info=prop_loc)
            actual_callee = PropertyAccessNode(base_expr=callee_expr, property_name=prop_ident_node, location_info=dot_loc)
            current_call_style = "dot_method" if method_or_across_token.type == IDENTIFIER else "dot_across"

            if self.current_token.type == SYM_LPAREN:
                arguments, _ = self.parse_paren_argument_list()
                call_style = current_call_style # Retain dot_method/dot_across
            elif self.current_token.type != NEWLINE and self.current_token.type != EOF and \
                 self.current_token.type not in [KW_IS, KW_ERR, KW_UNLESS, SYM_QUESTION, SYM_EXCLAMATION, SYM_COMMA, SYM_RPAREN, SYM_COLON, DEDENT] + self.get_binary_operator_tokens():
                arguments, _ = self.parse_no_paren_space_separated_argument_list()
                call_style = current_call_style # Retain dot_method/dot_across
            else:
                arguments = []
                call_style = current_call_style + "_empty" # e.g. "dot_method_empty"

        else: # Direct call (no dot for this specific call segment)
            if self.current_token.type == SYM_LPAREN:
                arguments, _ = self.parse_paren_argument_list()
                call_style = "paren_direct"
            # Check for no_paren_space_separated_argument_list (typically for IDENTIFIER callee)
            elif isinstance(actual_callee, IdentifierNode) and \
                 self.current_token.type != NEWLINE and self.current_token.type != EOF and \
                 self.current_token.type not in [KW_IS, KW_ERR, KW_UNLESS, SYM_QUESTION, SYM_EXCLAMATION, SYM_COMMA, SYM_RPAREN, SYM_COLON, DEDENT] + self.get_binary_operator_tokens():
                arguments, _ = self.parse_no_paren_space_separated_argument_list()
                call_style = "no_paren_identifier"
            # Check for no_paren_comma_separated_argument_list (typically for complex callee like (expr))
            elif not isinstance(actual_callee, IdentifierNode) and \
                 self.current_token.type != NEWLINE and self.current_token.type != EOF and \
                 self.current_token.type not in [KW_IS, KW_ERR, KW_UNLESS, SYM_QUESTION, SYM_EXCLAMATION, SYM_LPAREN, SYM_RPAREN, SYM_COLON, DEDENT] + self.get_binary_operator_tokens() and \
                 self.peek_has_comma_before_terminator(): # Heuristic: if first arg is followed by comma
                arguments, _ = self.parse_no_paren_comma_separated_argument_list()
                call_style = "no_paren_comma_direct"
            else:
                arguments = []
                call_style = "empty_direct"

        return CallOperationNode(callee=actual_callee, arguments=arguments, call_style=call_style, location_info=call_loc)

    def peek_has_comma_before_terminator(self):
        # Simple lookahead to see if a comma appears before a line break or other argument list terminator.
        # This helps distinguish `(func) arg1, arg2` from `(func) arg1` then something else.
        # This is a simplified heuristic.
        for i in range(1, 5): # Look ahead a few tokens
            pk = self.peek(i)
            if pk.type == SYM_COMMA: return True
            if pk.type in [NEWLINE, EOF, DEDENT, SYM_LPAREN, SYM_RPAREN]: return False
        return False

    def parse_no_paren_comma_separated_argument_list(self):
        # no_paren_comma_separated_argument_list = expression_for_no_paren_call , { ',' , expression_for_no_paren_call } ;
        args = []
        if self.current_token.type not in [NEWLINE, EOF, DEDENT, KW_IS, KW_ERR, KW_UNLESS, SYM_DOT, SYM_LPAREN, SYM_RPAREN, SYM_COMMA, SYM_COLON, SYM_QUESTION, SYM_EXCLAMATION] + list(self.get_binary_operator_tokens()):
            args.append(self.parse_expression_for_no_paren_call())
            while self.consume_if(SYM_COMMA):
                if self.current_token.type in [NEWLINE, EOF, DEDENT]: break # Stop if comma is trailing before newline etc.
                args.append(self.parse_expression_for_no_paren_call())
        return args, "no_paren_comma_args"

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
        # EBNF: primary_expression_base | additive_expression
        # This means it will not parse comparison, logical_or, logical_and if they are compound.
        # Try parsing additive_expression. If it fails, try primary_expression_base.
        # For now, simplified to additive_expression.
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
            stmt = self.parse_statement()
            # Ensure that single statement is properly terminated if it's not a block
            self.expect_newline_or_eof("Single statement in if/else/loop must be followed by newline or EOF.")
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
