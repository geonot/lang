import unittest
from lexer import Lexer, Token # Token might be needed for more complex AST checks if locations are verified
from parser import Parser, ParseError
from ast_nodes import * # Import all AST nodes

# Helper function to simplify AST generation for tests
def make_ast(code_string: str):
    lexer = Lexer(code_string)
    tokens = list(lexer.tokens())
    parser = Parser(tokens)
    # print("\nTokens for:", code_string.strip())
    # for t in tokens: print(repr(t))
    # print("-" * 10)
    return parser.parse_program()

class TestParserIntegration(unittest.TestCase):

    def test_empty_program(self):
        ast = make_ast("")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 0) # Empty input might result in one EOF, so ProgramNode might be empty or have one EOF marker. Parser current makes it empty.

    def test_parse_empty_statement(self):
        ast = make_ast("\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertTrue(len(ast.body) >= 1, "AST body should not be empty for a newline.")
        self.assertIsInstance(ast.body[0], EmptyStatementNode)

    def test_parse_multiple_empty_statements(self):
        ast = make_ast("\n\n\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 3)
        for stmt in ast.body:
            self.assertIsInstance(stmt, EmptyStatementNode)

    def test_parse_simple_assignment(self):
        ast = make_ast("x is 10\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 1)

        stmt = ast.body[0]
        self.assertIsInstance(stmt, AssignmentNode)

        self.assertIsInstance(stmt.target, IdentifierNode)
        self.assertEqual(stmt.target.name, "x")

        self.assertIsInstance(stmt.value, LiteralNode)
        self.assertEqual(stmt.value.value, 10)
        self.assertEqual(stmt.value.literal_type, 'INTEGER')

    def test_assignment_to_property_access(self):
        ast = make_ast("obj.prop is 1\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 1)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, AssignmentNode)
        self.assertIsInstance(stmt.target, PropertyAccessNode)
        self.assertIsInstance(stmt.target.base_expr, IdentifierNode)
        self.assertEqual(stmt.target.base_expr.name, "obj")
        self.assertEqual(stmt.target.property_name.name, "prop")
        self.assertIsInstance(stmt.value, LiteralNode)

    def test_assignment_to_list_element_access(self):
        ast = make_ast("my_list(0) is 1\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, AssignmentNode)
        self.assertIsInstance(stmt.target, ListElementAccessNode)
        self.assertIsInstance(stmt.target.base_expr, IdentifierNode)
        self.assertEqual(stmt.target.base_expr.name, "my_list")
        self.assertIsInstance(stmt.target.index_expr, LiteralNode) # Index is an expression
        self.assertEqual(stmt.target.index_expr.value, 0)
        self.assertIsInstance(stmt.value, LiteralNode)

    def test_map_block_assignment(self):
        code = "config is\n  host is 'localhost'\n  port is 80\n"
        ast = make_ast(code)
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, AssignmentNode)
        self.assertIsInstance(stmt.target, IdentifierNode)
        self.assertEqual(stmt.target.name, "config")
        self.assertIsInstance(stmt.value, MapBlockAssignmentRHSNode)
        self.assertEqual(len(stmt.value.entries), 2)
        self.assertIsInstance(stmt.value.entries[0], MapBlockEntryNode)
        self.assertEqual(stmt.value.entries[0].key.name, "host")
        self.assertEqual(stmt.value.entries[0].value.value, "localhost")
        self.assertEqual(stmt.value.entries[1].key.name, "port")
        self.assertEqual(stmt.value.entries[1].value.value, 80)


    def test_parse_string_assignment(self):
        ast = make_ast("msg is \"hello world\"\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, AssignmentNode)
        self.assertIsInstance(stmt.target, IdentifierNode)
        self.assertEqual(stmt.target.name, "msg")
        self.assertIsInstance(stmt.value, LiteralNode)
        self.assertEqual(stmt.value.value, "hello world")
        self.assertEqual(stmt.value.literal_type, 'STRING')

    # --- Expression Statement Tests ---
    def _check_expr_stmt(self, code, expected_expr_type):
        ast = make_ast(code)
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 1)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, ExpressionStatementNode)
        self.assertIsInstance(stmt.expression, FullExpressionNode)
        self.assertIsInstance(stmt.expression.expression, expected_expr_type)
        return stmt.expression.expression # Return the actual expression node for further checks

    def test_list_literal_statement(self):
        expr = self._check_expr_stmt("(1,2,3)\n", ListLiteralNode)
        self.assertEqual(len(expr.elements), 3)
        self.assertEqual(expr.elements[0].value, 1)

    def test_map_literal_statement(self):
        expr = self._check_expr_stmt("(a:1, b:2)\n", MapLiteralNode)
        self.assertEqual(len(expr.entries), 2)
        self.assertEqual(expr.entries[0].key.name, "a")
        self.assertEqual(expr.entries[0].value.value, 1)

    def test_binary_op_statement(self):
        expr = self._check_expr_stmt("1 + 2 * 3\n", BinaryOpNode) # Should parse as 1 + (2 * 3)
        self.assertEqual(expr.operator, "+")
        self.assertIsInstance(expr.left, LiteralNode)
        self.assertEqual(expr.left.value, 1)
        self.assertIsInstance(expr.right, BinaryOpNode) # 2 * 3
        self.assertEqual(expr.right.operator, "*")
        self.assertEqual(expr.right.left.value, 2)
        self.assertEqual(expr.right.right.value, 3)

    def test_unary_op_statement(self):
        expr = self._check_expr_stmt("not true\n", UnaryOpNode)
        self.assertEqual(expr.operator, "not")
        self.assertIsInstance(expr.operand, LiteralNode)
        self.assertEqual(expr.operand.value, True)

    def test_ternary_op_statement(self):
        expr = self._check_expr_stmt("a ? b ! c\n", TernaryConditionalExpressionNode)
        self.assertIsInstance(expr.condition, IdentifierNode)
        self.assertEqual(expr.condition.name, "a")
        self.assertIsInstance(expr.true_expr, IdentifierNode)
        self.assertEqual(expr.true_expr.name, "b")
        self.assertIsInstance(expr.false_expr, IdentifierNode)
        self.assertEqual(expr.false_expr.name, "c")

    def test_property_access_statement(self):
        expr = self._check_expr_stmt("a.b.c\n", PropertyAccessNode)
        self.assertEqual(expr.property_name.name, "c")
        self.assertIsInstance(expr.base_expr, PropertyAccessNode)
        self.assertEqual(expr.base_expr.property_name.name, "b")
        self.assertIsInstance(expr.base_expr.base_expr, IdentifierNode)
        self.assertEqual(expr.base_expr.base_expr.name, "a")

    def test_list_access_statement(self):
        expr = self._check_expr_stmt("a(0)(1)\n", ListElementAccessNode)
        self.assertIsInstance(expr.index_expr, LiteralNode)
        self.assertEqual(expr.index_expr.value, 1)
        self.assertIsInstance(expr.base_expr, ListElementAccessNode)
        self.assertEqual(expr.base_expr.index_expr.value, 0)
        self.assertIsInstance(expr.base_expr.base_expr, IdentifierNode)
        self.assertEqual(expr.base_expr.base_expr.name, "a")

    def test_call_statement(self):
        expr = self._check_expr_stmt("my_func(arg1, name:arg2)\n", CallOperationNode)
        self.assertIsInstance(expr.callee, IdentifierNode)
        self.assertEqual(expr.callee.name, "my_func")
        self.assertEqual(len(expr.arguments), 2)
        self.assertIsInstance(expr.arguments[0], ArgumentNode)
        self.assertIsNone(expr.arguments[0].name) # Positional
        self.assertIsInstance(expr.arguments[0].value, IdentifierNode)
        self.assertEqual(expr.arguments[0].value.name, "arg1")
        self.assertIsInstance(expr.arguments[1], ArgumentNode)
        self.assertIsNotNone(expr.arguments[1].name)
        self.assertEqual(expr.arguments[1].name.name, "name")
        self.assertIsInstance(expr.arguments[1].value, IdentifierNode)
        self.assertEqual(expr.arguments[1].value.name, "arg2")

    # --- Conditional Statement Tests ---
    def test_if_no_else(self):
        ast = make_ast("if x\n  y is 1\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, IfThenElseStatementNode)
        self.assertIsInstance(stmt.condition, IdentifierNode)
        self.assertEqual(stmt.condition.name, "x")
        self.assertIsInstance(stmt.if_block, list) # Block
        self.assertIsInstance(stmt.if_block[0], AssignmentNode)
        self.assertEqual(len(stmt.else_if_clauses), 0)
        self.assertIsNone(stmt.else_block)

    def test_if_elif_else(self):
        ast = make_ast("if x\n  a is 1\nelif y\n  b is 1\nelse\n  c is 1\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, IfThenElseStatementNode)
        self.assertEqual(stmt.condition.name, "x")
        self.assertEqual(len(stmt.else_if_clauses), 1)
        self.assertEqual(stmt.else_if_clauses[0]['condition'].name, "y")
        self.assertIsInstance(stmt.else_if_clauses[0]['block'], list)
        self.assertIsNotNone(stmt.else_block)
        self.assertIsInstance(stmt.else_block, list)

    def test_unless_prefix(self):
        ast = make_ast("unless x\n  y is 1\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, UnlessStatementNode)
        self.assertEqual(stmt.condition.name, "x")
        self.assertIsInstance(stmt.block, list)

    def test_unless_postfix(self):
        ast = make_ast("y is 1 unless x\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, PostfixUnlessStatementNode)
        self.assertEqual(stmt.condition.name, "x")
        self.assertIsInstance(stmt.expression_statement, ExpressionStatementNode)
        self.assertIsInstance(stmt.expression_statement.expression.expression, AssignmentNode) # y is 1

    # --- Loop Statement Tests ---
    def test_while_loop(self):
        ast = make_ast("while i < 10\n  i is i + 1\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, WhileLoopNode)
        self.assertIsInstance(stmt.condition, BinaryOpNode)
        self.assertEqual(stmt.condition.operator, "<")
        self.assertIsInstance(stmt.body, list)
        self.assertIsInstance(stmt.body[0], AssignmentNode)

    def test_until_loop(self):
        ast = make_ast("until i >= 10\n  i is i + 1\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, UntilLoopNode)
        self.assertIsInstance(stmt.condition, BinaryOpNode)
        self.assertEqual(stmt.condition.operator, ">=")
        self.assertIsInstance(stmt.body, list)

    def test_iterate_loop(self):
        ast = make_ast("iterate my_list (item)\n  print item\n") # Assuming print is a func call
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, IterateLoopNode)
        self.assertIsInstance(stmt.iterable, IdentifierNode)
        self.assertEqual(stmt.iterable.name, "my_list")
        self.assertIsNotNone(stmt.loop_variable)
        self.assertEqual(stmt.loop_variable.name, "item")
        self.assertIsInstance(stmt.body, list)
        call_stmt = stmt.body[0].expression.expression # ExpressionStatement -> FullExpressionNode -> CallOp
        self.assertIsInstance(call_stmt, CallOperationNode)
        self.assertEqual(call_stmt.callee.name, "print")

    def test_iterate_loop_no_var(self):
        ast = make_ast("iterate my_list\n  do_something()\n")
        self.assertIsInstance(ast, ProgramNode)
        stmt = ast.body[0]
        self.assertIsInstance(stmt, IterateLoopNode)
        self.assertIsInstance(stmt.iterable, IdentifierNode)
        self.assertEqual(stmt.iterable.name, "my_list")
        self.assertIsNone(stmt.loop_variable)
        self.assertIsInstance(stmt.body, list)
        self.assertIsInstance(stmt.body[0].expression.expression, CallOperationNode)

    # --- Definition Tests ---

    def test_parse_simple_function_def_expr_body(self):
        # Current EBNF: function_body = ( expression , NEWLINE ) | ( NEWLINE , INDENT ... DEDENT )
        # Test case for: fn my_func\n  1\n (block body with single expression)
        ast = make_ast("fn my_func\n  1\n")
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 1)

        func_def_node = ast.body[0]
        self.assertIsInstance(func_def_node, FunctionDefinitionNode)
        self.assertEqual(func_def_node.name.name, "my_func")
        self.assertEqual(len(func_def_node.params), 0)

        self.assertIsInstance(func_def_node.body, list, "Block body should be a list of statements")
        self.assertEqual(len(func_def_node.body), 1)
        expr_stmt = func_def_node.body[0] # ExpressionStatement
        self.assertIsInstance(expr_stmt, ExpressionStatementNode)
        self.assertIsInstance(expr_stmt.expression.expression, LiteralNode)
        self.assertEqual(expr_stmt.expression.expression.value, 1)

    def test_func_def_single_expr_body_same_line_style(self):
        # This tests `function_body = ( expression , NEWLINE )`
        # This requires the parser to handle `fn name expr\n` if no keyword like `->` is used.
        # Current parser expects block or explicit return for body if not single line.
        # Let's test `fn f(x) x*x\n` which is ambiguous without clear separator.
        # The parser's `parse_function_body_content` handles `expr NL` or `NL INDENT ...`.
        # So `fn f(x) x*x\n` should be parsed as `x*x` being the expression body.
        ast = make_ast("fn multiply(x, y) x * y\n")
        self.assertIsInstance(ast, ProgramNode)
        func_def = ast.body[0]
        self.assertIsInstance(func_def, FunctionDefinitionNode)
        self.assertEqual(func_def.name.name, "multiply")
        self.assertEqual(len(func_def.params), 2)
        self.assertIsInstance(func_def.body, BinaryOpNode) # Direct expression body
        self.assertEqual(func_def.body.operator, "*")


    def test_func_def_no_params_no_body_block(self):
        # An empty block is NEWLINE INDENT DEDENT. Parser creates EmptyStatement for the NEWLINE.
        # Then INDENT, then DEDENT. So body list should be empty.
        ast = make_ast("fn f\n  \n") # This is effectively `fn f \n INDENT NEWLINE DEDENT`
        self.assertIsInstance(ast, ProgramNode)
        func_def = ast.body[0]
        self.assertIsInstance(func_def, FunctionDefinitionNode)
        self.assertEqual(func_def.name.name, "f")
        self.assertIsInstance(func_def.body, list)
        # The body has one EmptyStatement due to the newline inside the block
        self.assertEqual(len(func_def.body), 1)
        self.assertIsInstance(func_def.body[0], EmptyStatementNode)


    def test_func_def_params_default_values(self):
        ast = make_ast("fn f(a, b:1, c:'s')\n  return a + b\n")
        self.assertIsInstance(ast, ProgramNode)
        func_def = ast.body[0]
        self.assertIsInstance(func_def, FunctionDefinitionNode)
        self.assertEqual(func_def.name.name, "f")
        self.assertEqual(len(func_def.params), 3)
        self.assertEqual(func_def.params[0].name.name, "a")
        self.assertIsNone(func_def.params[0].default_value)
        self.assertEqual(func_def.params[1].name.name, "b")
        self.assertIsInstance(func_def.params[1].default_value, LiteralNode)
        self.assertEqual(func_def.params[1].default_value.value, 1)
        self.assertEqual(func_def.params[2].name.name, "c")
        self.assertIsInstance(func_def.params[2].default_value, LiteralNode)
        self.assertEqual(func_def.params[2].default_value.value, "s")

    def test_object_def_empty(self):
        ast = make_ast("object MyObj\n  \n")
        self.assertIsInstance(ast, ProgramNode)
        obj_def = ast.body[0]
        self.assertIsInstance(obj_def, ObjectDefinitionNode)
        self.assertEqual(obj_def.name.name, "MyObj")
        self.assertEqual(len(obj_def.members), 1) # The newline inside makes an EmptyStatement
        self.assertIsInstance(obj_def.members[0], EmptyStatementNode)


    def test_object_def_with_fields(self):
        ast = make_ast("object User\n  name\n  age?0\n")
        self.assertIsInstance(ast, ProgramNode)
        obj_def = ast.body[0]
        self.assertIsInstance(obj_def, ObjectDefinitionNode)
        self.assertEqual(obj_def.name.name, "User")
        self.assertEqual(len(obj_def.members), 2)
        self.assertIsInstance(obj_def.members[0], FieldDefinitionNode)
        self.assertEqual(obj_def.members[0].name.name, "name")
        self.assertIsNone(obj_def.members[0].default_value)
        self.assertIsInstance(obj_def.members[1], FieldDefinitionNode)
        self.assertEqual(obj_def.members[1].name.name, "age")
        self.assertIsInstance(obj_def.members[1].default_value, LiteralNode)
        self.assertEqual(obj_def.members[1].default_value.value, 0)

    def test_object_def_with_methods(self):
        ast = make_ast("object Calc\n  add(a,b)\n    return a+b\n  get_val\n    10\n")
        self.assertIsInstance(ast, ProgramNode)
        obj_def = ast.body[0]
        self.assertIsInstance(obj_def, ObjectDefinitionNode)
        self.assertEqual(obj_def.name.name, "Calc")
        self.assertEqual(len(obj_def.members), 2)

        method_add = obj_def.members[0]
        self.assertIsInstance(method_add, MethodDefinitionNode)
        self.assertEqual(method_add.name.name, "add")
        self.assertEqual(len(method_add.params), 2)
        self.assertIsInstance(method_add.body, list) # Block body
        self.assertIsInstance(method_add.body[0], ReturnStatementNode)

        method_get_val = obj_def.members[1]
        self.assertIsInstance(method_get_val, MethodDefinitionNode)
        self.assertEqual(method_get_val.name.name, "get_val")
        self.assertEqual(len(method_get_val.params), 0)
        self.assertIsInstance(method_get_val.body, list) # Block body with single expression
        self.assertIsInstance(method_get_val.body[0].expression.expression, LiteralNode)


    def test_store_def_simple(self):
        ast = make_ast("store Item\n  name\n")
        self.assertIsInstance(ast, ProgramNode)
        store_def = ast.body[0]
        self.assertIsInstance(store_def, StoreDefinitionNode)
        self.assertEqual(store_def.name.name, "Item")
        self.assertFalse(store_def.is_actor)
        self.assertIsNone(store_def.for_target)
        self.assertEqual(len(store_def.members), 1)
        self.assertIsInstance(store_def.members[0], FieldDefinitionNode) # 'name' is a field

    def test_store_def_actor_for(self):
        ast = make_ast("store actor User for Account\n  name\n")
        self.assertIsInstance(ast, ProgramNode)
        store_def = ast.body[0]
        self.assertIsInstance(store_def, StoreDefinitionNode)
        self.assertEqual(store_def.name.name, "User")
        self.assertTrue(store_def.is_actor)
        self.assertIsNotNone(store_def.for_target)
        self.assertEqual(store_def.for_target.name, "Account")

    def test_store_def_with_relation(self):
        ast = make_ast("store Post\n  &comments\n")
        self.assertIsInstance(ast, ProgramNode)
        store_def = ast.body[0]
        self.assertIsInstance(store_def, StoreDefinitionNode)
        self.assertEqual(len(store_def.members), 1)
        self.assertIsInstance(store_def.members[0], RelationDefinitionNode)
        self.assertEqual(store_def.members[0].name.name, "comments")

    def test_store_def_with_cast_expr_body(self):
        ast = make_ast("store Data\n  as string name + ' ' + value\n")
        self.assertIsInstance(ast, ProgramNode)
        store_def = ast.body[0]
        self.assertIsInstance(store_def, StoreDefinitionNode)
        cast_def = store_def.members[0]
        self.assertIsInstance(cast_def, CastDefinitionNode)
        self.assertEqual(cast_def.cast_to_type, "string")
        self.assertIsInstance(cast_def.body, BinaryOpNode) # name + ' ' + value

    def test_store_def_with_cast_block_body(self):
        ast = make_ast("store Info\n  as map\n    id is self.id\n    val is self.value\n")
        self.assertIsInstance(ast, ProgramNode)
        store_def = ast.body[0]
        cast_def = store_def.members[0]
        self.assertIsInstance(cast_def, CastDefinitionNode)
        self.assertEqual(cast_def.cast_to_type, "map")
        self.assertIsInstance(cast_def.body, list)
        self.assertEqual(len(cast_def.body), 2)
        self.assertIsInstance(cast_def.body[0], AssignmentNode) # id is self.id

    def test_store_def_with_receive(self):
        ast = make_ast("store Actor\n  @handle_this(msg)\n    process msg\n")
        self.assertIsInstance(ast, ProgramNode)
        store_def = ast.body[0]
        recv_handler = store_def.members[0]
        self.assertIsInstance(recv_handler, ReceiveHandlerNode)
        self.assertEqual(recv_handler.message_name.name, "handle_this")
        self.assertEqual(len(recv_handler.body[0].expression.expression.arguments),1) # process(msg) -> CallOp with 1 arg

    def test_module_def(self):
        ast = make_ast("mod MyMod\n  fn helper\n    1\n")
        self.assertIsInstance(ast, ProgramNode)
        mod_def = ast.body[0]
        self.assertIsInstance(mod_def, ModuleDefinitionNode)
        self.assertEqual(mod_def.name.name, "MyMod")
        self.assertEqual(len(mod_def.body), 1)
        self.assertIsInstance(mod_def.body[0], FunctionDefinitionNode)

    # --- Error Handler Suffix Tests ---
    def test_expr_with_err_return_val(self):
        ast = make_ast("do_something() err return -1\n")
        expr_stmt = ast.body[0].expression # FullExpressionNode
        self.assertIsNotNone(expr_stmt.error_handler)
        self.assertIsInstance(expr_stmt.error_handler.action, ReturnStatementNode)
        self.assertEqual(expr_stmt.error_handler.action.value.operand.value, 1) # UnaryOp '-'

    def test_expr_with_err_return_no_val(self):
        ast = make_ast("do_something() err return\n")
        expr_stmt = ast.body[0].expression
        self.assertIsNotNone(expr_stmt.error_handler)
        self.assertIsInstance(expr_stmt.error_handler.action, ReturnStatementNode)
        self.assertIsNone(expr_stmt.error_handler.action.value)

    def test_expr_with_err_expr_handler(self):
        ast = make_ast("do_something() err handle_error()\n")
        expr_stmt = ast.body[0].expression
        self.assertIsNotNone(expr_stmt.error_handler)
        self.assertIsInstance(expr_stmt.error_handler.action, CallOperationNode)
        self.assertEqual(expr_stmt.error_handler.action.callee.name, "handle_error")

    def test_expr_with_err_block_handler(self):
        ast = make_ast("do_something() err\n  log 'error'\n  return false\n")
        expr_stmt = ast.body[0].expression
        self.assertIsNotNone(expr_stmt.error_handler)
        self.assertIsInstance(expr_stmt.error_handler.action, list) # Block of statements
        self.assertEqual(len(expr_stmt.error_handler.action), 2)
        self.assertIsInstance(expr_stmt.error_handler.action[0], ExpressionStatementNode) # log 'error'
        self.assertIsInstance(expr_stmt.error_handler.action[1], ReturnStatementNode) # return false


    def test_parse_function_def_with_params_and_return(self):
        code = "fn add(a: int, b: int)\n  return a + b\n" # Type hints not in EBNF yet
        code = "fn add(a, b)\n  return a + b\n"
        ast = make_ast(code)
        self.assertIsInstance(ast, ProgramNode)
        func_def = ast.body[0]
        self.assertIsInstance(func_def, FunctionDefinitionNode)
        self.assertEqual(func_def.name.name, "add")

        self.assertEqual(len(func_def.params), 2)
        self.assertIsInstance(func_def.params[0], ParameterNode)
        self.assertEqual(func_def.params[0].name.name, "a")
        self.assertIsNone(func_def.params[0].default_value)
        self.assertIsInstance(func_def.params[1], ParameterNode)
        self.assertEqual(func_def.params[1].name.name, "b")

        self.assertIsInstance(func_def.body, list) # Block body
        return_stmt = func_def.body[0]
        self.assertIsInstance(return_stmt, ReturnStatementNode)
        self.assertIsInstance(return_stmt.value, BinaryOpNode)
        self.assertEqual(return_stmt.value.operator, "+")

    def test_parse_error_invalid_token_sequence(self):
        # Example: an operator where an identifier is expected in definition
        with self.assertRaises(ParseError):
            make_ast("fn +\n  return 1\n")

    def test_parse_error_unexpected_dedent(self):
        code = "fn my_func\nreturn 1\n" # Missing INDENT
        with self.assertRaises(ParseError):
            make_ast(code)

    def test_parse_object_with_field_and_method(self):
        code = """
object Counter
  value? 0

  fn increment(amount)
    value is value + amount

  fn get_value
    return value
"""
        ast = make_ast(code)
        self.assertIsInstance(ast, ProgramNode)
        self.assertEqual(len(ast.body), 1)

        obj_def = ast.body[0]
        self.assertIsInstance(obj_def, ObjectDefinitionNode)
        self.assertEqual(obj_def.name.name, "Counter")

        self.assertEqual(len(obj_def.members), 3) # value, increment, get_value

        field_def = obj_def.members[0]
        self.assertIsInstance(field_def, FieldDefinitionNode)
        self.assertEqual(field_def.name.name, "value")
        self.assertIsInstance(field_def.default_value, LiteralNode)
        self.assertEqual(field_def.default_value.value, 0)

        method_increment = obj_def.members[1]
        self.assertIsInstance(method_increment, MethodDefinitionNode)
        self.assertEqual(method_increment.name.name, "increment")
        self.assertEqual(len(method_increment.params), 1)
        self.assertEqual(method_increment.params[0].name.name, "amount")

        method_get_value = obj_def.members[2]
        self.assertIsInstance(method_get_value, MethodDefinitionNode)
        self.assertEqual(method_get_value.name.name, "get_value")
        self.assertEqual(len(method_get_value.params), 0)
        self.assertIsInstance(method_get_value.body, list)
        self.assertIsInstance(method_get_value.body[0], ReturnStatementNode)

    def test_if_else_statement(self):
        code = """
if x > 10
  y is 1
else if x > 5
  y is 2
else
  y is 3
"""
        ast = make_ast(code)
        self.assertIsInstance(ast, ProgramNode)
        if_stmt_node = ast.body[0]
        self.assertIsInstance(if_stmt_node, IfThenElseStatementNode)
        self.assertIsInstance(if_stmt_node.condition, BinaryOpNode)
        self.assertEqual(if_stmt_node.condition.operator, ">")

        self.assertIsInstance(if_stmt_node.if_block, list)
        self.assertIsInstance(if_stmt_node.if_block[0], AssignmentNode)

        self.assertEqual(len(if_stmt_node.else_if_clauses), 1)
        self.assertIsInstance(if_stmt_node.else_if_clauses[0]['condition'], BinaryOpNode)

        self.assertIsInstance(if_stmt_node.else_block, list)
        self.assertIsInstance(if_stmt_node.else_block[0], AssignmentNode)

    # --- More Syntax Error Tests ---
    def test_error_missing_colon_map_entry(self):
        with self.assertRaisesRegex(ParseError, "Expected ':' in map entry."):
             make_ast("(a 1)\n")

    def test_error_unbalanced_parens(self):
        with self.assertRaisesRegex(ParseError, "Expected '\\)' after grouped expression."): # Or specific to list/map
             make_ast("((a+b)\n")

    def test_error_incomplete_if(self):
        # Parser expects a newline or further tokens after condition.
        # If 'if x' is at EOF, it's an unexpected EOF.
        with self.assertRaisesRegex(ParseError, "Expected newline or end of statement"):
             make_ast("if x")
        # If 'if x\n' then INDENT is missing for block
        with self.assertRaisesRegex(ParseError, "Expected INDENT"):
             make_ast("if x\n")


    def test_error_assignment_to_literal(self):
        with self.assertRaisesRegex(ParseError, "Invalid target for assignment."):
            make_ast("10 is x\n")

    def test_error_invalid_store_member_start(self):
        # Example: '!' is not a valid start for a store member
        with self.assertRaisesRegex(ParseError, "Expected identifier for field or method name."): # Or more specific error if checking for known member starters
            make_ast("store X\n  !invalid\n")


if __name__ == '__main__':
    unittest.main()
