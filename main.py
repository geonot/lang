from lexer import Lexer, Token
from parser import Parser, ParseError
from ast_nodes import ASTNode # For type hint or direct use if needed

def get_ast(coral_code: str):
    """
    Generates an AST from a string of Coral code.
    Prints a ParseError and returns None if parsing fails.
    """
    try:
        lexer = Lexer(coral_code)
        tokens = list(lexer.tokens())
        # print("Tokens:", [repr(t) for t in tokens]) # Optional: for debugging
        parser = Parser(tokens)
        ast = parser.parse_program()
        return ast
    except ParseError as e:
        print(e)
        return None

if __name__ == "__main__":
    sample_codes = [
        "",
        "// Just a comment\n",
        "x is 10\n",
        "y is \"hello\"\n",
        "fn my_func(a, b: 20)\n  return a + b\n",
        "object Point\n  x\n  y\n  m(self)\n    return self.x\n",
        "store DataStore for Thing\n  &relation_name\n  as map\n    original_id is self.id\n  @on_create\n    log(\"Created!\")\n",
        "x is @ // Invalid token",
        "fn broken\n  x is 1\n y is 2 // Missing dedent",
        "a is (1 + 2) * 3\n",
        "if x > 10\n  print(\"big\")\nelse if x > 5\n  print(\"medium\")\nelse\n  print(\"small\")\n",
        "val unless condition\n",
        "unless condition_is_true\n  do_something()\n",
        "while i < 10\n  i is i + 1\n",
        "iterate my_list (item)\n  print(item)\n"
    ]

    for i, code in enumerate(sample_codes):
        print(f"--- Sample Code {i+1} ---")
        print(code.strip())
        print("--- AST ---")
        ast_tree = get_ast(code)
        if ast_tree:
            print(repr(ast_tree))
        print("--- End ---")
        if i < len(sample_codes) - 1:
            print("\n" + "="*30 + "\n")
