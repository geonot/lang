from lexer import Lexer, Token
from parser import Parser, ParseError
from ast_nodes import ASTNode # For type hint or direct use if needed
import llvmlite.ir
from ir_generator import IRGenerator

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
    # Test script for scaffolding of Objects, Lists, Maps, Stores, and Error Handlers.
    test_coral_code = """
object Point
  x // Field

  y // Field

  fn move(dx, dy) // Method
    // this.x = this.x + dx // Future feature
    return dx // Placeholder to make it a valid function for now

store UserData // Store definition
  name
  age

  @on_create // Method-like store member
  fn created_log()
    return "UserData created"


// Test Object.make()
p is Point.make()
p // ExpressionStatement to observe 'p'

// Test List Literal (scaffolding)
my_list is (1, 2, "hello") // Elements are ignored by current list literal scaffolding
my_list // ExpressionStatement

// Test Map Literal (scaffolding)
my_map is (name: "Coral", version: 0.1) // Entries are ignored by current map literal scaffolding
my_map // ExpressionStatement

// Test Store (treated as object for .make())
ud is UserData.make()
ud // ExpressionStatement

// Test Error Handler Suffix (scaffolding)
fn process_data(data)
  return data // placeholder, could be an operation that might fail

raw_data is "some_info"
processed_data is process_data(raw_data) err "Error processing data"
processed_data // ExpressionStatement, should be 'raw_data' for now

another_val is 10 / 0 err "Division by zero!" // This would ideally use the error handler
another_val
    """

    print("--- Coral Source Code ---")
    print(test_coral_code)

    print("\n--- Abstract Syntax Tree (AST) ---")
    ast_tree = get_ast(test_coral_code)
    if ast_tree:
        print(repr(ast_tree))

        print("\n--- LLVM Intermediate Representation (IR) ---")
        try:
            ir_gen = IRGenerator()
            module = ir_gen.generate(ast_tree)
            if module:
                print(str(module))
            else:
                print("IR Generation returned None or an empty module.")
        except Exception as e:
            print(f"Error during IR generation: {e}")
            import traceback
            traceback.print_exc() # Print full stack trace for debugging

    else:
        print("AST generation failed, skipping IR generation.")

    print("\n--- End of Test Run ---")
