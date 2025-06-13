import unittest
import llvmlite.ir

# Assuming project structure allows these imports directly
# If not, sys.path manipulation might be needed for a test runner
from lexer import Lexer
from parser import Parser
from ir_generator import IRGenerator
import ast_nodes
import ir_runtime_types as rt

class TestIRGenerator(unittest.TestCase):

    def _generate_ir(self, coral_code: str) -> llvmlite.ir.Module:
        """
        Helper method to generate LLVM IR module from Coral code string.
        """
        lexer = Lexer(coral_code)
        tokens = lexer.lex()

        parser = Parser(tokens)
        ast_root = parser.parse_program() # Assuming parse_program is the entry point

        ir_gen = IRGenerator()
        llvm_module = ir_gen.generate(ast_root)
        return llvm_module

    def test_integer_literal_expression_statement(self):
        """
        Tests generation of IR for a simple integer literal expression statement.
        Coral code: "5."
        """
        coral_code = "5."
        llvm_module = self._generate_ir(coral_code)
        ir_string = str(llvm_module)

        # Print IR string for debugging (optional, can be removed later)
        # print("Generated LLVM IR:")
        # print(ir_string)

        # 1. Check for the main function definition
        self.assertIn("define void @main()", ir_string, "Main function definition not found.")

        # 2. Check for the creation of the integer CoralValue
        # This involves:
        #   a. Allocation for the i32 value itself
        #   b. Storing the constant i32 value (5)
        #   c. Calling create_coral_value with TYPE_TAG_INTEGER and a pointer to the i32

        # Check for allocation of the integer value (e.g., %int_val_alloc = alloca i32)
        # The exact name (%int_val_alloc) might vary due to unique name generation.
        self.assertRegex(r"%\w+ = alloca i32", ir_string, "Allocation for i32 not found.")

        # Check for storing the constant 5 into the allocated space
        self.assertIn("store i32 5, i32* %\w+", ir_string, "Storing i32 5 not found.")

        # Check for the GEP to get the type_tag field in CoralValue struct
        # Example: %type_tag_ptr = getelementptr %CoralValue, %CoralValue* %..., i32 0, i32 0
        self.assertRegex(r"%type_tag_ptr\w* = getelementptr inbounds %CoralValue, %CoralValue\* %\w+, i32 0, i32 0", ir_string, "GEP for type_tag not found.")

        # Check for storing the integer type tag
        # Example: store i8 0, i8* %type_tag_ptr... (TYPE_TAG_INTEGER is 0)
        self.assertIn(f"store i8 {rt.TYPE_TAG_INTEGER.value}, i8* %type_tag_ptr", ir_string, "Storing integer type tag not found.")

        # Check for the GEP to get the data_val_ptr field in CoralValue struct
        # Example: %data_val_ptr = getelementptr %CoralValue, %CoralValue* %..., i32 0, i32 1
        self.assertRegex(r"%data_val_ptr\w* = getelementptr inbounds %CoralValue, %CoralValue\* %\w+, i32 0, i32 1", ir_string, "GEP for data_val_ptr not found.")

        # Check for storing the pointer to the i32 (after bitcast to i8*) into data_val_ptr
        # Example: store i8* %casted_data_ptr..., i8** %data_val_ptr...
        # This involves a bitcast from i32* to i8*
        self.assertRegex(r"%\w+ = bitcast i32\* %\w+ to i8\*", ir_string, "Bitcast from i32* to i8* not found.")
        self.assertRegex(r"store i8\* %\w+, i8\*\* %\w+", ir_string, "Storing data pointer (to i32) not found.")

        # 3. Check that the main function ends with ret void
        self.assertIn("ret void", ir_string, "main function does not end with ret void.")

if __name__ == '__main__':
    unittest.main()
