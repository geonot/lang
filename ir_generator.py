import llvmlite.ir
from llvmlite.ir import Constant, FunctionType, VoidType, Function, IRBuilder
import ast_nodes # Assuming ast_nodes.py contains LiteralNode, IdentifierNode, etc.
import ir_runtime_types as rt # rt for runtime_types

class ModuleContext:
    def __init__(self, module_name="my_module"):
        self.module = llvmlite.ir.Module(name=module_name)
        self.function_context_stack = []
        self.global_symbol_table = {}

    def push_function_context(self, llvm_function: Function, symbol_table: dict, builder: IRBuilder):
        self.function_context_stack.append({
            'llvm_function': llvm_function,
            'symbol_table': symbol_table,
            'builder': builder
        })

    def pop_function_context(self) -> dict:
        if not self.function_context_stack:
            print("Warning: Popping from empty function context stack.")
            return None
        return self.function_context_stack.pop()

    def get_current_llvm_function_from_context_stack(self) -> Function | None:
        if not self.function_context_stack:
            return None
        return self.function_context_stack[-1]['llvm_function']

class IRGenerator:
    def __init__(self):
        self.module_context = ModuleContext()
        self.module = self.module_context.module
        self.current_builder: IRBuilder = None
        self.current_llvm_function: Function = None
        self.current_symbol_table: dict = self.module_context.global_symbol_table

    def _get_variable_storage(self, name: str) -> llvmlite.ir.Value | None:
        if name in self.current_symbol_table:
            return self.current_symbol_table[name]
        if self.current_symbol_table is not self.module_context.global_symbol_table and \
           name in self.module_context.global_symbol_table:
            return self.module_context.global_symbol_table[name]
        return None

    def _set_variable_storage(self, name: str, storage_ptr: llvmlite.ir.Value):
        self.current_symbol_table[name] = storage_ptr

    def _generate_literal(self, node: ast_nodes.LiteralNode) -> llvmlite.ir.Value:
        if self.current_builder is None:
            raise RuntimeError("Builder not initialized.")
        # ... (rest of _generate_literal implementation as before, using self.current_builder)
        literal_type = node.literal_type
        value = node.value
        if literal_type == ast_nodes.LiteralType.INTEGER:
            return rt.create_integer_value(self.current_builder, int(value))
        elif literal_type == ast_nodes.LiteralType.FLOAT:
            return rt.create_float_value(self.current_builder, float(value))
        elif literal_type == ast_nodes.LiteralType.BOOLEAN:
            return rt.create_boolean_value(self.current_builder, value)
        elif literal_type == ast_nodes.LiteralType.STRING:
            return rt.create_string_value(self.current_builder, self.module, str(value))
        elif literal_type == ast_nodes.LiteralType.EMPTY:
            return rt.create_null_value(self.current_builder)
        elif literal_type == ast_nodes.LiteralType.NOW:
            print("Warning: 'now' literal is not fully implemented, returning null.")
            return rt.create_null_value(self.current_builder)
        else:
            raise NotImplementedError(f"Literal type {literal_type} not yet supported.")

    def _generate_identifier(self, node: ast_nodes.IdentifierNode) -> llvmlite.ir.Value:
        if self.current_builder is None:
            raise RuntimeError("Builder not initialized.")
        var_name = node.name
        var_storage_ptr = self._get_variable_storage(var_name)
        if var_storage_ptr is None:
            print(f"Error: Identifier '{var_name}' used before assignment. Returning null value.")
            return rt.create_null_value(self.current_builder)
        return self.current_builder.load(var_storage_ptr, name=f"{var_name}_val_ptr_loaded")

    def _generate_assignment(self, node: ast_nodes.AssignmentNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or function not initialized for assignment.")
        if not isinstance(node.target, ast_nodes.IdentifierNode):
            print(f"Warning: Assignment to target type {type(node.target).__name__} not supported. Skipped.")
            return
        var_name = node.target.name
        rhs_coral_value_ptr = self._generate_expression(node.value)
        if rhs_coral_value_ptr is None:
            print(f"Error: RHS of assignment to '{var_name}' evaluated to None. Skipping.")
            return
        var_storage_ptr = self._get_variable_storage(var_name)
        if var_storage_ptr is None:
            target_func_entry_builder = IRBuilder(self.current_llvm_function.entry_basic_block)
            var_storage_ptr = target_func_entry_builder.alloca(rt.CoralValuePtrType, name=f"{var_name}_storage_ptr")
            self._set_variable_storage(var_name, var_storage_ptr)
        self.current_builder.store(rhs_coral_value_ptr, var_storage_ptr)

    def _generate_expression(self, node: ast_nodes.ASTNode) -> llvmlite.ir.Value:
        if isinstance(node, ast_nodes.LiteralNode):
            return self._generate_literal(node)
        elif isinstance(node, ast_nodes.IdentifierNode):
            return self._generate_identifier(node)
        elif isinstance(node, ast_nodes.BinaryOpNode):
            return self._generate_binary_op(node)
        elif isinstance(node, ast_nodes.UnaryOpNode):
            return self._generate_unary_op(node)
        elif isinstance(node, ast_nodes.CallOperationNode):
            return self._generate_call_operation(node)
        elif isinstance(node, ast_nodes.ListLiteralNode):
            return self._generate_list_literal(node)
        elif isinstance(node, ast_nodes.MapLiteralNode):
            return self._generate_map_literal(node)
        elif isinstance(node, ast_nodes.PropertyAccessNode):
            target_name = node.target_expr.name if hasattr(node.target_expr, 'name') else type(node.target_expr).__name__
            print(f"Warning: Direct generation of PropertyAccessNode '{target_name}.{node.property_name.name}' as expr value not fully implemented.")
            return rt.create_null_value(self.current_builder)
        else:
            raise NotImplementedError(f"Expression node type {type(node).__name__} not supported.")

    def _generate_list_literal(self, node: ast_nodes.ListLiteralNode) -> llvmlite.ir.Value:
        if self.current_builder is None: raise RuntimeError("Builder not init.")
        print("Warning: List literal generation not fully implemented. Calling runtime helper with no elements.")
        runtime_func_name = "coral_runtime_create_list"
        num_elements = llvmlite.ir.Constant(rt.IntegerType, 0)
        elements_array_ptr_type = rt.ptr_to(rt.CoralValuePtrType)
        elements_array_ptr = llvmlite.ir.Constant(elements_array_ptr_type, None)
        func_ty = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [elements_array_ptr_type, rt.IntegerType])
        llvm_function = self.module.globals.get(runtime_func_name)
        if llvm_function is None or not isinstance(llvm_function, llvmlite.ir.Function):
            llvm_function = llvmlite.ir.Function(self.module, func_ty, name=runtime_func_name)
        return self.current_builder.call(llvm_function, [elements_array_ptr, num_elements])

    def _generate_map_literal(self, node: ast_nodes.MapLiteralNode) -> llvmlite.ir.Value:
        if self.current_builder is None: raise RuntimeError("Builder not init.")
        print("Warning: Map literal generation not fully implemented. Calling runtime helper with no entries.")
        runtime_func_name = "coral_runtime_create_map"
        num_entries = llvmlite.ir.Constant(rt.IntegerType, 0)
        keys_array_ptr_type = rt.ptr_to(rt.CoralValuePtrType)
        values_array_ptr_type = rt.ptr_to(rt.CoralValuePtrType)
        keys_ptr = llvmlite.ir.Constant(keys_array_ptr_type, None)
        values_ptr = llvmlite.ir.Constant(values_array_ptr_type, None)
        func_ty = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [keys_array_ptr_type, values_array_ptr_type, rt.IntegerType])
        llvm_function = self.module.globals.get(runtime_func_name)
        if llvm_function is None or not isinstance(llvm_function, llvmlite.ir.Function):
            llvm_function = llvmlite.ir.Function(self.module, func_ty, name=runtime_func_name)
        return self.current_builder.call(llvm_function, [keys_ptr, values_ptr, num_entries])

    def _generate_call_operation(self, node: ast_nodes.CallOperationNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or function not initialized for call.")

        if isinstance(node.callee, ast_nodes.PropertyAccessNode):
            if node.callee.property_name.name == "make":
                if isinstance(node.callee.target_expr, ast_nodes.IdentifierNode):
                    object_name = node.callee.target_expr.name
                    print(f"Info: Object construction for {object_name}.make(). Calling runtime helper.")
                    obj_name_c_str_val = llvmlite.ir.Constant(llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), len(object_name) + 1),
                                                              bytearray(object_name.encode('utf-8') + b'\x00'))
                    g_obj_name_str_var_name = f".str.object_name.{object_name}"
                    g_obj_name_str_var = self.module.globals.get(g_obj_name_str_var_name)
                    if g_obj_name_str_var is None:
                        g_obj_name_str_var = llvmlite.ir.GlobalVariable(self.module, obj_name_c_str_val.type, name=g_obj_name_str_var_name)
                        g_obj_name_str_var.linkage = 'internal'
                        g_obj_name_str_var.global_constant = True
                        g_obj_name_str_var.initializer = obj_name_c_str_val
                    obj_name_ptr = self.current_builder.gep(g_obj_name_str_var,
                                                            [llvmlite.ir.Constant(rt.IntegerType, 0), llvmlite.ir.Constant(rt.IntegerType, 0)],
                                                            name=f"{object_name}_name_ptr")
                    runtime_func_name = "coral_runtime_make_object"
                    func_ty_make_obj = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.ptr_to(rt.IntegerType(8))])
                    llvm_make_func = self.module.globals.get(runtime_func_name)
                    if llvm_make_func is None or not isinstance(llvm_make_func, llvmlite.ir.Function):
                        llvm_make_func = llvmlite.ir.Function(self.module, func_ty_make_obj, name=runtime_func_name)
                    if node.arguments: print(f"Warning: Arguments to {object_name}.make() are ignored.")
                    return self.current_builder.call(llvm_make_func, [obj_name_ptr])
                else:
                    print(f"Warning: obj.make() on non-identifier target {type(node.callee.target_expr)}. Not supported.")
                    return rt.create_null_value(self.current_builder)
            else:
                print(f"Warning: Call on property access '{node.callee.property_name.name}' other than 'make' not supported.")
                return rt.create_null_value(self.current_builder)

        if not isinstance(node.callee, ast_nodes.IdentifierNode):
            print(f"Warning: Callee type {type(node.callee).__name__} not supported for standard calls. Returning null.")
            return rt.create_null_value(self.current_builder)

        func_name = node.callee.name
        generated_args = [self._generate_expression(arg) for arg in node.arguments] if node.arguments else []
        if node.call_style != ast_nodes.CallStyle.FUNCTION:
            print(f"Warning: Call style '{node.call_style}' for '{func_name}' not fully supported.")

        llvm_function = self.module.globals.get(func_name)
        if llvm_function is None or not isinstance(llvm_function, llvmlite.ir.Function):
            print(f"Warning: Function '{func_name}' not found. Declaring with assumed signature.")
            arg_types = [rt.CoralValuePtrType] * len(generated_args)
            func_type = llvmlite.ir.FunctionType(rt.CoralValuePtrType, arg_types)
            llvm_function = llvmlite.ir.Function(self.module, func_type, name=func_name)

        if len(llvm_function.args) != len(generated_args):
            raise ValueError(f"Call to '{func_name}': {len(generated_args)} args, expects {len(llvm_function.args)}.")
        return self.current_builder.call(llvm_function, generated_args, name=f"call_{func_name}")

    def _generate_unary_op(self, node: ast_nodes.UnaryOpNode) -> llvmlite.ir.Value:
        # ... (implementation as before, using self.current_builder, self.current_llvm_function)
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_llvm_function not initialized.")
        operand_coral_val_ptr = self._generate_expression(node.operand)
        operand_type_tag = rt.get_coral_value_type_tag(self.current_builder, operand_coral_val_ptr)
        op_token = node.operator_token.type
        if op_token == ast_nodes.TokenType.MINUS:
            # Simplified: assumes integer, needs proper type checking & branching/phi
            int_val = rt.get_integer_value(self.current_builder, operand_coral_val_ptr, self.current_llvm_function)
            neg_int_val = self.current_builder.neg(int_val, name="neg_int")
            temp_int_alloc = self.current_builder.alloca(rt.IntegerType)
            self.current_builder.store(neg_int_val, temp_int_alloc)
            return rt.create_coral_value(self.current_builder, rt.TYPE_TAG_INTEGER, temp_int_alloc)
        elif op_token == ast_nodes.TokenType.NOT:
            bool_val = rt.get_boolean_value(self.current_builder, operand_coral_val_ptr, self.current_llvm_function)
            not_bool_val = self.current_builder.not_(bool_val, name="not_bool")
            temp_bool_alloc = self.current_builder.alloca(rt.BooleanType)
            self.current_builder.store(not_bool_val, temp_bool_alloc)
            return rt.create_coral_value(self.current_builder, rt.TYPE_TAG_BOOLEAN, temp_bool_alloc)
        else:
            raise NotImplementedError(f"Unary operator {op_token} not yet supported.")
        return rt.create_null_value(self.current_builder)

    def _generate_binary_op(self, node: ast_nodes.BinaryOpNode) -> llvmlite.ir.Value:
        # ... (implementation as before, using self.current_builder, self.current_llvm_function)
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_llvm_function not initialized.")
        lhs_coral_ptr = self._generate_expression(node.left)
        rhs_coral_ptr = self._generate_expression(node.right)
        # Simplified: assumes integer operations, needs proper type checking & branching/phi
        lhs_int = rt.get_integer_value(self.current_builder, lhs_coral_ptr, self.current_llvm_function)
        rhs_int = rt.get_integer_value(self.current_builder, rhs_coral_ptr, self.current_llvm_function)
        op_token = node.operator_token.type
        result_val = None
        if op_token == ast_nodes.TokenType.PLUS:
            result_val = self.current_builder.add(lhs_int, rhs_int, name="add_res")
        elif op_token == ast_nodes.TokenType.MINUS:
            result_val = self.current_builder.sub(lhs_int, rhs_int, name="sub_res")
        elif op_token == ast_nodes.TokenType.MULTIPLY:
            result_val = self.current_builder.mul(lhs_int, rhs_int, name="mul_res")
        elif op_token == ast_nodes.TokenType.DIVIDE:
            result_val = self.current_builder.sdiv(lhs_int, rhs_int, name="sdiv_res")
        elif op_token in [ast_nodes.TokenType.GREATER, ast_nodes.TokenType.GREATER_EQUAL,
                          ast_nodes.TokenType.LESS, ast_nodes.TokenType.LESS_EQUAL,
                          ast_nodes.TokenType.EQUALS_EQUALS, ast_nodes.TokenType.NOT_EQUAL]:
            cmp_op_map = {
                ast_nodes.TokenType.GREATER: 'sgt', ast_nodes.TokenType.GREATER_EQUAL: 'sge',
                ast_nodes.TokenType.LESS: 'slt', ast_nodes.TokenType.LESS_EQUAL: 'sle',
                ast_nodes.TokenType.EQUALS_EQUALS: 'eq', ast_nodes.TokenType.NOT_EQUAL: 'ne',
            }
            cmp_llvm_op = cmp_op_map[op_token]
            result_bool = self.current_builder.icmp_signed(cmp_llvm_op, lhs_int, rhs_int, name=f"cmp_{cmp_llvm_op}_res")
            temp_bool_alloc = self.current_builder.alloca(rt.BooleanType)
            self.current_builder.store(result_bool, temp_bool_alloc)
            return rt.create_coral_value(self.current_builder, rt.TYPE_TAG_BOOLEAN, temp_bool_alloc)
        elif op_token == ast_nodes.TokenType.AND:
            # Assuming boolean operands from CoralValue (after extraction)
            lhs_bool_raw = rt.get_boolean_value(self.current_builder, lhs_coral_ptr, self.current_llvm_function)
            rhs_bool_raw = rt.get_boolean_value(self.current_builder, rhs_coral_ptr, self.current_llvm_function)
            result_bool = self.current_builder.and_(lhs_bool_raw, rhs_bool_raw, name="and_res")
            temp_bool_alloc = self.current_builder.alloca(rt.BooleanType)
            self.current_builder.store(result_bool, temp_bool_alloc)
            return rt.create_coral_value(self.current_builder, rt.TYPE_TAG_BOOLEAN, temp_bool_alloc)
        elif op_token == ast_nodes.TokenType.OR:
            lhs_bool_raw = rt.get_boolean_value(self.current_builder, lhs_coral_ptr, self.current_llvm_function)
            rhs_bool_raw = rt.get_boolean_value(self.current_builder, rhs_coral_ptr, self.current_llvm_function)
            result_bool = self.current_builder.or_(lhs_bool_raw, rhs_bool_raw, name="or_res")
            temp_bool_alloc = self.current_builder.alloca(rt.BooleanType)
            self.current_builder.store(result_bool, temp_bool_alloc)
            return rt.create_coral_value(self.current_builder, rt.TYPE_TAG_BOOLEAN, temp_bool_alloc)
        else:
            raise NotImplementedError(f"Binary operator {op_token} not supported for assumed integer types.")

        # Wrap result if it's a numeric operation that produced result_val
        if result_val is not None:
            temp_res_alloc = self.current_builder.alloca(rt.IntegerType)
            self.current_builder.store(result_val, temp_res_alloc)
            return rt.create_coral_value(self.current_builder, rt.TYPE_TAG_INTEGER, temp_res_alloc)

        print(f"Warning: Binary op {op_token} fell through. Returning null.")
        return rt.create_null_value(self.current_builder)

    def _generate_statement_list(self, statements: list[ast_nodes.ASTNode]):
        if not statements: return
        for stmt_node in statements:
            if stmt_node is None: continue
            self._generate_statement(stmt_node)

    def _generate_statement(self, stmt_node: ast_nodes.ASTNode):
        if isinstance(stmt_node, ast_nodes.AssignmentNode):
            self._generate_assignment(stmt_node)
        elif isinstance(stmt_node, ast_nodes.ExpressionStatementNode):
            full_expr_node = stmt_node.expression
            if not isinstance(full_expr_node, ast_nodes.FullExpressionNode):
                 print(f"Error: ExpressionStatement does not contain FullExpressionNode, got {type(full_expr_node)}. Skipping.")
                 return
            main_expr_value = self._generate_expression(full_expr_node.expression)
            if full_expr_node.error_handler:
                self._generate_error_handler_suffix(full_expr_node.error_handler, main_expr_value)
        elif isinstance(stmt_node, ast_nodes.IfThenElseStatementNode):
            self._generate_if_then_else(stmt_node)
        elif isinstance(stmt_node, ast_nodes.WhileLoopNode):
            self._generate_while_loop(stmt_node)
        elif isinstance(stmt_node, ast_nodes.FunctionDefinitionNode):
            self._generate_function_definition(stmt_node)
        elif isinstance(stmt_node, ast_nodes.ReturnStatementNode):
            self._generate_return_statement(stmt_node)
        elif isinstance(stmt_node, ast_nodes.ObjectDefinitionNode):
            self._generate_object_definition(stmt_node)
        elif isinstance(stmt_node, ast_nodes.StoreDefinitionNode):
            self._generate_store_definition(stmt_node)
        else:
            print(f"Warning: Statement type {type(stmt_node).__name__} not handled by _generate_statement.")

    def _generate_object_definition(self, node: ast_nodes.ObjectDefinitionNode):
        print(f"Info: Processing object definition for '{node.name.name}'. Methods will be defined as mangled functions.")
        for member in node.members:
            if isinstance(member, ast_nodes.FieldDefinitionNode):
                self._generate_field_definition(member, obj_name=node.name.name)
            elif isinstance(member, ast_nodes.MethodDefinitionNode):
                self._generate_method_definition(member, obj_name=node.name.name)

    def _generate_field_definition(self, node: ast_nodes.FieldDefinitionNode, obj_name: str):
        print(f"Info: Field definition '{node.name.name}' in object '{obj_name}' noted. IR generation deferred.")

    def _generate_method_definition(self, node: ast_nodes.MethodDefinitionNode, obj_name: str):
        mangled_name = f"{obj_name}_{node.name.name}"
        print(f"Info: Defining method '{node.name.name}' of '{obj_name}' as function '{mangled_name}'.")
        synthetic_func_node = ast_nodes.FunctionDefinitionNode(
            name=ast_nodes.IdentifierNode(mangled_name, token_details=node.name.token_details),
            params=node.params, body=node.body, token_details=node.token_details)
        self._generate_function_definition(synthetic_func_node)

    def _generate_store_definition(self, node: ast_nodes.StoreDefinitionNode):
        store_name = node.name.name
        print(f"Warning: Store definition for '{store_name}' not fully implemented. Methods processed.")
        if hasattr(node, 'members') and node.members:
             for member in node.members:
                if isinstance(member, ast_nodes.MethodDefinitionNode):
                    mangled_name = f"{store_name}_{member.name.name}"
                    print(f"Info: Defining store method '{member.name.name}' as function '{mangled_name}'.")
                    synthetic_func_node = ast_nodes.FunctionDefinitionNode(
                        name=ast_nodes.IdentifierNode(mangled_name, token_details=member.name.token_details),
                        params=member.params, body=member.body, token_details=member.token_details)
                    self._generate_function_definition(synthetic_func_node)
                elif isinstance(member, (ast_nodes.RelationDefinitionNode, ast_nodes.CastDefinitionNode, ast_nodes.ReceiveDefinitionNode, ast_nodes.FieldDefinitionNode)):
                    print(f"Info: Store member type {type(member).__name__} in '{store_name}' noted. IR deferred.")
                else:
                    print(f"Warning: Unsupported member type {type(member).__name__} in store '{store_name}'.")

    def _generate_error_handler_suffix(self, node: ast_nodes.ErrorHandlerSuffixNode, expression_value: llvmlite.ir.Value) -> llvmlite.ir.Value:
        error_var_name = node.error_variable.name if node.error_variable else "_"
        error_expr_repr = "..."
        if node.error_expression:
            if hasattr(node.error_expression, 'value'): error_expr_repr = str(node.error_expression.value)
            elif hasattr(node.error_expression, 'name'): error_expr_repr = node.error_expression.name
        print(f"Warning: Error handler suffix ('err {error_var_name}' with '{error_expr_repr}') not implemented.")
        return expression_value

    def _generate_return_statement(self, node: ast_nodes.ReturnStatementNode):
        # ... (implementation as before, using self.current_builder)
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Return statement generated outside of a function context.")
        if node.value:
            return_val_ptr = self._generate_expression(node.value)
            self.current_builder.ret(return_val_ptr)
        else:
            null_coral_val = rt.create_null_value(self.current_builder)
            self.current_builder.ret(null_coral_val)

    def _generate_function_definition(self, node: ast_nodes.FunctionDefinitionNode):
        # ... (implementation as before, using self.current_builder, self.current_llvm_function, self.current_symbol_table)
        func_name = node.name.name
        outer_builder = self.current_builder
        outer_llvm_function = self.current_llvm_function
        outer_symbol_table = self.current_symbol_table
        param_types = [rt.CoralValuePtrType] * len(node.params)
        return_type = rt.CoralValuePtrType
        func_type = llvmlite.ir.FunctionType(return_type, param_types)
        llvm_func = Function(self.module, func_type, name=func_name)
        self.current_llvm_function = llvm_func
        self.current_symbol_table = {}
        entry_block = llvm_func.append_basic_block(name="entry")
        self.current_builder = IRBuilder(entry_block)
        for i, param_node in enumerate(node.params):
            param_name = param_node.name.name
            llvm_arg = llvm_func.args[i]
            llvm_arg.name = param_name
            param_storage_ptr = self.current_builder.alloca(rt.CoralValuePtrType, name=f"{param_name}_param_storage")
            self.current_builder.store(llvm_arg, param_storage_ptr)
            self._set_variable_storage(param_name, param_storage_ptr)

        # Process body (assuming node.body is a list of statements)
        if not isinstance(node.body, list): # Check if body is a single expression node (older AST?)
             if isinstance(node.body, ast_nodes.ExpressionNode):
                print(f"Warning: Single expression body for func '{func_name}' - wrapping in return.")
                # This might be too simplistic; depends on how parser creates FunctionDefinitionNode.body
                # If it's an ExpressionNode, it should be wrapped in a ReturnStatementNode for explicit return.
                # For now, assume parser creates a list of statements for body.
                # If it's a single expression that should be returned, it must be part of a ReturnStatement.
                return_stmt = ast_nodes.ReturnStatementNode(value=node.body, token_details=node.body.token_details) # Synthetic
                self._generate_statement_list([return_stmt])

             elif node.body is None: # No body
                 pass # Will fall through to default null return
             else: # Not an ExpressionNode and not a list
                 print(f"Error: Unsupported function body type for '{func_name}': {type(node.body)}. Expecting list of statements.")
                 self._generate_statement_list([]) # Effectively empty body
        else: # node.body is a list
            self._generate_statement_list(node.body)

        if not self.current_builder.block.is_terminated:
            if return_type == rt.CoralValuePtrType:
                null_return_val = rt.create_null_value(self.current_builder)
                self.current_builder.ret(null_return_val)
            elif str(return_type) == "void":
                 self.current_builder.ret_void()
            else:
                print(f"Warning: Func '{func_name}' non-void, lacks return. LLVM might error.")
        self.current_llvm_function = outer_llvm_function
        self.current_symbol_table = outer_symbol_table
        self.current_builder = outer_builder

    def _generate_while_loop(self, node: ast_nodes.WhileLoopNode):
        # ... (implementation as before, using self.current_builder, self.current_llvm_function)
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for while loop.")
        current_llvm_func = self.current_llvm_function
        loop_header_block = current_llvm_func.append_basic_block(name="while.header")
        loop_body_block = current_llvm_func.append_basic_block(name="while.body")
        loop_exit_block = current_llvm_func.append_basic_block(name="while.exit")
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)
        self.current_builder.position_at_end(loop_header_block)
        condition_coral_val_ptr = self._generate_expression(node.condition)
        boolean_condition = rt.get_boolean_value(self.current_builder, condition_coral_val_ptr, current_llvm_func)
        self.current_builder.cbranch(boolean_condition, loop_body_block, loop_exit_block)
        self.current_builder.position_at_end(loop_body_block)
        self._generate_statement_list(node.body)
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)
        self.current_builder.position_at_end(loop_exit_block)

    def _generate_if_then_else(self, node: ast_nodes.IfThenElseStatementNode):
        # ... (implementation as before, using self.current_builder, self.current_llvm_function)
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for if/else.")
        current_llvm_func = self.current_llvm_function
        then_block = current_llvm_func.append_basic_block(name="if.then")
        elseif_cond_blocks = []
        elseif_body_blocks = []
        for i, _ in enumerate(node.else_if_clauses):
            elseif_cond_blocks.append(current_llvm_func.append_basic_block(name=f"elseif.cond.{i}"))
            elseif_body_blocks.append(current_llvm_func.append_basic_block(name=f"elseif.body.{i}"))
        else_block = None
        if node.else_block_stmts:
            else_block = current_llvm_func.append_basic_block(name="if.else")
        merge_block = current_llvm_func.append_basic_block(name="if.merge")
        condition_val_ptr = self._generate_expression(node.condition)
        boolean_condition = rt.get_boolean_value(self.current_builder, condition_val_ptr, current_llvm_func)
        first_alternative_block = elseif_cond_blocks[0] if elseif_cond_blocks else (else_block if else_block else merge_block)
        self.current_builder.cbranch(boolean_condition, then_block, first_alternative_block)
        self.current_builder.position_at_end(then_block)
        self._generate_statement_list(node.if_block_stmts)
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(merge_block)
        for i, else_if_clause in enumerate(node.else_if_clauses):
            self.current_builder.position_at_end(elseif_cond_blocks[i])
            elseif_condition_val_ptr = self._generate_expression(else_if_clause.condition)
            elseif_boolean_condition = rt.get_boolean_value(self.current_builder, elseif_condition_val_ptr, current_llvm_func)
            next_alternative_block = elseif_cond_blocks[i+1] if i + 1 < len(elseif_cond_blocks) else (else_block if else_block else merge_block)
            self.current_builder.cbranch(elseif_boolean_condition, elseif_body_blocks[i], next_alternative_block)
            self.current_builder.position_at_end(elseif_body_blocks[i])
            self._generate_statement_list(else_if_clause.body_stmts)
            if not self.current_builder.block.is_terminated:
                self.current_builder.branch(merge_block)
        if node.else_block_stmts:
            if not else_block: raise Exception("Else block expected but not created.")
            self.current_builder.position_at_end(else_block)
            self._generate_statement_list(node.else_block_stmts)
            if not self.current_builder.block.is_terminated:
                self.current_builder.branch(merge_block)
        self.current_builder.position_at_end(merge_block)
        if not merge_block.predecessors:
             pass

    def generate(self, program_node: ast_nodes.ProgramNode):
        if not isinstance(program_node, ast_nodes.ProgramNode):
            raise TypeError("Expected ProgramNode")
        main_func_type = FunctionType(VoidType(), [])
        main_llvm_func = Function(self.module, main_func_type, name="main")
        self.current_llvm_function = main_llvm_func
        self.current_symbol_table = self.module_context.global_symbol_table
        entry_block = main_llvm_func.append_basic_block(name="entry")
        self.current_builder = IRBuilder(entry_block)
        self._generate_statement_list(program_node.body)
        if self.current_builder.block and self.current_builder.block.terminator is None:
            self.current_builder.ret_void()
        self.current_builder = None
        self.current_llvm_function = None
        self.current_symbol_table = None
        return self.module
