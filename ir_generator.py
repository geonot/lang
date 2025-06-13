import llvmlite.ir
from llvmlite.ir import Constant, FunctionType, VoidType, Function, IRBuilder
import ast_nodes
import ir_runtime_types as rt

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
        rt.declare_runtime_functions(self.module)
        self.current_builder: IRBuilder = None
        self.current_llvm_function: Function = None
        self.current_symbol_table: dict = self.module_context.global_symbol_table
        self.current_module_prefix: list[str] = []
        self.current_object_fields = None
        self.current_store_fields = None
        self.defining_object_or_store_name = None

    def _handle_possible_error_value(self, coral_value_ptr: llvmlite.ir.Value, success_block: llvmlite.ir.Block, name_prefix: str = "err_check"):
        """
        Checks if coral_value_ptr is an error. If so, creates an error block that returns
        the error from the current function. Otherwise, branches to success_block.
        The caller is responsible for positioning the builder into success_block afterwards.
        """
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current LLVM function not initialized for error handling.")

        # Check if the current block is already terminated. If so, no further branching/IR gen needed from here.
        if self.current_builder.block.is_terminated:
            # However, the success_block might still be reachable from other paths,
            # so it's important the caller manages its population.
            # This function's contract is to ensure a branch to success_block if no error,
            # or to an error-returning block if an error. If current path is dead, it can't make that decision.
            # This scenario should ideally be avoided by structuring calling code.
            # For now, we'll assume if the current block is terminated, something else handled control flow.
            # Or, perhaps more correctly, this function should not be called if current_builder.block.is_terminated.
            # Let's add a check for this.
             print(f"Warning: _handle_possible_error_value called from already terminated block {self.current_builder.block.name}. Skipping error check for {coral_value_ptr.name}.")
             # If we skip, the success_block might not be branched to from this path.
             # This implies the caller needs to be careful.
             # A more robust approach might be to ensure this function is only called from unterminated blocks.
             # For now, we'll just proceed, and if the block is terminated, no new branches will be added.
             # This means the success_block must have other predecessors or be the target of a previous branch.
             return


        type_tag = rt.get_coral_value_type_tag(self.current_builder, coral_value_ptr)
        is_error_cond = self.current_builder.icmp_unsigned('==', type_tag, rt.TYPE_TAG_ERROR, name=f"{name_prefix}_is_err_cond")

        error_block = self.current_llvm_function.append_basic_block(name=f"{name_prefix}_{coral_value_ptr.name}_err_path")

        # Ensure success_block is valid and part of the current function
        if success_block not in self.current_llvm_function.basic_blocks:
             # This can happen if success_block was created for a different function context
             # or not properly appended. For safety, let's make a new one if it's problematic.
             # However, the design implies success_block is managed by the caller within the current function.
             # Re-appending it if it's detached might be an option, but could lead to issues if it's from elsewhere.
             # For now, assume success_block is correctly managed by the caller.
             pass


        self.current_builder.cbranch(is_error_cond, error_block, success_block)

        # Populate the error block
        self.current_builder.position_at_end(error_block)
        # Ensure the return type of the function matches CoralValuePtrType
        if self.current_llvm_function.return_value.type == rt.CoralValuePtrType:
            self.current_builder.ret(coral_value_ptr)
        elif self.current_llvm_function.return_value.type == llvmlite.ir.VoidType():
            # If function is void, we can't return the error value directly.
            # This indicates a deeper issue: void functions shouldn't typically be propagating errors this way.
            # For now, print a warning and insert an unreachable to signify broken logic.
            # A better approach might involve a global error flag or a different error propagation mechanism for void functions.
            print(f"Warning: Propagating error value from a void function '{self.current_llvm_function.name}'. This error will be lost.")
            # Optionally, call a runtime error printing function here if available and appropriate.
            # Example: self.current_builder.call(rt.get_runtime_function("coral_runtime_print_detailed_error"), [coral_value_ptr])
            self.current_builder.unreachable() # Or ret_void() if errors are to be ignored in void functions.
        else:
            # Mismatch in return type, problematic.
            print(f"Error: Cannot return CoralValue* error from function '{self.current_llvm_function.name}' with return type {self.current_llvm_function.return_value.type}. Inserting unreachable.")
            self.current_builder.unreachable()

        # DO NOT position builder at success_block here. The caller does that.
        # This is because the caller might want to create PHI nodes in success_block
        # and needs to know all predecessors before filling it.

    def _get_mangled_name(self, original_name: str) -> str:
        if not self.current_module_prefix:
            return original_name
        return "_".join(self.current_module_prefix + [original_name])

    def _create_llvm_global_string_ptr(self, py_string: str, name_prefix: str = ".str") -> llvmlite.ir.Value:
        if self.current_builder is None:
            raise RuntimeError("Builder not initialized for creating global string.")

        clean_py_string = ''.join(c if c.isalnum() else '_' for c in py_string)
        if not clean_py_string or clean_py_string[0].isdigit():
            clean_py_string = f"s_{clean_py_string}"
        global_var_name = self.module.get_unique_name(f"{name_prefix}.{clean_py_string}")

        g_var = self.module.globals.get(global_var_name)
        if g_var is None:
            c_str_val = llvmlite.ir.Constant(
                llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), len(py_string) + 1),
                bytearray(py_string.encode('utf-8') + b'\x00')
            )
            g_var = llvmlite.ir.GlobalVariable(self.module, c_str_val.type, name=global_var_name)
            g_var.linkage = 'internal'
            g_var.global_constant = True
            g_var.initializer = c_str_val

        idx_type = llvmlite.ir.IntType(32)
        return self.current_builder.gep(
            g_var,
            [llvmlite.ir.Constant(idx_type, 0), llvmlite.ir.Constant(idx_type, 0)],
            name=f"{clean_py_string}_gep_ptr"
        )

    def _create_coral_string_literal_val(self, py_string: str) -> llvmlite.ir.Value:
        """Helper to create a CoralValue* for a Python string literal."""
        if self.current_builder is None or self.module is None:
            raise RuntimeError("Builder or Module not available for creating Coral string literal.")
        return rt.create_string_value(self.current_builder, self.module, py_string)

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
        literal_type = node.literal_type
        value = node.value

        if literal_type == ast_nodes.LiteralType.INTEGER:
            return rt.create_integer_value(self.current_builder, int(value))
        elif literal_type == ast_nodes.LiteralType.FLOAT:
            return rt.create_float_value(self.current_builder, float(value))
        elif literal_type == ast_nodes.LiteralType.BOOLEAN:
            return rt.create_boolean_value(self.current_builder, value)
        elif literal_type == ast_nodes.LiteralType.STRING:
            return self._create_coral_string_literal_val(str(value)) # Use helper
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

        if var_name == "self" and self.defining_object_or_store_name:
             pass

        var_storage_ptr = self._get_variable_storage(var_name)
        if var_storage_ptr is None:
            print(f"Error: Identifier '{var_name}' used before assignment. Returning null value.")
            return rt.create_null_value(self.current_builder)
        return self.current_builder.load(var_storage_ptr, name=f"{var_name}_val_ptr_loaded")

    def _generate_assignment(self, node: ast_nodes.AssignmentNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or function not initialized for assignment.")

        rhs_coral_value_ptr: llvmlite.ir.Value | None = None
        if isinstance(node.value, ast_nodes.MapBlockAssignmentRHSNode):
            rhs_coral_value_ptr = self._generate_map_from_block(node.value)
        elif isinstance(node.value, ast_nodes.ASTNode):
            rhs_coral_value_ptr = self._generate_expression(node.value)
        else:
            raise TypeError(f"Unsupported RHS type for assignment: {type(node.value)}")

        if rhs_coral_value_ptr is None:
            target_name_for_error = getattr(node.target, 'name', 'unknown_target')
            print(f"Error: RHS of assignment to '{target_name_for_error}' evaluated to None. Skipping assignment.")
            return

        if isinstance(node.target, ast_nodes.IdentifierNode):
            var_name = node.target.name
            var_storage_ptr = self._get_variable_storage(var_name)
            if var_storage_ptr is None:
                entry_b = self.current_llvm_function.entry_basic_block
                entry_builder = IRBuilder(entry_b)
                first_instr_not_alloca = next((instr for instr in entry_b.instructions if not isinstance(instr, llvmlite.ir.AllocaInstr)), None)
                if first_instr_not_alloca:
                    entry_builder.position_before(first_instr_not_alloca)
                elif entry_b.instructions:
                     entry_builder.position_after(entry_b.instructions[-1])

                var_storage_ptr = entry_builder.alloca(rt.CoralValuePtrType, name=f"{var_name}_storage")
                self._set_variable_storage(var_name, var_storage_ptr)
            self.current_builder.store(rhs_coral_value_ptr, var_storage_ptr)
        elif isinstance(node.target, ast_nodes.ListElementAccessNode):
            target_list_access_node = node.target
            list_ptr = self._generate_expression(target_list_access_node.base_expr)
            index_ptr = self._generate_expression(target_list_access_node.index_expr)
            set_elem_func = rt.get_runtime_function("coral_runtime_list_set_element")
            self.current_builder.call(set_elem_func, [list_ptr, index_ptr, rhs_coral_value_ptr])
        elif isinstance(node.target, ast_nodes.PropertyAccessNode):
            target_prop_access_node = node.target
            obj_ptr = self._generate_expression(target_prop_access_node.base_expr)
            prop_name_str = target_prop_access_node.property_name.name
            # Property names for runtime calls should also be CoralValue strings
            prop_name_cv_ptr = self._create_coral_string_literal_val(prop_name_str)
            set_prop_func = rt.get_runtime_function("coral_runtime_object_set_property")
            self.current_builder.call(set_prop_func, [obj_ptr, prop_name_cv_ptr, rhs_coral_value_ptr])
        else:
            raise NotImplementedError(f"Assignment target type {type(node.target).__name__} not implemented.")

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
        elif isinstance(node, ast_nodes.ListElementAccessNode):
            return self._generate_list_element_access(node)
        elif isinstance(node, ast_nodes.PropertyAccessNode):
            return self._generate_property_access(node)
        elif isinstance(node, ast_nodes.TernaryConditionalExpressionNode):
            return self._generate_ternary_conditional_expression(node)
        elif isinstance(node, ast_nodes.DollarParamNode):
            return self._generate_dollar_param(node)
        else:
            raise NotImplementedError(f"Expression node type {type(node).__name__} not supported.")

    def _generate_list_literal(self, node: ast_nodes.ListLiteralNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for ListLiteralNode.")
        num_elements = len(node.elements)
        llvm_num_elements_i32 = llvmlite.ir.Constant(llvmlite.ir.IntType(32), num_elements)
        elements_array_storage_ptr: llvmlite.ir.Value
        if num_elements == 0:
            elements_array_storage_ptr = llvmlite.ir.Constant(rt.ptr_to(rt.CoralValuePtrType), None)
        else:
            elements_array_alloca = self.current_builder.alloca(rt.CoralValuePtrType, size=llvm_num_elements_i32, name="list_lit_elems_storage")
            for i, element_ast in enumerate(node.elements):
                elem_val_ptr = self._generate_expression(element_ast)
                slot_ptr = self.current_builder.gep(elements_array_alloca, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)], name=f"list_elem_slot_{i}_ptr")
                self.current_builder.store(elem_val_ptr, slot_ptr)
            elements_array_storage_ptr = elements_array_alloca
        create_list_func = rt.get_runtime_function("coral_runtime_create_list")
        new_list_ptr = self.current_builder.call(create_list_func, [elements_array_storage_ptr, llvm_num_elements_i32], name="new_list_from_literal")
        return new_list_ptr

    def _generate_map_from_block(self, rhs_node: ast_nodes.MapBlockAssignmentRHSNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for MapBlockAssignmentRHSNode.")
        num_entries = len(rhs_node.entries)
        llvm_num_entries_i32 = llvmlite.ir.Constant(llvmlite.ir.IntType(32), num_entries)
        keys_array_ptr: llvmlite.ir.Value
        values_array_ptr: llvmlite.ir.Value
        if num_entries == 0:
            keys_array_ptr = llvmlite.ir.Constant(rt.ptr_to(rt.CoralValuePtrType), None)
            values_array_ptr = llvmlite.ir.Constant(rt.ptr_to(rt.CoralValuePtrType), None)
        else:
            keys_array_alloca = self.current_builder.alloca(rt.CoralValuePtrType, size=llvm_num_entries_i32, name="map_block_keys_storage")
            values_array_alloca = self.current_builder.alloca(rt.CoralValuePtrType, size=llvm_num_entries_i32, name="map_block_vals_storage")
            for i, map_block_entry_node in enumerate(rhs_node.entries):
                key_name_str = map_block_entry_node.key.name
                key_coral_str_ptr = self._create_coral_string_literal_val(key_name_str)
                value_coral_val_ptr = self._generate_expression(map_block_entry_node.value)
                key_slot_ptr = self.current_builder.gep(keys_array_alloca, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)])
                self.current_builder.store(key_coral_str_ptr, key_slot_ptr)
                val_slot_ptr = self.current_builder.gep(values_array_alloca, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)])
                self.current_builder.store(value_coral_val_ptr, val_slot_ptr)
            keys_array_ptr = keys_array_alloca
            values_array_ptr = values_array_alloca
        create_map_func = rt.get_runtime_function("coral_runtime_create_map")
        new_map_ptr = self.current_builder.call(create_map_func, [keys_array_ptr, values_array_ptr, llvm_num_entries_i32], name="new_map_from_block")
        return new_map_ptr

    def _generate_map_literal(self, node: ast_nodes.MapLiteralNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for MapLiteralNode.")
        num_entries = len(node.entries)
        llvm_num_entries_i32 = llvmlite.ir.Constant(llvmlite.ir.IntType(32), num_entries)
        keys_array_ptr: llvmlite.ir.Value
        values_array_ptr: llvmlite.ir.Value
        if num_entries == 0:
            keys_array_ptr = llvmlite.ir.Constant(rt.ptr_to(rt.CoralValuePtrType), None)
            values_array_ptr = llvmlite.ir.Constant(rt.ptr_to(rt.CoralValuePtrType), None)
        else:
            keys_array_alloca = self.current_builder.alloca(rt.CoralValuePtrType, size=llvm_num_entries_i32, name="map_lit_keys_storage")
            values_array_alloca = self.current_builder.alloca(rt.CoralValuePtrType, size=llvm_num_entries_i32, name="map_lit_vals_storage")
            for i, map_entry_node in enumerate(node.entries):
                key_name_str = map_entry_node.key.name
                key_coral_str_ptr = self._create_coral_string_literal_val(key_name_str)
                value_coral_val_ptr = self._generate_expression(map_entry_node.value)
                key_slot_ptr = self.current_builder.gep(keys_array_alloca, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)])
                self.current_builder.store(key_coral_str_ptr, key_slot_ptr)
                val_slot_ptr = self.current_builder.gep(values_array_alloca, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)])
                self.current_builder.store(value_coral_val_ptr, val_slot_ptr)
            keys_array_ptr = keys_array_alloca
            values_array_ptr = values_array_alloca
        create_map_func = rt.get_runtime_function("coral_runtime_create_map")
        new_map_ptr = self.current_builder.call(create_map_func, [keys_array_ptr, values_array_ptr, llvm_num_entries_i32], name="new_map_from_literal")
        return new_map_ptr

    def _generate_list_element_access(self, node: ast_nodes.ListElementAccessNode) -> llvmlite.ir.Value:
        if self.current_builder is None: raise RuntimeError("Builder not set")
        base_val_ptr = self._generate_expression(node.base_expr)
        index_val_ptr = self._generate_expression(node.index_expr)
        get_elem_func = rt.get_runtime_function("coral_runtime_list_get_element")
        return self.current_builder.call(get_elem_func, [base_val_ptr, index_val_ptr], name="list_elem_ptr")

    def _generate_property_access(self, node: ast_nodes.PropertyAccessNode) -> llvmlite.ir.Value:
        if self.current_builder is None: raise RuntimeError("Builder not set")
        base_val_ptr = self._generate_expression(node.base_expr)
        property_name_str = node.property_name.name
        prop_name_cv_ptr = self._create_coral_string_literal_val(property_name_str) # Property name as CoralValue
        get_prop_func = rt.get_runtime_function("coral_runtime_object_get_property")
        return self.current_builder.call(get_prop_func, [base_val_ptr, prop_name_cv_ptr], name="prop_val_ptr")

    def _generate_ternary_conditional_expression(self, node: ast_nodes.TernaryConditionalExpressionNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or function not initialized for ternary conditional.")

        condition_cv = self._generate_expression(node.condition)
        require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
        checked_condition_cv = self.current_builder.call(require_bool_fn, [condition_cv], name="checked_tern_cond_cv")

        # Create a success block for boolean extraction
        bool_extract_block = self.current_llvm_function.append_basic_block(name="tern_cond_bool_extract")
        self._handle_possible_error_value(checked_condition_cv, bool_extract_block, name_prefix="tern_cond")
        self.current_builder.position_at_end(bool_extract_block)

        boolean_condition_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_condition_cv)

        then_block = self.current_llvm_function.append_basic_block(name="ternary.then")
        else_block = self.current_llvm_function.append_basic_block(name="ternary.else")
        merge_block = self.current_llvm_function.append_basic_block(name="ternary.merge")
        # Ensure the current block (bool_extract_block) is not terminated before adding new cbranch
        if not self.current_builder.block.is_terminated:
            self.current_builder.cbranch(boolean_condition_i1, then_block, else_block)
        else:
            # If bool_extract_block was terminated (e.g. by error handling),
            # then_block/else_block might be orphaned if not careful.
            # This implies _handle_possible_error_value should be the last thing before caller repositions.
            # For now, this structure assumes _handle_possible_error_value doesn't terminate bool_extract_block itself.
            pass
        self.current_builder.position_at_end(then_block)
        then_val_ptr = self._generate_expression(node.true_expr)
        then_pred_block = self.current_builder.block
        if not then_block.is_terminated: self.current_builder.branch(merge_block)
        self.current_builder.position_at_end(else_block)
        else_val_ptr = self._generate_expression(node.false_expr)
        else_pred_block = self.current_builder.block
        if not else_block.is_terminated: self.current_builder.branch(merge_block)
        self.current_builder.position_at_end(merge_block)
        if not merge_block.predecessors:
            return rt.create_null_value(self.current_builder)
        phi_node = self.current_builder.phi(rt.CoralValuePtrType, name="ternary.val")
        if then_pred_block in merge_block.predecessors:
             phi_node.add_incoming(then_val_ptr, then_pred_block)
        if else_pred_block in merge_block.predecessors:
            phi_node.add_incoming(else_val_ptr, else_pred_block)
        if not phi_node.incoming:
             if len(merge_block.predecessors) == 1:
                 if merge_block.predecessors[0] == then_pred_block: return then_val_ptr
                 if merge_block.predecessors[0] == else_pred_block: return else_val_ptr
             if merge_block.predecessors:
                 print("Warning: Ternary merge block is reachable but PHI node has no incoming values.")
             return rt.create_null_value(self.current_builder)
        return phi_node

    def _generate_dollar_param(self, node: ast_nodes.DollarParamNode) -> llvmlite.ir.Value:
        if self.current_builder is None: raise RuntimeError("Builder not set")
        print(f"Warning: $parameter '{node.name_or_index}' not fully implemented. Returning null.")
        return rt.create_null_value(self.current_builder)

    def _generate_call_operation(self, node: ast_nodes.CallOperationNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or function not initialized for call.")

        generated_args = [self._generate_expression(arg.value if isinstance(arg, ast_nodes.ArgumentNode) else arg) for arg in node.arguments]

        if isinstance(node.callee, ast_nodes.PropertyAccessNode):
            base_object_cv_ptr = self._generate_expression(node.callee.base_expr)
            method_name = node.callee.property_name.name

            class_name_for_mangling = self.defining_object_or_store_name
            if isinstance(node.callee.base_expr, ast_nodes.IdentifierNode):
                if node.callee.base_expr.name != "self": # If it's not 'self', assume it might be a direct class name
                    class_name_for_mangling = node.callee.base_expr.name

            if not class_name_for_mangling: # Fallback if class context is unclear
                # This situation is tricky. Without type info, we're guessing.
                # A robust solution needs runtime type checking or better static info.
                # For now, if it's not 'self' and not an identifiable class, this will be an issue.
                print(f"Warning: Cannot determine class for method call '{method_name}' on base '{node.callee.base_expr}'. Attempting direct mangling.")
                # Attempt a more direct mangling if base_expr is an identifier (e.g. could be module.class)
                if isinstance(node.callee.base_expr, ast_nodes.IdentifierNode):
                    mangled_callee_name = self._get_mangled_name(f"{node.callee.base_expr.name}_{method_name}")
                else: # Cannot determine class context
                     print(f"Error: Method call on non-identifier base '{type(node.callee.base_expr)}' without clear class context for '{method_name}'. Returning null.")
                     return rt.create_null_value(self.current_builder)
            else:
                mangled_callee_name = self._get_mangled_name(f"{class_name_for_mangling}_{method_name}")

            llvm_function = self.module.globals.get(mangled_callee_name)
            if llvm_function is None or not isinstance(llvm_function, llvmlite.ir.Function):
                print(f"Warning: Method '{mangled_callee_name}' not found. Returning null.")
                return rt.create_null_value(self.current_builder)

            final_args = [base_object_cv_ptr] + generated_args

            if len(llvm_function.args) != len(final_args):
                raise ValueError(f"Method call to '{mangled_callee_name}': {len(final_args)} args (incl. self), expects {len(llvm_function.args)}.")
            return self.current_builder.call(llvm_function, final_args, name=f"call_method_{method_name}")

        elif isinstance(node.callee, ast_nodes.IdentifierNode): # Global function call
            original_func_name = node.callee.name
            mangled_callee_name = self._get_mangled_name(original_func_name)

            llvm_function = self.module.globals.get(mangled_callee_name)
            if llvm_function is None or not isinstance(llvm_function, llvmlite.ir.Function):
                print(f"Warning: Function '{mangled_callee_name}' not found. Declaring with assumed signature.")
                arg_count = len(generated_args)
                assumed_arg_types = [rt.CoralValuePtrType] * arg_count
                assumed_func_type = llvmlite.ir.FunctionType(rt.CoralValuePtrType, assumed_arg_types)
                llvm_function = llvmlite.ir.Function(self.module, assumed_func_type, name=mangled_callee_name)

            if len(llvm_function.args) != len(generated_args):
                raise ValueError(f"Call to '{mangled_callee_name}': incorrect number of arguments. "
                                 f"Provided {len(generated_args)}, expected {len(llvm_function.args)}.")
            return self.current_builder.call(llvm_function, generated_args, name=f"call_{original_func_name}")
        else:
            print(f"Warning: Callee type {type(node.callee).__name__} not supported for calls. Returning null.")
            return rt.create_null_value(self.current_builder)

    def _generate_unary_op(self, node: ast_nodes.UnaryOpNode) -> llvmlite.ir.Value:
        operand_coral_val_ptr = self._generate_expression(node.operand)
        op_str_map = {"-": "neg", "not": "not"}
        runtime_op_key = op_str_map.get(node.operator)
        if runtime_op_key is None:
            raise NotImplementedError(f"Unary operator '{node.operator}' not mapped.")
        runtime_fn = rt.get_runtime_function(runtime_op_key)

        result_cv = self.current_builder.call(runtime_fn, [operand_coral_val_ptr], name=f"unary_{runtime_op_key}_res")

        op_success_block = self.current_llvm_function.append_basic_block(name=f"unary_{runtime_op_key}_success")
        self._handle_possible_error_value(result_cv, op_success_block, name_prefix=f"unary_{runtime_op_key}")
        self.current_builder.position_at_end(op_success_block)

        return result_cv

    def _generate_binary_op(self, node: ast_nodes.BinaryOpNode) -> llvmlite.ir.Value:
        lhs_coral_val_ptr = self._generate_expression(node.left)
        rhs_coral_val_ptr = self._generate_expression(node.right)
        op_str_map = {
            "+": "add", "-": "sub", "*": "mul", "/": "div", "%": "mod",
            "equals": "eq", "==": "eq", "!=": "ne",
            "<": "lt", "<=": "le", ">": "gt", ">=": "ge",
            "and": "and", "or": "or"
        }
        runtime_op_key = op_str_map.get(node.operator)
        if runtime_op_key is None:
            raise NotImplementedError(f"Binary operator '{node.operator}' not mapped.")
        runtime_fn = rt.get_runtime_function(runtime_op_key)

        result_cv = self.current_builder.call(runtime_fn, [lhs_coral_val_ptr, rhs_coral_val_ptr], name=f"binary_{runtime_op_key}_res")

        op_success_block = self.current_llvm_function.append_basic_block(name=f"binary_{runtime_op_key}_success")
        self._handle_possible_error_value(result_cv, op_success_block, name_prefix=f"binary_{runtime_op_key}")
        self.current_builder.position_at_end(op_success_block)

        return result_cv

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
                _ = self._generate_error_handler_suffix(full_expr_node.error_handler, main_expr_value)
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
        elif isinstance(stmt_node, ast_nodes.ModuleDefinitionNode):
            self._generate_module_definition(stmt_node)
        elif isinstance(stmt_node, ast_nodes.UnlessStatementNode):
            self._generate_unless_statement(stmt_node)
        elif isinstance(stmt_node, ast_nodes.UntilLoopNode):
            self._generate_until_loop(stmt_node)
        elif isinstance(stmt_node, ast_nodes.IterateLoopNode):
            self._generate_iterate_loop(stmt_node)
        elif isinstance(stmt_node, ast_nodes.UseStatementNode):
            self._generate_use_statement(stmt_node)
        elif isinstance(stmt_node, ast_nodes.EmptyStatementNode):
            self._generate_empty_statement(stmt_node)
        else:
            print(f"Warning: Statement type {type(stmt_node).__name__} not handled by _generate_statement.")

    def _generate_object_definition(self, node: ast_nodes.ObjectDefinitionNode):
        obj_name = node.name.name
        print(f"Info: Processing object definition for '{obj_name}'.")

        original_object_fields_backup = self.current_object_fields
        self.current_object_fields = []

        original_defining_object_name = self.defining_object_or_store_name
        self.defining_object_or_store_name = obj_name

        self.current_module_prefix.append(obj_name)
        try:
            for member in node.members:
                if isinstance(member, ast_nodes.FieldDefinitionNode):
                    self._generate_field_definition(member, owner_name=obj_name, is_store_field=False)
                elif isinstance(member, ast_nodes.MethodDefinitionNode):
                    self._generate_method_definition(member, obj_name=obj_name)

            field_summary_parts = []
            if self.current_object_fields:
                for field_name, default_val_ir in self.current_object_fields:
                    field_summary_parts.append(f"{field_name} ({'has default IR' if default_val_ir is not None else 'no default'})")

            summary_message = f"Object '{obj_name}' defined"
            if field_summary_parts:
                summary_message += f" with fields: {', '.join(field_summary_parts)}."
            else:
                summary_message += " with no fields."
            print(summary_message)

        finally:
            if self.current_module_prefix and self.current_module_prefix[-1] == obj_name:
                self.current_module_prefix.pop()
            self.current_object_fields = original_object_fields_backup
            self.defining_object_or_store_name = original_defining_object_name

    def _generate_field_definition(self, node: ast_nodes.FieldDefinitionNode, owner_name: str, is_store_field: bool = False):
        field_name = node.name.name
        default_value_cv_ptr = None

        if node.default_value:
            if self.current_builder and self.current_llvm_function: # Check if we are in a gen context
                try:
                    default_value_cv_ptr = self._generate_expression(node.default_value)
                except Exception as e:
                    print(f"Error generating default value IR for field '{field_name}' in '{owner_name}': {e}")
            # else: Default value AST exists, but no context to generate its IR now.

        field_info = (field_name, default_value_cv_ptr)

        if is_store_field:
            if self.current_store_fields is not None:
                self.current_store_fields.append(field_info)
            else:
                print(f"Warning: current_store_fields not initialized for field '{field_name}' in store '{owner_name}'.")
        else:
            if self.current_object_fields is not None:
                self.current_object_fields.append(field_info)
            else:
                print(f"Warning: current_object_fields not initialized for field '{field_name}' in object '{owner_name}'.")

    def _generate_relation_definition(self, node: ast_nodes.RelationDefinitionNode, owner_name: str):
        if not self.current_builder: return # Cannot generate runtime call without builder
        store_name_cv_ptr = self._create_coral_string_literal_val(owner_name)
        relation_name_cv_ptr = self._create_coral_string_literal_val(node.name.name)
        runtime_fn = rt.get_runtime_function("coral_runtime_define_relation")
        self.current_builder.call(runtime_fn, [store_name_cv_ptr, relation_name_cv_ptr])
        print(f"Info: Relation '{node.name.name}' defined for store '{owner_name}'.")


    def _generate_cast_definition(self, node: ast_nodes.CastDefinitionNode, owner_name: str):
        if not self.current_builder: return
        store_name_cv_ptr = self._create_coral_string_literal_val(owner_name)
        target_type_cv_ptr = self._create_coral_string_literal_val(node.cast_to_type)
        cast_body_func_cv_ptr = None

        if isinstance(node.body, ast_nodes.ASTNode): # ExpressionNode
             # This simplification passes the *result* of the expression if evaluated now.
             # A true cast body would be a function. For now, if it's an expr, we can't turn it into a func ptr easily here.
             # So, we'll treat it as if the expression itself IS the cast body representation for now, or null.
            print(f"Warning: Cast body expression for '{node.cast_to_type}' in store '{owner_name}' cannot be directly used as function pointer. Passing null.")
            cast_body_func_cv_ptr = rt.create_null_value(self.current_builder)
        elif isinstance(node.body, list): # Block of statements
            print(f"Info: Cast body block for '{node.cast_to_type}' in store '{owner_name}'. IR generation deferred, passing null for body.")
            cast_body_func_cv_ptr = rt.create_null_value(self.current_builder)
        else: # Should not happen
            cast_body_func_cv_ptr = rt.create_null_value(self.current_builder)

        runtime_fn = rt.get_runtime_function("coral_runtime_define_cast")
        self.current_builder.call(runtime_fn, [store_name_cv_ptr, target_type_cv_ptr, cast_body_func_cv_ptr])
        print(f"Info: Cast to '{node.cast_to_type}' defined for store '{owner_name}'.")


    def _generate_receive_handler(self, node: ast_nodes.ReceiveHandlerNode, owner_name: str):
        if not self.current_builder: return

        store_name_cv_ptr = self._create_coral_string_literal_val(owner_name)
        message_name_cv_ptr = self._create_coral_string_literal_val(node.message_name.name)

        # Generate a new LLVM function for the handler body
        handler_func_name = self._get_mangled_name(f"{owner_name}_{node.message_name.name}_handler")

        # Assume handler takes (self_store_ptr: CoralValue*, message_payload_ptr: CoralValue*) -> CoralValue*
        # For now, simplify to (message_payload_ptr: CoralValue*) -> CoralValue*
        # The actual self_store_ptr might be implicitly available in runtime or passed if actor model needs it.
        # Let's assume a simplified (MessagePayload_CoralValue*) -> CoralValue* for now.
        # The runtime will call this. The 'self' of the store is known to the runtime dispatcher.
        param_types = [rt.CoralValuePtrType] # For the message
        func_type = llvmlite.ir.FunctionType(rt.CoralValuePtrType, param_types)

        handler_llvm_func = llvmlite.ir.Function(self.module, func_type, name=handler_func_name)

        # Generate body of this handler function
        original_builder = self.current_builder
        original_func = self.current_llvm_function
        original_sym_table = self.current_symbol_table
        original_defining_owner = self.defining_object_or_store_name

        self.current_llvm_function = handler_llvm_func
        self.current_symbol_table = {} # New scope
        self.defining_object_or_store_name = owner_name # Context for 'self' if used in handler body

        entry_block = handler_llvm_func.append_basic_block(name="entry")
        self.current_builder = IRBuilder(entry_block)

        # TODO: Setup parameters for the handler_llvm_func if its signature requires them (e.g. message object)
        # For now, assuming node.body doesn't rely on specific params other than what's in its scope
        # If the handler signature is (message_param), then:
        # msg_arg = handler_llvm_func.args[0]
        # msg_arg.name = "message"
        # msg_storage = self.current_builder.alloca(rt.CoralValuePtrType, name="message_storage")
        # self.current_builder.store(msg_arg, msg_storage)
        # self._set_variable_storage("message", msg_storage)


        if not isinstance(node.body, list):
            if isinstance(node.body, ast_nodes.ASTNode):
                ret_val = self._generate_expression(node.body)
                if not self.current_builder.block.is_terminated: self.current_builder.ret(ret_val)
            else: # Should not happen
                 if not self.current_builder.block.is_terminated: self.current_builder.ret(rt.create_null_value(self.current_builder))
        else:
            self._generate_statement_list(node.body)
            if not self.current_builder.block.is_terminated: # Ensure default return if no explicit one
                self.current_builder.ret(rt.create_null_value(self.current_builder))

        self.current_builder = original_builder
        self.current_llvm_function = original_func
        self.current_symbol_table = original_sym_table
        self.defining_object_or_store_name = original_defining_owner

        # Wrap LLVM function pointer for the runtime call.
        # This is a simplification. Ideally, a function pointer type or a callable CoralValue type exists.
        # Casting to i8* and wrapping in an object CoralValue for now.
        func_ptr_i8 = self.current_builder.bitcast(handler_llvm_func, rt.OpaquePtrType)
        handler_func_cv_ptr = rt.create_object_value(self.current_builder, func_ptr_i8) # Using TYPE_TAG_OBJECT as placeholder for function

        runtime_fn_define_handler = rt.get_runtime_function("coral_runtime_define_receive_handler")
        self.current_builder.call(runtime_fn_define_handler, [store_name_cv_ptr, message_name_cv_ptr, handler_func_cv_ptr])
        print(f"Info: Receive handler for '{node.message_name.name}' in store '{owner_name}' defined with LLVM func '{handler_func_name}'.")


    def _generate_method_definition(self, node: ast_nodes.MethodDefinitionNode, obj_name: str):
        intermediate_mangled_method_name = f"{obj_name}_{node.name.name}"
        final_llvm_name = self._get_mangled_name(intermediate_mangled_method_name)

        original_defining_object_name = self.defining_object_or_store_name
        self.defining_object_or_store_name = obj_name

        # Parameters for LLVM: first is 'self', then actual params
        llvm_param_types = [rt.CoralValuePtrType] + [rt.CoralValuePtrType] * len(node.params)
        return_type = rt.CoralValuePtrType
        func_type = llvmlite.ir.FunctionType(return_type, llvm_param_types)

        llvm_func = Function(self.module, func_type, name=final_llvm_name)

        outer_builder = self.current_builder
        outer_llvm_function = self.current_llvm_function
        outer_symbol_table = self.current_symbol_table

        self.current_llvm_function = llvm_func
        self.current_symbol_table = {}

        entry_block = llvm_func.append_basic_block(name="entry")
        self.current_builder = IRBuilder(entry_block)

        # Handle 'self' - llvm_func.args[0]
        self_llvm_arg = llvm_func.args[0]
        self_llvm_arg.name = "self"
        self_storage_ptr = self.current_builder.alloca(rt.CoralValuePtrType, name="self_storage")
        self.current_builder.store(self_llvm_arg, self_storage_ptr)
        self._set_variable_storage("self", self_storage_ptr)

        # Handle other params - llvm_func.args[1:]
        for i, param_node in enumerate(node.params):
            param_name = param_node.name.name
            llvm_arg = llvm_func.args[i + 1] # +1 because args[0] is 'self'
            llvm_arg.name = param_name
            param_storage_ptr = self.current_builder.alloca(rt.CoralValuePtrType, name=f"{param_name}_param_storage")

            final_value_to_store = llvm_arg # Default to using the passed argument.

            if param_node.default_value:
                default_value_cv_ptr = self._generate_expression(param_node.default_value)
                if rt.CoralValuePtrType == llvm_arg.type: # Ensure it's a CoralValue*
                    is_passed_arg_null_cv = self.current_builder.call(
                        rt.get_runtime_function("coral_runtime_is_null"), [llvm_arg], name=f"is_null_{param_name}_cv"
                    )
                    require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
                    checked_is_null_cv = self.current_builder.call(require_bool_fn, [is_passed_arg_null_cv], name=f"checked_is_null_{param_name}_cv")
                    is_passed_arg_null_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_is_null_cv)

                    with self.current_builder.if_then(is_passed_arg_null_i1):
                        self.current_builder.store(default_value_cv_ptr, param_storage_ptr)
                    with self.current_builder.else_():
                        self.current_builder.store(llvm_arg, param_storage_ptr)
                    self._set_variable_storage(param_name, param_storage_ptr)
                    continue # Skip the store below

            # If no default or the "magic null" condition wasn't met, store the original llvm_arg.
            self.current_builder.store(final_value_to_store, param_storage_ptr)
            self._set_variable_storage(param_name, param_storage_ptr)

        # Body generation (refined for single expression)
        if not isinstance(node.body, list):
            if isinstance(node.body, ast_nodes.ASTNode):
                return_val_cv_ptr = self._generate_expression(node.body)
                if not self.current_builder.block.is_terminated: # Check if expression itself caused termination
                    self.current_builder.ret(return_val_cv_ptr)
            elif node.body is None: # No body, implicit null return
                 if not self.current_builder.block.is_terminated:
                    self.current_builder.ret(rt.create_null_value(self.current_builder))
            else:
                 print(f"Error: Unsupported method body type for '{final_llvm_name}': {type(node.body)}.")
                 if not self.current_builder.block.is_terminated:
                    self.current_builder.ret(rt.create_null_value(self.current_builder))
        else:
            self._generate_statement_list(node.body)
            if not self.current_builder.block.is_terminated: # Default null return if block doesn't end with ret
                if return_type == rt.CoralValuePtrType:
                    self.current_builder.ret(rt.create_null_value(self.current_builder))
                elif return_type == VoidType():
                    self.current_builder.ret_void()

        self.current_llvm_function = outer_llvm_function
        self.current_symbol_table = outer_symbol_table
        self.current_builder = outer_builder
        self.defining_object_or_store_name = original_defining_object_name


    def _generate_store_definition(self, node: ast_nodes.StoreDefinitionNode):
        store_name = node.name.name
        print(f"Info: Processing store definition for '{store_name}'.")

        if node.is_actor:
            print(f"Info: Store '{store_name}' is an actor.")
        if node.for_target:
            target_name = node.for_target.name if hasattr(node.for_target, 'name') else str(node.for_target)
            print(f"Info: Store '{store_name}' is for target '{target_name}'.")

        original_store_fields_backup = self.current_store_fields
        self.current_store_fields = []

        original_defining_object_name = self.defining_object_or_store_name
        self.defining_object_or_store_name = store_name

        self.current_module_prefix.append(store_name)
        # Temporarily set builder context if not already in one, for store member runtime calls
        # This is tricky as store defs are usually global. For now, assume builder is available from main or module init.
        # If not, calls to runtime for define_relation etc. will fail.
        # For this pass, we assume _generate_statement_list for ProgramNode sets up a builder.

        try:
            if hasattr(node, 'members') and node.members:
                 for member in node.members:
                    if isinstance(member, ast_nodes.MethodDefinitionNode):
                        self._generate_method_definition(member, obj_name=store_name)
                    elif isinstance(member, ast_nodes.FieldDefinitionNode):
                        self._generate_field_definition(member, owner_name=store_name, is_store_field=True)
                    elif isinstance(member, ast_nodes.RelationDefinitionNode):
                        self._generate_relation_definition(member, owner_name=store_name)
                    elif isinstance(member, ast_nodes.CastDefinitionNode):
                        self._generate_cast_definition(member, owner_name=store_name)
                    elif isinstance(member, ast_nodes.ReceiveHandlerNode):
                        self._generate_receive_handler(member, owner_name=store_name)
                    else:
                        print(f"Warning: Unsupported member type {type(member).__name__} in store '{store_name}'.")

            field_summary_parts = []
            if self.current_store_fields:
                for field_name, default_val_ir in self.current_store_fields:
                    field_summary_parts.append(f"{field_name} ({'has default IR' if default_val_ir is not None else 'no default'})")

            summary_message = f"Store '{store_name}' defined"
            if field_summary_parts:
                summary_message += f" with fields: {', '.join(field_summary_parts)}."
            else:
                summary_message += " with no fields."
            print(summary_message)

        finally:
            if self.current_module_prefix and self.current_module_prefix[-1] == store_name:
                self.current_module_prefix.pop()
            self.current_store_fields = original_store_fields_backup
            self.defining_object_or_store_name = original_defining_object_name

    def _generate_error_handler_suffix(self, node: ast_nodes.ErrorHandlerSuffixNode, main_expr_value: llvmlite.ir.Value) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or function not initialized for error handler suffix.")

        current_llvm_func = self.current_llvm_function

        error_occurred_block = current_llvm_func.append_basic_block(name="err.handler")
        no_error_block = current_llvm_func.append_basic_block(name="err.no_error")
        merge_block = current_llvm_func.append_basic_block(name="err.merge")

        tag = rt.get_coral_value_type_tag(self.current_builder, main_expr_value)
        is_error = self.current_builder.icmp_unsigned('==', tag, rt.TYPE_TAG_ERROR, name="is_error_check")
        self.current_builder.cbranch(is_error, error_occurred_block, no_error_block)

        # No Error Block
        self.current_builder.position_at_end(no_error_block)
        val_if_no_error = main_expr_value
        no_error_pred_block = self.current_builder.block
        if not no_error_block.is_terminated:
             self.current_builder.branch(merge_block)

        # Error Occurred Block
        self.current_builder.position_at_end(error_occurred_block)
        val_if_error_handled: llvmlite.ir.Value = main_expr_value
        error_path_terminates_early = False

        if node.error_variable:
            var_name = node.error_variable.name
            var_storage_ptr = self._get_variable_storage(var_name)
            if var_storage_ptr is None:
                entry_b = self.current_llvm_function.entry_basic_block
                entry_builder = IRBuilder(entry_b)
                if entry_b.instructions: entry_builder.position_before(entry_b.instructions[0])
                var_storage_ptr = entry_builder.alloca(rt.CoralValuePtrType, name=f"{var_name}_err_var_storage")
                self._set_variable_storage(var_name, var_storage_ptr)
            self.current_builder.store(main_expr_value, var_storage_ptr)

        action_node = node.action

        if isinstance(action_node, ast_nodes.ReturnStatementNode):
            self._generate_return_statement(action_node)
            error_path_terminates_early = True
        elif isinstance(action_node, ast_nodes.ASTNode) and not isinstance(action_node, list): # ExpressionNode
            val_if_error_handled = self._generate_expression(action_node)
        elif isinstance(action_node, list): # Block of statements
            self._generate_statement_list(action_node)
            val_if_error_handled = rt.create_null_value(self.current_builder)
        else:
            print(f"Warning: Unknown action type in error handler: {type(action_node)}")
            error_path_terminates_early = True

        error_pred_block = self.current_builder.block
        if not self.current_builder.block.is_terminated and not error_path_terminates_early:
            self.current_builder.branch(merge_block)

        # Merge Block
        self.current_builder.position_at_end(merge_block)

        if not merge_block.predecessors:
            return main_expr_value

        phi_node = self.current_builder.phi(rt.CoralValuePtrType, name="err_handler_res")

        if no_error_pred_block.terminator and any(succ == merge_block for succ in no_error_pred_block.terminator.successors):
             phi_node.add_incoming(val_if_no_error, no_error_pred_block)

        if not error_path_terminates_early and error_pred_block.terminator and \
           any(succ == merge_block for succ in error_pred_block.terminator.successors):
            if 'val_if_error_handled' not in locals():
                 val_if_error_handled = rt.create_null_value(self.current_builder)
            phi_node.add_incoming(val_if_error_handled, error_pred_block)

        if not phi_node.incoming:
             if len(merge_block.predecessors) == 1:
                 if merge_block.predecessors[0] == no_error_pred_block: return val_if_no_error
                 if merge_block.predecessors[0] == error_pred_block and not error_path_terminates_early:
                     return val_if_error_handled if 'val_if_error_handled' in locals() else rt.create_null_value(self.current_builder)
             if merge_block.predecessors:
                 print("Warning: err_handler_res PHI node has predecessors but no incoming values were added. Returning null.")
                 return rt.create_null_value(self.current_builder)
             return main_expr_value

        return phi_node

    def _generate_return_statement(self, node: ast_nodes.ReturnStatementNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Return statement generated outside of a function context.")
        if node.value:
            return_val_ptr = self._generate_expression(node.value)
            self.current_builder.ret(return_val_ptr)
        else:
            null_coral_val = rt.create_null_value(self.current_builder)
            self.current_builder.ret(null_coral_val)

    def _generate_function_definition(self, node: ast_nodes.FunctionDefinitionNode):
        original_func_name = node.name.name
        mangled_func_name = original_func_name
        # If not called from method def, apply module prefix. Method defs have obj_name already prefixed.
        is_method_call = self.defining_object_or_store_name and original_func_name.startswith(self.defining_object_or_store_name + "_")

        if not is_method_call:
             mangled_func_name = self._get_mangled_name(original_func_name)

        outer_builder = self.current_builder
        outer_llvm_function = self.current_llvm_function
        outer_symbol_table = self.current_symbol_table

        param_types = [rt.CoralValuePtrType] * len(node.params) # Assumes all params are CoralValue*
        return_type = rt.CoralValuePtrType
        func_type = llvmlite.ir.FunctionType(return_type, param_types)

        llvm_func = Function(self.module, func_type, name=mangled_func_name)
        self.current_llvm_function = llvm_func
        self.current_symbol_table = {}

        entry_block = llvm_func.append_basic_block(name="entry")
        self.current_builder = IRBuilder(entry_block)

        for i, param_node in enumerate(node.params):
            param_name = param_node.name.name
            llvm_arg = llvm_func.args[i]
            llvm_arg.name = param_name
            param_storage_ptr = self.current_builder.alloca(rt.CoralValuePtrType, name=f"{param_name}_param_storage")

            final_value_to_store = llvm_arg # Default to using the passed argument.

            if param_node.default_value:
                default_value_cv_ptr = self._generate_expression(param_node.default_value)
                # TODO: Implement conditional logic here.
                # If llvm_arg is a special marker indicating "use default", then:
                # final_value_to_store = default_value_cv_ptr
                # This requires defining the marker and updating call sites.
                # For now, the generated default_value_cv_ptr is not conditionally used to overwrite llvm_arg,
                # but its generation is tested.
                # As a temporary measure, to make it somewhat testable if we imagine a convention,
                # let's assume if a null is passed for an arg with a default, the default is used.
                # This is NOT a robust solution.
                if rt.CoralValuePtrType == llvm_arg.type: # Ensure it's a CoralValue*
                    is_passed_arg_null_cv = self.current_builder.call(
                        rt.get_runtime_function("coral_runtime_is_null"), [llvm_arg], name=f"is_null_{param_name}_cv"
                    )
                    require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
                    checked_is_null_cv = self.current_builder.call(require_bool_fn, [is_passed_arg_null_cv], name=f"checked_is_null_{param_name}_cv")

                    # Handle potential error from require_boolean
                    bool_extract_block = self.current_llvm_function.append_basic_block(name=f"param_{param_name}_default_bool_extract")
                    self._handle_possible_error_value(checked_is_null_cv, bool_extract_block, name_prefix=f"param_{param_name}_default_check")

                    # If we reach here, checked_is_null_cv was not an error. Position builder for the 'then' part.
                    self.current_builder.position_at_end(bool_extract_block)
                    is_passed_arg_null_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_is_null_cv)

                    # Conditional store based on the boolean value
                    # Need to ensure current block is not terminated before if_then
                    if not self.current_builder.block.is_terminated:
                        # Create blocks for the if/else of the default value assignment
                        then_block_default = self.current_llvm_function.append_basic_block(name=f"param_{param_name}_use_default")
                        else_block_default = self.current_llvm_function.append_basic_block(name=f"param_{param_name}_use_arg")
                        merge_block_default = self.current_llvm_function.append_basic_block(name=f"param_{param_name}_store_done")

                        self.current_builder.cbranch(is_passed_arg_null_i1, then_block_default, else_block_default)

                        self.current_builder.position_at_end(then_block_default)
                        self.current_builder.store(default_value_cv_ptr, param_storage_ptr)
                        if not self.current_builder.block.is_terminated: self.current_builder.branch(merge_block_default)

                        self.current_builder.position_at_end(else_block_default)
                        self.current_builder.store(llvm_arg, param_storage_ptr)
                        if not self.current_builder.block.is_terminated: self.current_builder.branch(merge_block_default)

                        self.current_builder.position_at_end(merge_block_default)

                    self._set_variable_storage(param_name, param_storage_ptr) # This might need to be inside then/else if store happens there
                                                                             # Or after merge if PHI is used. For direct store, it's fine here if merge is always reached.
                                                                             # Given the structure, if error occurs, this part is skipped.
                    continue # Skip the final store below as we've handled it.

            # This final_value_to_store will only be stored if the default value logic was NOT taken OR there was no default.
            # The 'continue' inside the if block ensures that.
            self.current_builder.store(final_value_to_store, param_storage_ptr)
                    with self.current_builder.else_():
                        self.current_builder.store(llvm_arg, param_storage_ptr)
                    self._set_variable_storage(param_name, param_storage_ptr)
                    continue # Skip the store below as we've handled it.

            # If no default or the "magic null" condition wasn't met, store the original llvm_arg.
            self.current_builder.store(final_value_to_store, param_storage_ptr)
            self._set_variable_storage(param_name, param_storage_ptr)

        # Refined body generation
        if not isinstance(node.body, list): # Single expression body
            if isinstance(node.body, ast_nodes.ASTNode):
                return_val_cv_ptr = self._generate_expression(node.body)
                if not self.current_builder.block.is_terminated:
                    self.current_builder.ret(return_val_cv_ptr)
            elif node.body is None: # No body
                 if not self.current_builder.block.is_terminated:
                    self.current_builder.ret(rt.create_null_value(self.current_builder))
            else:
                 print(f"Error: Unsupported function body type for '{mangled_func_name}': {type(node.body)}.")
                 if not self.current_builder.block.is_terminated:
                    self.current_builder.ret(rt.create_null_value(self.current_builder))
        else: # List of statements
            self._generate_statement_list(node.body)
            if not self.current_builder.block.is_terminated: # Default null return if block doesn't end with ret
                if return_type == rt.CoralValuePtrType:
                    self.current_builder.ret(rt.create_null_value(self.current_builder))
                elif return_type == VoidType():
                    self.current_builder.ret_void()

        self.current_llvm_function = outer_llvm_function
        self.current_symbol_table = outer_symbol_table
        self.current_builder = outer_builder

    def _generate_while_loop(self, node: ast_nodes.WhileLoopNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for while loop.")

        current_llvm_func = self.current_llvm_function

        loop_header_block = current_llvm_func.append_basic_block(name="while.header")
        loop_body_block = current_llvm_func.append_basic_block(name="while.body")
        loop_exit_block = current_llvm_func.append_basic_block(name="while.exit")

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_header_block)
        condition_cv = self._generate_expression(node.condition)
        require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
        checked_condition_cv = self.current_builder.call(require_bool_fn, [condition_cv], name="checked_while_cond_cv")

        bool_extract_block = self.current_llvm_function.append_basic_block(name="while_cond_bool_extract")
        self._handle_possible_error_value(checked_condition_cv, bool_extract_block, name_prefix="while_cond")
        self.current_builder.position_at_end(bool_extract_block)

        if not self.current_builder.block.is_terminated: # Check if block already terminated by error handler
            boolean_condition_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_condition_cv)
            self.current_builder.cbranch(boolean_condition_i1, loop_body_block, loop_exit_block)

        self.current_builder.position_at_end(loop_body_block) # This might be an orphan if error occurred.
                                                            # Caller needs to ensure loop_body_block is correctly populated
                                                            # or that control flow makes sense if error occurs in header.
                                                            # If an error occurs, loop_body_block will not be entered from this path.
        self._generate_statement_list(node.body if isinstance(node.body, list) else [node.body])
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_exit_block)

    def _generate_iterate_loop(self, node: ast_nodes.IterateLoopNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for iterate loop.")

        current_llvm_func = self.current_llvm_function
        iterable_val_ptr = self._generate_expression(node.iterable)

        iter_start_func = rt.get_runtime_function("coral_runtime_iterate_start")
        iterator_state_ptr = self.current_builder.call(iter_start_func, [iterable_val_ptr], name="iter_state")

        loop_header_block = current_llvm_func.append_basic_block(name="iter.header")
        loop_body_block = current_llvm_func.append_basic_block(name="iter.body")
        loop_exit_block = current_llvm_func.append_basic_block(name="iter.exit")

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_header_block)
        iter_has_next_func = rt.get_runtime_function("coral_runtime_iterate_has_next")
        has_next_cv = self.current_builder.call(iter_has_next_func, [iterator_state_ptr], name="iter_has_next_cv")

        require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
        checked_has_next_cv = self.current_builder.call(require_bool_fn, [has_next_cv], name="checked_has_next_cv")

        bool_extract_block = self.current_llvm_function.append_basic_block(name="iter_cond_bool_extract")
        self._handle_possible_error_value(checked_has_next_cv, bool_extract_block, name_prefix="iter_cond")
        self.current_builder.position_at_end(bool_extract_block)

        if not self.current_builder.block.is_terminated:
            boolean_has_next_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_has_next_cv)
            self.current_builder.cbranch(boolean_has_next_i1, loop_body_block, loop_exit_block)

        self.current_builder.position_at_end(loop_body_block) # Potentially orphaned if error in header
        iter_next_func = rt.get_runtime_function("coral_runtime_iterate_next")
        current_element_ptr = self.current_builder.call(iter_next_func, [iterator_state_ptr], name="iter_elem")

        if node.loop_variable:
            var_name = node.loop_variable.name
            var_storage_ptr = self._get_variable_storage(var_name)
            if var_storage_ptr is None:
                entry_b = current_llvm_func.entry_basic_block
                entry_builder = IRBuilder(entry_b)
                if entry_b.instructions: entry_builder.position_before(entry_b.instructions[0])
                var_storage_ptr = entry_builder.alloca(rt.CoralValuePtrType, name=f"{var_name}_iter_storage")
                self._set_variable_storage(var_name, var_storage_ptr)
            self.current_builder.store(current_element_ptr, var_storage_ptr)

        body_stmts = node.body if isinstance(node.body, list) else [node.body]
        self._generate_statement_list(body_stmts)

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_exit_block)

    def _generate_until_loop(self, node: ast_nodes.UntilLoopNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for until loop.")

        current_llvm_func = self.current_llvm_function

        loop_header_block = current_llvm_func.append_basic_block(name="until.header")
        loop_body_block = current_llvm_func.append_basic_block(name="until.body")
        loop_exit_block = current_llvm_func.append_basic_block(name="until.exit")

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_header_block)
        condition_cv = self._generate_expression(node.condition)
        require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
        checked_condition_cv = self.current_builder.call(require_bool_fn, [condition_cv], name="checked_until_cond_cv")

        bool_extract_block = self.current_llvm_function.append_basic_block(name="until_cond_bool_extract")
        self._handle_possible_error_value(checked_condition_cv, bool_extract_block, name_prefix="until_cond")
        self.current_builder.position_at_end(bool_extract_block)

        if not self.current_builder.block.is_terminated:
            boolean_condition_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_condition_cv)
            self.current_builder.cbranch(boolean_condition_i1, loop_exit_block, loop_body_block)

        self.current_builder.position_at_end(loop_body_block) # Potentially orphaned
        self._generate_statement_list(node.body if isinstance(node.body, list) else [node.body])
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_exit_block)

    def _generate_use_statement(self, node: ast_nodes.UseStatementNode):
        qname_parts = [part.name for part in node.qualified_identifier.parts]
        qname_str = ".".join(qname_parts)
        alias_str = f" as {node.alias.name}" if hasattr(node, 'alias') and node.alias else ""
        print(f"Info: 'use {qname_str}{alias_str}' noted. No direct IR generation for 'use' currently.")
        pass

    def _generate_empty_statement(self, node: ast_nodes.EmptyStatementNode):
        pass

    def _generate_module_definition(self, node: ast_nodes.ModuleDefinitionNode):
        self.current_module_prefix.append(node.name.name)
        try:
            self._generate_statement_list(node.body)
        finally:
            if self.current_module_prefix and self.current_module_prefix[-1] == node.name.name:
                self.current_module_prefix.pop()

    def _generate_unless_statement(self, node: ast_nodes.UnlessStatementNode): # Prefix unless
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for unless statement.")

        current_llvm_func = self.current_llvm_function

        body_block = current_llvm_func.append_basic_block(name="unless.body")
        end_block = current_llvm_func.append_basic_block(name="unless.end")

        condition_cv = self._generate_expression(node.condition)
        require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")
        checked_condition_cv = self.current_builder.call(require_bool_fn, [condition_cv], name="checked_unless_cond_cv")

        bool_extract_block = self.current_llvm_function.append_basic_block(name="unless_cond_bool_extract")
        self._handle_possible_error_value(checked_condition_cv, bool_extract_block, name_prefix="unless_cond")
        self.current_builder.position_at_end(bool_extract_block)

        if not self.current_builder.block.is_terminated:
            boolean_condition_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_condition_cv)
            self.current_builder.cbranch(boolean_condition_i1, end_block, body_block)

        self.current_builder.position_at_end(body_block) # Potentially orphaned
        self._generate_statement_list(node.block if isinstance(node.block, list) else [node.block])
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(end_block)

        self.current_builder.position_at_end(end_block)


    def _generate_if_then_else(self, node: ast_nodes.IfThenElseStatementNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for if/else.")

        current_llvm_func = self.current_llvm_function
        require_bool_fn = rt.get_runtime_function("coral_runtime_require_boolean")

        condition_cv = self._generate_expression(node.condition)
        checked_condition_cv = self.current_builder.call(require_bool_fn, [condition_cv], name="checked_if_cond_cv")

        # Create success block for initial if condition
        if_cond_success_block = current_llvm_func.append_basic_block(name="if_cond_success")
        self._handle_possible_error_value(checked_condition_cv, if_cond_success_block, name_prefix="if_cond")
        self.current_builder.position_at_end(if_cond_success_block)

        # These blocks might be created after current_builder's block is already terminated by error handling.
        then_block = current_llvm_func.append_basic_block(name="if.then")
        else_if_cond_blocks = [current_llvm_func.append_basic_block(name=f"elseif.cond.{i}") for i, _ in enumerate(node.else_if_clauses)]
        else_if_body_blocks = [current_llvm_func.append_basic_block(name=f"elseif.body.{i}") for i, _ in enumerate(node.else_if_clauses)]
        else_block_ast_stmts = node.else_block
        else_block_llvm = current_llvm_func.append_basic_block(name="if.else") if else_block_ast_stmts else None
        merge_block = current_llvm_func.append_basic_block(name="if.merge")

        if not self.current_builder.block.is_terminated: # Only proceed if initial condition didn't error out
            boolean_condition_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_condition_cv)
            next_block_after_if_cond_false = else_if_cond_blocks[0] if else_if_cond_blocks else (else_block_llvm if else_block_llvm else merge_block)
            self.current_builder.cbranch(boolean_condition_i1, then_block, next_block_after_if_cond_false)
        # If initial condition errored, if_cond_success_block is empty and terminated by _handle_possible_error_value.
        # The then_block and subsequent else path might become detached if not handled carefully.
        # The key is that if an error occurs, the function returns early.

        # Populate then_block (only if reachable)
        self.current_builder.position_at_end(then_block) # Position builder, but block might be unreachable
        if then_block.predecessors: # Check if actually reachable
            self._generate_statement_list(node.if_block if isinstance(node.if_block, list) else [node.if_block])
            if not self.current_builder.block.is_terminated:
                self.current_builder.branch(merge_block)

        for i, clause in enumerate(node.else_if_clauses):
            self.current_builder.position_at_end(else_if_cond_blocks[i])
            if not else_if_cond_blocks[i].predecessors and i > 0 and not else_if_cond_blocks[i-1].is_terminated : # if previous block didn't lead here
                 # This check is tricky. If previous elseif errored, this block might be detached.
                 # For now, assume if we position here, it's meant to be populated if reachable.
                 pass

            if not self.current_builder.block.is_terminated: # Only process if this cond block is reachable
                elseif_cond_cv = self._generate_expression(clause['condition'])
                checked_elseif_cond_cv = self.current_builder.call(require_bool_fn, [elseif_cond_cv], name=f"checked_elseif_cond_{i}_cv")

                elseif_cond_success_block = current_llvm_func.append_basic_block(name=f"elseif_cond_{i}_success")
                self._handle_possible_error_value(checked_elseif_cond_cv, elseif_cond_success_block, name_prefix=f"elseif_cond_{i}")
                self.current_builder.position_at_end(elseif_cond_success_block)

                if not self.current_builder.block.is_terminated:
                    elseif_boolean_condition_i1 = rt.unsafe_get_boolean_value(self.current_builder, checked_elseif_cond_cv)
                    next_block_after_elseif_cond_false = else_if_cond_blocks[i+1] if (i+1) < len(else_if_cond_blocks) else (else_block_llvm if else_block_llvm else merge_block)
                    self.current_builder.cbranch(elseif_boolean_condition_i1, else_if_body_blocks[i], next_block_after_elseif_cond_false)

            self.current_builder.position_at_end(else_if_body_blocks[i])
            if else_if_body_blocks[i].predecessors: # Check if actually reachable
                self._generate_statement_list(clause['block'] if isinstance(clause['block'], list) else [clause['block']])
                if not self.current_builder.block.is_terminated:
                    self.current_builder.branch(merge_block)

        if else_block_llvm:
            self.current_builder.position_at_end(else_block_llvm)
            if else_block_llvm.predecessors: # Check if actually reachable
                self._generate_statement_list(else_block_ast_stmts if isinstance(else_block_ast_stmts, list) else [else_block_ast_stmts])
                if not self.current_builder.block.is_terminated:
                    self.current_builder.branch(merge_block)

        self.current_builder.position_at_end(merge_block)
        # If merge_block has no predecessors, it means all paths returned or were otherwise terminated.
        # In such a case, trying to insert further instructions here would be problematic.
        # So, only proceed if merge_block is actually reachable.
        if not merge_block.predecessors:
            # This block might be orphaned if all preceding paths terminated (e.g., due to error returns).
            # We should ensure it's properly handled or removed if truly unreachable.
            # For now, if it has no predecessors, we assume current path is effectively dead from here.
            # No further instructions should be added to merge_block unless it's reachable.
            # If current_builder is already on a terminated block, this new positioning might be on an unreachable one.
            # However, llvmlite handles appending to an existing block. If it's unreachable, these ops are nops.
            pass # Builder is at merge_block. If it has no predecessors, it's likely dead code.
        # The original code:
        # if merge_block.predecessors:
        #    self.current_builder.position_at_end(merge_block)
        # else:
        #    pass
        # This is effectively the same as just self.current_builder.position_at_end(merge_block)
        # as llvmlite will create it if it doesn't exist, or allow appending.
        # The critical part is that subsequent code should not assume this block is reachable
        # if all paths to it might have been terminated by error returns.
            self._generate_statement_list(clause['block'] if isinstance(clause['block'], list) else [clause['block']])
            if not self.current_builder.block.is_terminated:
                self.current_builder.branch(merge_block)

        if else_block_llvm:
            self.current_builder.position_at_end(else_block_llvm)
            self._generate_statement_list(else_block_ast_stmts if isinstance(else_block_ast_stmts, list) else [else_block_ast_stmts])
            if not self.current_builder.block.is_terminated:
                self.current_builder.branch(merge_block)

        if merge_block.predecessors:
            self.current_builder.position_at_end(merge_block)
        else:
            # If merge_block has no predecessors, it implies all paths returned or diverged.
            # No need to position the builder here as it's effectively dead code.
            pass


    def generate(self, program_node: ast_nodes.ProgramNode):
        if not isinstance(program_node, ast_nodes.ProgramNode):
            raise TypeError("Expected ProgramNode")

        main_func_type = FunctionType(VoidType(), [])
        main_llvm_func = Function(self.module, main_func_type, name="main")

        # Setup main function context using ModuleContext
        entry_block = main_llvm_func.append_basic_block(name="entry")
        builder = IRBuilder(entry_block)

        # Backup outer context (if any, though for top-level generate, these are usually None)
        outer_builder = self.current_builder
        outer_llvm_function = self.current_llvm_function
        outer_symbol_table = self.current_symbol_table

        # Set new context for main function
        self.current_builder = builder
        self.current_llvm_function = main_llvm_func
        # Main function's symbols could be global or a fresh scope.
        # Using global_symbol_table from ModuleContext for top-level functions.
        self.current_symbol_table = self.module_context.global_symbol_table

        self._generate_statement_list(program_node.body)

        if not self.current_builder.block.is_terminated:
            self.current_builder.ret_void()

        # Restore outer context
        self.current_builder = outer_builder
        self.current_llvm_function = outer_llvm_function
        self.current_symbol_table = outer_symbol_table
        return self.module

[end of ir_generator.py]
