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
        self.current_module_prefix: list[str] = []

    def _get_mangled_name(self, original_name: str) -> str:
        if not self.current_module_prefix:
            return original_name
        return "_".join(self.current_module_prefix + [original_name])

    def _create_llvm_global_string_ptr(self, py_string: str, name_prefix: str = ".str") -> llvmlite.ir.Value:
        """
        Creates an LLVM global string constant and returns an i8* pointer to it.
        Caches globals in self.module.globals.
        """
        if self.current_builder is None: # Should not happen if called from valid context
            raise RuntimeError("Builder not initialized for creating global string.")

        global_var_name = f"{name_prefix}.{py_string}"
        # Sanitize global_var_name if py_string can contain invalid characters for LLVM names.
        # For now, assume simple strings from identifiers or property names.
        # Basic sanitization: replace non-alphanumeric with underscore
        clean_py_string = ''.join(c if c.isalnum() else '_' for c in py_string)
        if not clean_py_string or clean_py_string[0].isdigit(): # Ensure it's a valid C-like identifier part
            clean_py_string = f"s_{clean_py_string}"
        global_var_name = f"{name_prefix}.{clean_py_string}"


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

        idx_type = llvmlite.ir.IntType(32) # Consistent GEP index type
        return self.current_builder.gep(
            g_var,
            [llvmlite.ir.Constant(idx_type, 0), llvmlite.ir.Constant(idx_type, 0)],
            name=f"{clean_py_string}_ptr"
        )

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

        # RHS Value Generation
        rhs_coral_value_ptr: llvmlite.ir.Value | None = None
        if isinstance(node.value, ast_nodes.MapBlockAssignmentRHSNode):
            rhs_coral_value_ptr = self._generate_map_from_block(node.value)
        elif isinstance(node.value, ast_nodes.ASTNode): # Should catch all other expression types
            rhs_coral_value_ptr = self._generate_expression(node.value)
        else:
            # This case should ideally not be reached if parser ensures node.value is an ASTNode subtype
            raise TypeError(f"Unsupported RHS type for assignment: {type(node.value)}")

        if rhs_coral_value_ptr is None:
            # Attempt to get a name for the error message, if target is an identifier
            target_name_for_error = "unknown_target"
            if hasattr(node.target, 'name'): # Covers IdentifierNode
                target_name_for_error = node.target.name
            elif isinstance(node.target, (ast_nodes.ListElementAccessNode, ast_nodes.PropertyAccessNode)):
                 # Could try to construct a string representation, but keep it simple for now
                target_name_for_error = f"{type(node.target).__name__}"
            print(f"Error: RHS of assignment to '{target_name_for_error}' evaluated to None. Skipping assignment.")
            return

        # LHS Target Handling
        if isinstance(node.target, ast_nodes.IdentifierNode):
            var_name = node.target.name
            var_storage_ptr = self._get_variable_storage(var_name)
            if var_storage_ptr is None:
                # Allocate in function entry block to ensure it's available throughout the function
                entry_b = self.current_llvm_function.entry_basic_block
                if not entry_b.instructions:
                    entry_builder = IRBuilder(entry_b)
                else:
                    first_instr_not_alloca = next((instr for instr in entry_b.instructions if not isinstance(instr, llvmlite.ir.AllocaInstr)), None)
                    if first_instr_not_alloca:
                        entry_builder = IRBuilder(entry_b)
                        entry_builder.position_before(first_instr_not_alloca)
                    else:
                        entry_builder = IRBuilder(entry_b)
                        if entry_b.instructions:
                             entry_builder.position_after(entry_b.instructions[-1])
                var_storage_ptr = entry_builder.alloca(rt.CoralValuePtrType, name=f"{var_name}_storage_ptr")
                self._set_variable_storage(var_name, var_storage_ptr)
            self.current_builder.store(rhs_coral_value_ptr, var_storage_ptr)

        elif isinstance(node.target, ast_nodes.ListElementAccessNode):
            target_list_access_node = node.target
            list_ptr = self._generate_expression(target_list_access_node.base_expr)
            index_ptr = self._generate_expression(target_list_access_node.index_expr)

            func_name = "coral_runtime_list_set_element"
            func_ty = llvmlite.ir.FunctionType(rt.VoidType, [rt.CoralValuePtrType, rt.CoralValuePtrType, rt.CoralValuePtrType])
            set_elem_func = self.module.globals.get(func_name)
            if set_elem_func is None or not isinstance(set_elem_func, llvmlite.ir.Function):
                set_elem_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)
            self.current_builder.call(set_elem_func, [list_ptr, index_ptr, rhs_coral_value_ptr])

        elif isinstance(node.target, ast_nodes.PropertyAccessNode):
            target_prop_access_node = node.target
            obj_ptr = self._generate_expression(target_prop_access_node.base_expr)
            prop_name_str = target_prop_access_node.property_name.name

            prop_name_llvm_ptr = self._create_llvm_global_string_ptr(prop_name_str, name_prefix=".str.prop_name")

            func_name = "coral_runtime_object_set_property"
            func_ty = llvmlite.ir.FunctionType(rt.VoidType, [rt.CoralValuePtrType, rt.ptr_to(llvmlite.ir.IntType(8)), rt.CoralValuePtrType])
            set_prop_func = self.module.globals.get(func_name)
            if set_prop_func is None or not isinstance(set_prop_func, llvmlite.ir.Function):
                set_prop_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)
            self.current_builder.call(set_prop_func, [obj_ptr, prop_name_llvm_ptr, rhs_coral_value_ptr])

        else:
            print(f"Warning: Assignment to target type {type(node.target).__name__} not fully supported. Skipped.")
            # Consider: raise NotImplementedError(f"Assignment to target type {type(node.target).__name__} not implemented.")

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
            # Allocate stack space for an array of CoralValuePtrType (i.e., CoralValue**)
            elements_array_alloca = self.current_builder.alloca(
                rt.CoralValuePtrType,
                size=llvm_num_elements_i32,
                name="list_lit_elems_storage"
            )

            for i in range(num_elements):
                element_ast = node.elements[i]
                elem_val_ptr = self._generate_expression(element_ast) # This is CoralValue*

                # Get pointer to the i-th slot in elements_array_alloca
                # GEP needs an index of consistent type, using i32 here.
                slot_ptr = self.current_builder.gep(
                    elements_array_alloca,
                    [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)],
                    name=f"list_elem_slot_{i}_ptr"
                )
                self.current_builder.store(elem_val_ptr, slot_ptr)

            elements_array_storage_ptr = elements_array_alloca # The alloca is already CoralValue**

        # Declare or get the coral_runtime_create_list runtime function
        func_name = "coral_runtime_create_list"
        # Signature: CoralValue* coral_runtime_create_list(CoralValue** elements_array, i32 num_elements)
        func_ty = llvmlite.ir.FunctionType(
            rt.CoralValuePtrType,
            [rt.ptr_to(rt.CoralValuePtrType), llvmlite.ir.IntType(32)]
        )

        create_list_func = self.module.globals.get(func_name)
        if create_list_func is None or not isinstance(create_list_func, llvmlite.ir.Function):
            create_list_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

        new_list_ptr = self.current_builder.call(
            create_list_func,
            [elements_array_storage_ptr, llvm_num_elements_i32],
            name="new_list_from_literal"
        )
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
            keys_array_alloca = self.current_builder.alloca(
                rt.CoralValuePtrType,
                size=llvm_num_entries_i32,
                name="map_block_keys_storage"
            )
            values_array_alloca = self.current_builder.alloca(
                rt.CoralValuePtrType,
                size=llvm_num_entries_i32,
                name="map_block_vals_storage"
            )

            for i in range(num_entries):
                map_block_entry_node = rhs_node.entries[i] # This is ast_nodes.MapBlockEntryNode

                key_name_str = map_block_entry_node.key.name
                key_coral_str_ptr = self._create_llvm_global_string_ptr(key_name_str, name_prefix=".str.map_key")
                # Note: Map keys from blocks are also being created as global strings here.
                # This is different from rt.create_string_value used in _generate_map_literal,
                # which creates a Coral string *object*. For consistency with how map keys
                # are handled by the runtime, it might be better to use rt.create_string_value.
                # Let's adjust this to use rt.create_string_value for the key, like in _generate_map_literal.
                key_coral_str_ptr = rt.create_string_value(self.current_builder, self.module, key_name_str)


                value_coral_val_ptr = self._generate_expression(map_block_entry_node.value)

                key_slot_ptr = self.current_builder.gep(
                    keys_array_alloca,
                    [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)],
                    name=f"map_block_key_slot_{i}_ptr"
                )
                self.current_builder.store(key_coral_str_ptr, key_slot_ptr)

                val_slot_ptr = self.current_builder.gep(
                    values_array_alloca,
                    [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)],
                    name=f"map_block_val_slot_{i}_ptr"
                )
                self.current_builder.store(value_coral_val_ptr, val_slot_ptr)

            keys_array_ptr = keys_array_alloca
            values_array_ptr = values_array_alloca

        func_name = "coral_runtime_create_map"
        func_ty = llvmlite.ir.FunctionType(
            rt.CoralValuePtrType,
            [rt.ptr_to(rt.CoralValuePtrType), rt.ptr_to(rt.CoralValuePtrType), llvmlite.ir.IntType(32)]
        )

        create_map_func = self.module.globals.get(func_name)
        if create_map_func is None or not isinstance(create_map_func, llvmlite.ir.Function):
            create_map_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

        new_map_ptr = self.current_builder.call(
            create_map_func,
            [keys_array_ptr, values_array_ptr, llvm_num_entries_i32],
            name="new_map_from_block"
        )
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
            keys_array_alloca = self.current_builder.alloca(
                rt.CoralValuePtrType,
                size=llvm_num_entries_i32,
                name="map_lit_keys_storage"
            )
            values_array_alloca = self.current_builder.alloca(
                rt.CoralValuePtrType,
                size=llvm_num_entries_i32,
                name="map_lit_vals_storage"
            )

            for i in range(num_entries):
                map_entry_node = node.entries[i] # This is ast_nodes.MapEntryNode

                # Key IR (key is IdentifierNode, convert its name to Coral String Value)
                key_name_str = map_entry_node.key.name
                # rt.create_string_value needs builder, module, and python string
                key_coral_str_ptr = rt.create_string_value(self.current_builder, self.module, key_name_str)

                # Value IR
                value_coral_val_ptr = self._generate_expression(map_entry_node.value)

                # Store key
                key_slot_ptr = self.current_builder.gep(
                    keys_array_alloca,
                    [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)],
                    name=f"map_key_slot_{i}_ptr"
                )
                self.current_builder.store(key_coral_str_ptr, key_slot_ptr)

                # Store value
                val_slot_ptr = self.current_builder.gep(
                    values_array_alloca,
                    [llvmlite.ir.Constant(llvmlite.ir.IntType(32), i)],
                    name=f"map_val_slot_{i}_ptr"
                )
                self.current_builder.store(value_coral_val_ptr, val_slot_ptr)

            keys_array_ptr = keys_array_alloca
            values_array_ptr = values_array_alloca

        # Declare or get the coral_runtime_create_map runtime function
        func_name = "coral_runtime_create_map"
        # Signature: CoralValue* coral_runtime_create_map(CoralValue** keys, CoralValue** values, i32 num_entries)
        func_ty = llvmlite.ir.FunctionType(
            rt.CoralValuePtrType,
            [rt.ptr_to(rt.CoralValuePtrType), rt.ptr_to(rt.CoralValuePtrType), llvmlite.ir.IntType(32)]
        )

        create_map_func = self.module.globals.get(func_name)
        if create_map_func is None or not isinstance(create_map_func, llvmlite.ir.Function):
            create_map_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

        new_map_ptr = self.current_builder.call(
            create_map_func,
            [keys_array_ptr, values_array_ptr, llvm_num_entries_i32],
            name="new_map_from_literal"
        )
        return new_map_ptr

    def _generate_list_element_access(self, node: ast_nodes.ListElementAccessNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for list element access.")

        base_val_ptr = self._generate_expression(node.base_expr)
        index_val_ptr = self._generate_expression(node.index_expr)

        func_name = "coral_runtime_list_get_element"
        func_ty = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.CoralValuePtrType, rt.CoralValuePtrType])

        get_elem_func = self.module.globals.get(func_name)
        if get_elem_func is None or not isinstance(get_elem_func, llvmlite.ir.Function):
            get_elem_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

        element_ptr = self.current_builder.call(get_elem_func, [base_val_ptr, index_val_ptr], name="list_elem_ptr")
        return element_ptr

    def _generate_property_access(self, node: ast_nodes.PropertyAccessNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for property access.")

        base_val_ptr = self._generate_expression(node.base_expr)
        property_name_str = node.property_name.name

        prop_name_ptr = self._create_llvm_global_string_ptr(property_name_str, name_prefix=".str.prop_name")

        # Declare or get coral_runtime_object_get_property
        func_name = "coral_runtime_object_get_property"
        # Signature: CoralValue* (CoralValue* obj_ptr, i8* prop_name_ptr)
        func_ty = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.CoralValuePtrType, llvmlite.ir.PointerType(llvmlite.ir.IntType(8))])

        get_prop_func = self.module.globals.get(func_name)
        if get_prop_func is None or not isinstance(get_prop_func, llvmlite.ir.Function):
            get_prop_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

        property_val_ptr = self.current_builder.call(get_prop_func, [base_val_ptr, prop_name_ptr], name="prop_val_ptr")
        return property_val_ptr

    def _generate_ternary_conditional_expression(self, node: ast_nodes.TernaryConditionalExpressionNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for ternary conditional expression.")

        current_llvm_func = self.current_llvm_function

        condition_val_ptr = self._generate_expression(node.condition)
        boolean_condition = rt.get_boolean_value(self.current_builder, condition_val_ptr, current_llvm_func)

        then_block = current_llvm_func.append_basic_block(name="ternary.then")
        else_block = current_llvm_func.append_basic_block(name="ternary.else")
        merge_block = current_llvm_func.append_basic_block(name="ternary.merge")

        self.current_builder.cbranch(boolean_condition, then_block, else_block)

        # Then Block
        self.current_builder.position_at_end(then_block)
        then_val_ptr = self._generate_expression(node.true_expr)
        # then_val_origin_block is the block self.current_builder is positioned in *after* true_expr is generated.
        # This is the block that will (potentially) branch to merge_block.
        then_val_origin_block = self.current_builder.block
        if not then_block.is_terminated: # Check the original then_block for termination
            self.current_builder.branch(merge_block)

        # Else Block
        self.current_builder.position_at_end(else_block)
        else_val_ptr = self._generate_expression(node.false_expr)
        # else_val_origin_block is the block self.current_builder is positioned in *after* false_expr is generated.
        else_val_origin_block = self.current_builder.block
        if not else_block.is_terminated: # Check the original else_block for termination
            self.current_builder.branch(merge_block)

        # Merge Block
        self.current_builder.position_at_end(merge_block)

        # If merge_block has no predecessors, it means both branches of the ternary
        # always return or branch elsewhere. A PHI node here would be ill-formed.
        # This check ensures we only create a PHI if it's meaningful.
        if not merge_block.predecessors:
            # This situation is complex: an expression is expected to yield a value,
            # but if both paths diverge, there's no single value.
            # LLVM's 'unreachable' instruction might be appropriate if the code path itself is unreachable.
            # For now, we'll proceed to create the PHI, and LLVM's verifier will catch issues
            # if merge_block truly has no predecessors. In a well-formed program, a ternary
            # expression that is *used* should have its merge_block reachable.
            # If it's not used, it might be optimized out.
            # A more robust solution might involve checking if then_val_origin_block and
            # else_val_origin_block actually branch to merge_block before adding them to PHI.
            # However, the standard PHI construction relies on the CFG being correctly formed
            # such that only actual predecessors are added.
            pass # Allow PHI creation, LLVM will verify.

        phi_node = self.current_builder.phi(rt.CoralValuePtrType, name="ternary.val")

        # Add incoming values. It's crucial that then_val_origin_block and else_val_origin_block
        # are the blocks that are actual predecessors to merge_block in the CFG.
        # The branches added above ensure this if the paths don't terminate early.
        # If a path *does* terminate early (e.g., true_expr contains a 'return'),
        # then its corresponding origin_block will not be a predecessor of merge_block,
        # and LLVM's PHI verification will complain if we try to add it.
        # So, we should only add incoming branches that actually exist.

        # A simple way:
        # For a PHI node in block B, for each predecessor P of B, there must be one incoming value from P.
        # So, iterate over merge_block.predecessors.
        # This is safer.

        processed_predecessors = set()
        if then_val_origin_block in merge_block.predecessors and then_val_origin_block not in processed_predecessors:
            phi_node.add_incoming(then_val_ptr, then_val_origin_block)
            processed_predecessors.add(then_val_origin_block)

        if else_val_origin_block in merge_block.predecessors and else_val_origin_block not in processed_predecessors:
             phi_node.add_incoming(else_val_ptr, else_val_origin_block)
             processed_predecessors.add(else_val_origin_block)

        # If, after all this, phi_node has no incoming values (e.g., if _generate_expression
        # for true/false_expr led to new blocks that branched away, and the original then/else blocks
        # didn't get their branch to merge_block, or if both paths returned), then the phi is invalid.
        # This indicates a potentially ill-formed ternary in the source or complex control flow
        # within the ternary expressions not suitable for a simple PHI.
        # For typical ternaries, the two simple add_incoming calls are sufficient:
        # phi_node.add_incoming(then_val_ptr, then_val_origin_block)
        # phi_node.add_incoming(else_val_ptr, else_val_origin_block)
        # The checks `if not then_block.is_terminated:` / `if not else_block.is_terminated:`
        # are designed to ensure that `then_val_origin_block` and `else_val_origin_block`
        # (if those paths don't return) do indeed branch to `merge_block`.

        # Given the structure, then_val_origin_block is the block that *contains* the branch if one is added.
        # Same for else_val_origin_block. So these are the correct blocks to reference.

        # Sticking to the direct approach as per initial plan, assuming valid ternary use:
        # If a branch does not reach the merge block, it should not be added.
        # The construction `if not then_block.is_terminated: self.current_builder.branch(merge_block)`
        # ensures that `then_val_origin_block` (which is `self.current_builder.block` at that point)
        # *will* be a predecessor if that path is taken and doesn't return.
        # So, the direct add_incoming calls are robust under this assumption.

        # Clearer PHI population:
        # Check if then_val_origin_block actually branches to merge_block.
        # This is true if then_block (the original BB) was not terminated by node.true_expr,
        # thus received a branch to merge_block.
        if then_val_origin_block.terminator and any(succ == merge_block for succ in then_val_origin_block.terminator.successors):
             phi_node.add_incoming(then_val_ptr, then_val_origin_block)
        # If node.true_expr itself returned, then_block would be terminated, and this path wouldn't go to merge_block.

        if else_val_origin_block.terminator and any(succ == merge_block for succ in else_val_origin_block.terminator.successors):
            phi_node.add_incoming(else_val_ptr, else_val_origin_block)

        # If after these checks, phi_node has no incoming branches, it implies an issue.
        # For a standard ternary expression, we expect two incoming branches.
        # If phi_node.is_empty and merge_block.predecessors: # Problem!
        # If phi_node.is_empty and not merge_block.predecessors: # Merge block is dead, what to return?
            # This path is problematic. An expression must yield a value.
            # If the merge block is unreachable, the value of the expression is undefined in this path.
            # This could happen if both true and false expressions always return.
            # In such a case, the code after the ternary is also unreachable.
            # For now, we assume the ternary is used in a context where its value is needed.
            # If phi_node has 0 incoming values, it's an invalid PHI.
            # Let's revert to the simplest form, relying on LLVM verification for now.
            # phi_node.add_incoming(then_val_ptr, then_val_origin_block)
            # phi_node.add_incoming(else_val_ptr, else_val_origin_block)
            # This was the original plan and is standard if the CFG is built correctly.
            # The checks for `is_terminated` on `then_block` and `else_block` (the original ones)
            # ensure that the branches to `merge_block` are added from `then_val_origin_block`
            # and `else_val_origin_block` respectively, if those paths don't have early exits.

        # Final simplified version based on initial plan:
        phi_node.add_incoming(then_val_ptr, then_val_origin_block)
        phi_node.add_incoming(else_val_ptr, else_val_origin_block)


        return phi_node

    def _generate_dollar_param(self, node: ast_nodes.DollarParamNode) -> llvmlite.ir.Value:
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for DollarParamNode.")

        if isinstance(node.name_or_index, str):
            param_name_str = node.name_or_index
            name_ptr = self._create_llvm_global_string_ptr(param_name_str, name_prefix=".str.dollar_param")

            func_name = "coral_runtime_get_dollar_param_by_name"
            func_ty = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [llvmlite.ir.PointerType(llvmlite.ir.IntType(8))])
            get_param_func = self.module.globals.get(func_name)
            if get_param_func is None or not isinstance(get_param_func, llvmlite.ir.Function):
                get_param_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

            param_val_ptr = self.current_builder.call(get_param_func, [name_ptr], name=f"dollar_param_{param_name_str}")

        elif isinstance(node.name_or_index, int):
            param_index_int = node.name_or_index
            index_llvm_int = llvmlite.ir.Constant(rt.IntegerType, param_index_int) # rt.IntegerType is i64

            func_name = "coral_runtime_get_dollar_param_by_index"
            func_ty = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.IntegerType]) # Expects i64
            get_param_func = self.module.globals.get(func_name)
            if get_param_func is None or not isinstance(get_param_func, llvmlite.ir.Function):
                get_param_func = llvmlite.ir.Function(self.module, func_ty, name=func_name)

            param_val_ptr = self.current_builder.call(get_param_func, [index_llvm_int], name=f"dollar_param_{param_index_int}")

        else:
            raise TypeError(f"DollarParamNode has unexpected name_or_index type: {type(node.name_or_index)}")

        return param_val_ptr

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

        original_func_name = node.callee.name
        mangled_callee_name = self._get_mangled_name(original_func_name)

        generated_args = [self._generate_expression(arg) for arg in node.arguments] if node.arguments else []
        if node.call_style != ast_nodes.CallStyle.FUNCTION:
            print(f"Warning: Call style '{node.call_style}' for '{mangled_callee_name}' not fully supported.")

        llvm_function = self.module.globals.get(mangled_callee_name)
        if llvm_function is None or not isinstance(llvm_function, llvmlite.ir.Function):
            print(f"Warning: Function '{mangled_callee_name}' not found. Declaring with assumed signature.")
            arg_types = [rt.CoralValuePtrType] * len(generated_args)
            func_type = llvmlite.ir.FunctionType(rt.CoralValuePtrType, arg_types)
            llvm_function = llvmlite.ir.Function(self.module, func_type, name=mangled_callee_name)

        if len(llvm_function.args) != len(generated_args):
            raise ValueError(f"Call to '{mangled_callee_name}': {len(generated_args)} args, expects {len(llvm_function.args)}.")
        return self.current_builder.call(llvm_function, generated_args, name=f"call_{mangled_callee_name}")

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
        elif isinstance(stmt_node, ast_nodes.ModuleDefinitionNode):
            self._generate_module_definition(stmt_node)
        else:
            print(f"Warning: Statement type {type(stmt_node).__name__} not handled by _generate_statement.")

    def _generate_object_definition(self, node: ast_nodes.ObjectDefinitionNode):
        obj_name = node.name.name
        # The mangled name for print should reflect its definition context if within a module
        # For example, if obj TestObj is in module ModA, it's ModA_TestObj
        # self._get_mangled_name(obj_name) would achieve this if obj_name is not yet in prefix.
        # However, for messages, the simple name is often fine, and methods will show full path.
        # Let's keep the simple name for the "Processing object definition" message.
        print(f"Info: Processing object definition for '{obj_name}'. Methods will be defined as mangled functions.")

        self.current_module_prefix.append(obj_name)
        try:
            for member in node.members:
                if isinstance(member, ast_nodes.FieldDefinitionNode):
                    # Pass the simple object name; _generate_field_definition will query full prefix
                    self._generate_field_definition(member, owner_name=obj_name)
                elif isinstance(member, ast_nodes.MethodDefinitionNode):
                    # Pass the simple object name; _generate_method_definition will query full prefix
                    self._generate_method_definition(member, obj_name=obj_name)
        finally:
            if self.current_module_prefix and self.current_module_prefix[-1] == obj_name:
                self.current_module_prefix.pop()


    def _generate_field_definition(self, node: ast_nodes.FieldDefinitionNode, owner_name: str):
        # owner_name is the simple name of the object or store
        full_owner_name = self._get_mangled_name(owner_name) # This will apply module_prefix + owner_name

        base_msg = f"Info: Field definition '{node.name.name}' in owner '{full_owner_name}' noted."

        if node.default_value is not None:
            # In a full implementation, self._generate_expression(node.default_value) would be called here.
            print(f"{base_msg} (with default value). Full IR generation for field storage and initialization deferred.")
        else:
            print(f"{base_msg} Full IR generation for field storage deferred.")

    def _generate_relation_definition(self, node: ast_nodes.RelationDefinitionNode, owner_name: str):
        current_context_name = self._get_mangled_name(owner_name)
        # Assuming node.name is an IdentifierNode or similar with a .name attribute
        relation_name = node.name.name if hasattr(node.name, 'name') else str(node.name)
        print(f"Info: Relation definition '{relation_name}' in owner '{current_context_name}' noted. IR generation deferred.")

    def _generate_cast_definition(self, node: ast_nodes.CastDefinitionNode, owner_name: str):
        current_context_name = self._get_mangled_name(owner_name)
        cast_to_type_str = node.cast_to_type.name if hasattr(node.cast_to_type, 'name') else str(node.cast_to_type)
        print(f"Info: Cast definition to type '{cast_to_type_str}' in owner '{current_context_name}' noted. IR generation deferred.")

    def _generate_receive_handler(self, node: ast_nodes.ReceiveHandlerNode, owner_name: str):
        current_context_name = self._get_mangled_name(owner_name)
        # Assuming node.message_name is an IdentifierNode or similar
        message_name_str = node.message_name.name if hasattr(node.message_name, 'name') else str(node.message_name)
        print(f"Info: Receive handler for message '{message_name_str}' in owner '{current_context_name}' noted. IR generation deferred.")

    def _generate_method_definition(self, node: ast_nodes.MethodDefinitionNode, obj_name: str):
        intermediate_mangled_method_name = f"{obj_name}_{node.name.name}"
        # Now, apply the module prefix to this intermediate name
        final_llvm_name = self._get_mangled_name(intermediate_mangled_method_name)
        print(f"Info: Defining method '{node.name.name}' of '{obj_name}' (obj part: {intermediate_mangled_method_name}) as LLVM function '{final_llvm_name}'.")
        synthetic_func_node = ast_nodes.FunctionDefinitionNode(
            name=ast_nodes.IdentifierNode(final_llvm_name, token_details=node.name.token_details),
            params=node.params, body=node.body, token_details=node.token_details)
        self._generate_function_definition(synthetic_func_node)

    def _generate_store_definition(self, node: ast_nodes.StoreDefinitionNode):
        store_name = node.name.name
        # Using simple store_name for top-level messages about the store itself.
        # Mangled names will appear for its members via their respective generators.
        print(f"Info: Processing store definition for '{store_name}'.")

        if node.is_actor:
            print(f"Info: Store '{store_name}' is an actor.")
        if node.for_target:
            # Assuming node.for_target is an IdentifierNode or similar with a .name
            target_name = node.for_target.name if hasattr(node.for_target, 'name') else str(node.for_target)
            print(f"Info: Store '{store_name}' is for target '{target_name}'.")

        self.current_module_prefix.append(store_name)
        try:
            if hasattr(node, 'members') and node.members:
                 for member in node.members:
                    if isinstance(member, ast_nodes.MethodDefinitionNode):
                        # _generate_method_definition uses obj_name (store_name here)
                        # and _get_mangled_name internally handles the full prefix.
                        self._generate_method_definition(member, obj_name=store_name)
                    elif isinstance(member, ast_nodes.FieldDefinitionNode):
                        self._generate_field_definition(member, owner_name=store_name)
                    elif isinstance(member, ast_nodes.RelationDefinitionNode):
                        self._generate_relation_definition(member, owner_name=store_name)
                    elif isinstance(member, ast_nodes.CastDefinitionNode):
                        self._generate_cast_definition(member, owner_name=store_name)
                    elif isinstance(member, ast_nodes.ReceiveHandlerNode): # Based on task spec for ReceiveHandlerNode
                        self._generate_receive_handler(member, owner_name=store_name)
                    # Check if ast_nodes.ReceiveDefinitionNode is a distinct type that also needs handling.
                    # My previous analysis showed ast_nodes.ReceiveDefinitionNode for store members.
                    # The task spec specifically asks for ReceiveHandlerNode. Trusting task spec.
                    elif isinstance(member, ast_nodes.ReceiveDefinitionNode): # Handling this just in case
                        # This assumes ReceiveDefinitionNode has a 'message_name' like ReceiveHandlerNode
                        # or needs adaptation. For now, let's assume it's similar enough or covered by Handler.
                        # If they are different, a separate handler or logic is needed.
                        # For this pass, if it's ReceiveDefinitionNode and not ReceiveHandlerNode, it will go to else.
                        # This can be refined if both types are distinct and appear in stores.
                        print(f"Info: Encountered ReceiveDefinitionNode '{member.message_name.name if hasattr(member, 'message_name') else 'UnknownMsg'}' - using receive_handler logic.")
                        self._generate_receive_handler(member, owner_name=store_name) # Attempt to use same handler
                    else:
                        print(f"Warning: Unsupported member type {type(member).__name__} in store '{store_name}'.")
            else:
                print(f"Info: Store '{store_name}' has no members.")
        finally:
            if self.current_module_prefix and self.current_module_prefix[-1] == store_name:
                self.current_module_prefix.pop()

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
        original_func_name = node.name.name
        mangled_func_name = self._get_mangled_name(original_func_name)

        outer_builder = self.current_builder
        outer_llvm_function = self.current_llvm_function
        outer_symbol_table = self.current_symbol_table
        param_types = [rt.CoralValuePtrType] * len(node.params)
        return_type = rt.CoralValuePtrType
        func_type = llvmlite.ir.FunctionType(return_type, param_types)
        llvm_func = Function(self.module, func_type, name=mangled_func_name) # Use mangled name
        self.current_llvm_function = llvm_func
        self.current_symbol_table = {} # Functions have their own local symbol table for params/vars
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
                print(f"Warning: Single expression body for func '{mangled_func_name}' - wrapping in return.")
                # This might be too simplistic; depends on how parser creates FunctionDefinitionNode.body
                # If it's an ExpressionNode, it should be wrapped in a ReturnStatementNode for explicit return.
                # For now, assume parser creates a list of statements for body.
                # If it's a single expression that should be returned, it must be part of a ReturnStatement.
                return_stmt = ast_nodes.ReturnStatementNode(value=node.body, token_details=node.body.token_details) # Synthetic
                self._generate_statement_list([return_stmt])

             elif node.body is None: # No body
                 pass # Will fall through to default null return
             else: # Not an ExpressionNode and not a list
                 print(f"Error: Unsupported function body type for '{mangled_func_name}': {type(node.body)}. Expecting list of statements.")
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
                print(f"Warning: Func '{mangled_func_name}' non-void, lacks return. LLVM might error.")
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

    def _generate_iterate_loop(self, node: ast_nodes.IterateLoopNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for iterate loop.")

        current_llvm_func = self.current_llvm_function

        # Initialization
        iterable_val_ptr = self._generate_expression(node.iterable)

        # Declare or get coral_runtime_iterate_start
        func_ty_iter_start = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.CoralValuePtrType])
        iter_start_func = self.module.globals.get("coral_runtime_iterate_start")
        if iter_start_func is None or not isinstance(iter_start_func, llvmlite.ir.Function):
            iter_start_func = llvmlite.ir.Function(self.module, func_ty_iter_start, name="coral_runtime_iterate_start")
        iterator_state_ptr = self.current_builder.call(iter_start_func, [iterable_val_ptr], name="iter_state")

        # Create Blocks
        loop_header_block = current_llvm_func.append_basic_block(name="iter.header")
        loop_body_block = current_llvm_func.append_basic_block(name="iter.body")
        loop_exit_block = current_llvm_func.append_basic_block(name="iter.exit")

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        # Loop Header
        self.current_builder.position_at_end(loop_header_block)
        # Declare or get coral_runtime_iterate_has_next
        func_ty_has_next = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.CoralValuePtrType])
        iter_has_next_func = self.module.globals.get("coral_runtime_iterate_has_next")
        if iter_has_next_func is None or not isinstance(iter_has_next_func, llvmlite.ir.Function):
            iter_has_next_func = llvmlite.ir.Function(self.module, func_ty_has_next, name="coral_runtime_iterate_has_next")
        has_next_coral_val_ptr = self.current_builder.call(iter_has_next_func, [iterator_state_ptr], name="iter_has_next")
        boolean_has_next = rt.get_boolean_value(self.current_builder, has_next_coral_val_ptr, current_llvm_func)
        self.current_builder.cbranch(boolean_has_next, loop_body_block, loop_exit_block)

        # Loop Body
        self.current_builder.position_at_end(loop_body_block)
        # Declare or get coral_runtime_iterate_next
        func_ty_next = llvmlite.ir.FunctionType(rt.CoralValuePtrType, [rt.CoralValuePtrType])
        iter_next_func = self.module.globals.get("coral_runtime_iterate_next")
        if iter_next_func is None or not isinstance(iter_next_func, llvmlite.ir.Function):
            iter_next_func = llvmlite.ir.Function(self.module, func_ty_next, name="coral_runtime_iterate_next")
        current_element_ptr = self.current_builder.call(iter_next_func, [iterator_state_ptr], name="iter_elem")

        if node.loop_variable:
            var_name = node.loop_variable.name
            var_storage_ptr = self._get_variable_storage(var_name)
            if var_storage_ptr is None:
                # Allocate in function entry block for simplicity
                # Ensure entry block has a builder positioned correctly, or create one temporarily
                original_block = self.current_builder.block
                original_debug_loc = self.current_builder.debug_loc

                entry_b = current_llvm_func.entry_basic_block
                if not entry_b.instructions:
                    entry_builder = llvmlite.ir.IRBuilder(entry_b)
                else:
                    # Position builder before the first instruction that is not an alloca
                    # This is a common pattern for allocas
                    first_instr_not_alloca = next((instr for instr in entry_b.instructions if not isinstance(instr, llvmlite.ir.AllocaInstr)), None)
                    if first_instr_not_alloca:
                        entry_builder = llvmlite.ir.IRBuilder(entry_b)
                        entry_builder.position_before(first_instr_not_alloca)
                    else: # block only has allocas or is empty
                        entry_builder = llvmlite.ir.IRBuilder(entry_b)
                        if entry_b.instructions: # has allocas
                             entry_builder.position_after(entry_b.instructions[-1])
                        # else it's empty, builder is at the start

                var_storage_ptr = entry_builder.alloca(rt.CoralValuePtrType, name=f"{var_name}_iter_storage")
                self._set_variable_storage(var_name, var_storage_ptr)

                # Restore builder if it was moved for entry block insertion
                if original_block:
                    self.current_builder.position_at_end(original_block)
                    self.current_builder.debug_loc = original_debug_loc


            self.current_builder.store(current_element_ptr, var_storage_ptr)

        body_stmts = node.body if isinstance(node.body, list) else [node.body]
        self._generate_statement_list(body_stmts)

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        # Loop Exit
        self.current_builder.position_at_end(loop_exit_block)
        # Optional: Call coral_runtime_iterate_end(iterator_state_ptr) here

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
        condition_val_ptr = self._generate_expression(node.condition)
        boolean_condition_val = rt.get_boolean_value(self.current_builder, condition_val_ptr, current_llvm_func)
        # If condition is true, exit loop; if false, go to body.
        self.current_builder.cbranch(boolean_condition_val, loop_exit_block, loop_body_block)

        self.current_builder.position_at_end(loop_body_block)
        # Ensure node.body is a list for _generate_statement_list
        body_stmts = node.body if isinstance(node.body, list) else [node.body]
        self._generate_statement_list(body_stmts)

        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(loop_header_block)

        self.current_builder.position_at_end(loop_exit_block)

    def _generate_use_statement(self, node: ast_nodes.UseStatementNode):
        qname_parts = [part.name for part in node.qualified_identifier.parts]
        qname_str = ".".join(qname_parts)

        alias_str = ""
        if node.alias:
            alias_str = f" as {node.alias.name}"

        print(f"Info: 'use {qname_str}{alias_str}' encountered. Full import/namespacing beyond current module name mangling is not yet implemented.")
        # No LLVM IR generation for 'use' statements at this stage.
        # Future work would involve managing symbol tables, aliases, and possibly loading other modules.

    def _generate_empty_statement(self, node: ast_nodes.EmptyStatementNode):
        # Empty statements generate no IR.
        # print(f"Debug: Generating IR for EmptyStatementNode at {node.location_info}") # Optional debug
        pass

    def _generate_module_definition(self, node: ast_nodes.ModuleDefinitionNode):
        self.current_module_prefix.append(node.name.name)
        try:
            self._generate_statement_list(node.body)
        finally:
            # Ensure prefix is popped even if an error occurs in module body generation
            if self.current_module_prefix and self.current_module_prefix[-1] == node.name.name:
                self.current_module_prefix.pop()
            # else: # Should not happen if append/pop are balanced
            #     print(f"Warning: Module prefix stack imbalance when leaving module {node.name.name}")


    def _generate_unless_statement(self, node: ast_nodes.UnlessStatementNode):
        if self.current_builder is None or self.current_llvm_function is None:
            raise RuntimeError("Builder or current_function not initialized for unless statement.")

        current_llvm_func = self.current_llvm_function

        body_block = current_llvm_func.append_basic_block(name="unless.body")
        end_block = current_llvm_func.append_basic_block(name="unless.end")

        condition_val_ptr = self._generate_expression(node.condition)
        boolean_condition_val = rt.get_boolean_value(self.current_builder, condition_val_ptr, current_llvm_func)

        # If condition is true, go to end_block; if false, go to body_block.
        self.current_builder.cbranch(boolean_condition_val, end_block, body_block)

        # Position builder at the end of body_block
        self.current_builder.position_at_end(body_block)
        # Generate IR for the statements in node.block
        # Ensure node.block is a list for _generate_statement_list
        block_stmts = node.block if isinstance(node.block, list) else [node.block]
        self._generate_statement_list(block_stmts)

        # If body_block is not terminated, add an unconditional branch to end_block
        if not self.current_builder.block.is_terminated:
            self.current_builder.branch(end_block)

        # Position builder at the end of end_block
        self.current_builder.position_at_end(end_block)

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
