import llvmlite.ir

# Primitive LLVM types
IntegerType = llvmlite.ir.IntType(32)  # Assuming 32-bit integers
FloatType = llvmlite.ir.DoubleType()   # Using double precision for floats
BooleanType = llvmlite.ir.IntType(1)   # 1-bit integer for booleans
OpaquePtrType = llvmlite.ir.IntType(8).गंगा() # Pointer to opaque type

# String type: { i32, [0 x i8] } (length + flexible array member)
# llvmlite doesn't directly support flexible array members in a struct type definition
# in a way that's immediately usable for global constants.
# We'll define it conceptually and handle its creation/manipulation in the IR generation.
# For now, a pointer to such a structure will often be used.
StringType = llvmlite.ir.global_context.get_identified_type("String")
if not StringType.elements: # Check if it's a forward declaration
    StringType.set_body(llvmlite.ir.IntType(32), llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), 0))


# CoralValue struct: { i8 type_tag, %OpaquePtr }
CoralValueType = llvmlite.ir.global_context.get_identified_type("CoralValue")
if not CoralValueType.elements: # Check if it's a forward declaration
    CoralValueType.set_body(
        llvmlite.ir.IntType(8),  # type_tag
        OpaquePtrType            # pointer to actual data (or the data itself if small)
    )

# Type tags (constants)
TYPE_TAG_INTEGER = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 0)
TYPE_TAG_FLOAT = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 1)
TYPE_TAG_BOOLEAN = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 2)
TYPE_TAG_STRING = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 3)
TYPE_TAG_NULL = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 4)
TYPE_TAG_OBJECT = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 5)
TYPE_TAG_LIST = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 6)
TYPE_TAG_MAP = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 7)
TYPE_TAG_ERROR = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 8) # New error tag


# Placeholder complex types (opaque for now, primarily for pointer typing)
# These would eventually point to heap-allocated structures managed by a runtime.
# For now, defining them as an empty struct or a struct with a dummy field (e.g. i32)
# allows us to have a concrete type for pointers.
_ctx = llvmlite.ir.global_context

CoralObjectType = _ctx.get_identified_type("CoralObject")
if not CoralObjectType.elements:
    CoralObjectType.set_body(llvmlite.ir.IntType(8)) # Dummy body, makes it non-opaque for alloca

CoralListType = _ctx.get_identified_type("CoralList")
if not CoralListType.elements:
    CoralListType.set_body(llvmlite.ir.IntType(8)) # Dummy body

CoralMapType = _ctx.get_identified_type("CoralMap")
if not CoralMapType.elements:
    CoralMapType.set_body(llvmlite.ir.IntType(8)) # Dummy body

CoralObjectPtrType = ptr_to(CoralObjectType)
CoralListPtrType = ptr_to(CoralListType)
CoralMapPtrType = ptr_to(CoralMapType)


# Example of how you might represent a specific value
# This is more for conceptual understanding; actual value creation will be in IRGenerator
# IntValStruct = llvmlite.ir.LiteralStructType([IntegerType])
# FloatValStruct = llvmlite.ir.LiteralStructType([FloatType])
# BoolValStruct = llvmlite.ir.LiteralStructType([BooleanType])

# Note: For strings, the OpaquePtr in CoralValue would point to a StringType instance.
# For primitive types like int, float, bool, you might store them directly
# if you modify the CoralValue union/struct, or point to a heap-allocated version.
# The current OpaquePtrType is generic. We will refine how data is stored (direct or pointer)
# as we implement expression evaluation in IRGenerator.
# For now, OpaquePtrType implies the actual data is allocated elsewhere and this is a pointer to it.

# Helper to create a pointer to a type
def ptr_to(llvm_type):
    return llvm_type.गंगा()

# Example: Pointer to our CoralValueType
CoralValuePtrType = ptr_to(CoralValueType)
IntegerPtrType = ptr_to(IntegerType)
FloatPtrType = ptr_to(FloatType)
BooleanPtrType = ptr_to(BooleanType)
StringPtrType = ptr_to(StringType)

# We might need a generic ValueType that can hold any of our values by pointer
# This is useful if we want to pass around "any Coral value"
# This will be an opaque pointer for now, to be casted at runtime
AnyCoralValuePtrType = OpaquePtrType


# --- Helper functions to create CoralValue instances ---

def create_coral_value(builder: llvmlite.ir.IRBuilder, type_tag: llvmlite.ir.Constant, data_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """
    Allocates a CoralValue struct on the stack, stores the type_tag and data_ptr into it.
    Returns a pointer to the allocated CoralValue.
    """
    coral_value_ptr = builder.alloca(CoralValueType)

    # Store type_tag
    type_tag_ptr = builder.gep(coral_value_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0), llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0)], name="type_tag_ptr")
    builder.store(type_tag, type_tag_ptr)

    # Cast data_ptr to OpaquePtrType if needed
    # This assumes data_ptr is already of a type that can be cast to OpaquePtrType (e.g., i32*, double*)
    casted_data_ptr = builder.bitcast(data_ptr, OpaquePtrType, name="casted_data_ptr")

    # Store data_ptr
    data_val_ptr = builder.gep(coral_value_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0), llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1)], name="data_val_ptr")
    builder.store(casted_data_ptr, data_val_ptr)

    return coral_value_ptr # Return pointer to the CoralValue struct


def create_integer_value(builder: llvmlite.ir.IRBuilder, value: int) -> llvmlite.ir.Value:
    """Creates a CoralValue for an integer literal."""
    # Allocate memory for the i32 value itself
    int_val_ptr = builder.alloca(IntegerType, name="int_val_alloc")
    builder.store(llvmlite.ir.Constant(IntegerType, value), int_val_ptr)
    # Create the CoralValue struct, pointing to the allocated i32
    return create_coral_value(builder, TYPE_TAG_INTEGER, int_val_ptr)

def create_float_value(builder: llvmlite.ir.IRBuilder, value: float) -> llvmlite.ir.Value:
    """Creates a CoralValue for a float literal."""
    float_val_ptr = builder.alloca(FloatType, name="float_val_alloc")
    builder.store(llvmlite.ir.Constant(FloatType, value), float_val_ptr)
    return create_coral_value(builder, TYPE_TAG_FLOAT, float_val_ptr)

def create_boolean_value(builder: llvmlite.ir.IRBuilder, value: bool) -> llvmlite.ir.Value:
    """Creates a CoralValue for a boolean literal."""
    bool_val_ptr = builder.alloca(BooleanType, name="bool_val_alloc")
    builder.store(llvmlite.ir.Constant(BooleanType, 1 if value else 0), bool_val_ptr)
    return create_coral_value(builder, TYPE_TAG_BOOLEAN, bool_val_ptr)

def create_string_value(builder: llvmlite.ir.IRBuilder, module: llvmlite.ir.Module, value: str) -> llvmlite.ir.Value:
    """
    Creates a CoralValue for a string literal.
    The string data is stored as a global constant.
    The CoralValue will point to this global string.
    """
    # Create a global constant for the string data
    str_val = value.encode('utf-8')
    str_len = len(str_val)

    # LLVM string constant: null-terminated char array
    # For Coral's StringType {i32 len, [0 x i8] data}, we need to store len separately
    # from the actual character data.

    # 1. Create the character array global constant (null-terminated for C compatibility if ever needed)
    char_array_type = llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), str_len + 1) # +1 for null terminator
    g_str_data_name = f".str_data.{module.get_unique_name_suffix()}"
    g_str_data = llvmlite.ir.GlobalVariable(module, char_array_type, name=g_str_data_name)
    g_str_data.linkage = 'internal'
    g_str_data.global_constant = True
    g_str_data.initializer = llvmlite.ir.Constant(char_array_type, bytearray(str_val + b'\x00'))

    # 2. Allocate the Coral StringType struct {i32 len, [0 x i8]} equivalent
    # Since StringType has a flexible array member, we can't directly alloca it with data.
    # Instead, we will store a pointer to the global char array.
    # For simplicity, we'll treat our StringType as {i32, i8*} for heap allocation.
    # Or, we can make the CoralValue point directly to a structure containing len and char*

    # Let's define a specific struct for string payload: {i32 len, i8* data}
    # This is different from StringType which implies flexible array member.
    # This is a common way to handle strings when the flexible array member is tricky.
    ConcreteStringType = llvmlite.ir.LiteralStructType([llvmlite.ir.IntType(32), ptr_to(llvmlite.ir.IntType(8))])

    string_struct_ptr = builder.alloca(ConcreteStringType, name="string_struct_alloc")

    # Store length
    len_ptr = builder.gep(string_struct_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32),0), llvmlite.ir.Constant(llvmlite.ir.IntType(32),0)], name="len_ptr")
    builder.store(llvmlite.ir.Constant(llvmlite.ir.IntType(32), str_len), len_ptr)

    # Store pointer to global string data
    data_gep = builder.gep(g_str_data, [llvmlite.ir.Constant(llvmlite.ir.IntType(32),0), llvmlite.ir.Constant(llvmlite.ir.IntType(32),0)], name="g_str_data_ptr")

    str_data_field_ptr = builder.gep(string_struct_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32),0), llvmlite.ir.Constant(llvmlite.ir.IntType(32),1)], name="str_data_field_ptr")
    builder.store(data_gep, str_data_field_ptr)

    # Create CoralValue pointing to this String struct
    return create_coral_value(builder, TYPE_TAG_STRING, string_struct_ptr)


def create_null_value(builder: llvmlite.ir.IRBuilder) -> llvmlite.ir.Value:
    """Creates a CoralValue for a null/empty value."""
    # For null, the data pointer can be a null pointer of OpaquePtrType
    null_data_ptr = llvmlite.ir.Constant(OpaquePtrType, None) # Creates a null pointer

    coral_value_ptr = builder.alloca(CoralValueType)
    type_tag_ptr = builder.gep(coral_value_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0), llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0)])
    builder.store(TYPE_TAG_NULL, type_tag_ptr)

    data_val_ptr = builder.gep(coral_value_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0), llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1)])
    builder.store(null_data_ptr, data_val_ptr)

    return coral_value_ptr


def create_object_value(builder: llvmlite.ir.IRBuilder, obj_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """Creates a CoralValue for an object, where obj_ptr points to the actual object data."""
    # obj_ptr should be a pointer to some specific object struct, castable to OpaquePtrType
    return create_coral_value(builder, TYPE_TAG_OBJECT, obj_ptr)

def create_list_value(builder: llvmlite.ir.IRBuilder, list_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """Creates a CoralValue for a list, where list_ptr points to the actual list data (e.g., CoralListType*)."""
    return create_coral_value(builder, TYPE_TAG_LIST, list_ptr)

def create_map_value(builder: llvmlite.ir.IRBuilder, map_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """Creates a CoralValue for a map, where map_ptr points to the actual map data (e.g., CoralMapType*)."""
    return create_coral_value(builder, TYPE_TAG_MAP, map_ptr)


# --- Helper functions to extract actual values from CoralValue* ---

def get_coral_value_type_tag(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """Loads and returns the type tag from a CoralValue*."""
    if not coral_value_ptr.type.is_pointer or not coral_value_ptr.type.pointee == CoralValueType:
        raise TypeError(f"Expected a pointer to CoralValueType, got {coral_value_ptr.type}")
    type_tag_gep = builder.gep(coral_value_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0), llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0)], name="type_tag_ptr_gep")
    return builder.load(type_tag_gep, name="type_tag_loaded")

def get_coral_value_data_ptr(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """Loads and returns the opaque data pointer from a CoralValue*."""
    if not coral_value_ptr.type.is_pointer or not coral_value_ptr.type.pointee == CoralValueType:
        raise TypeError(f"Expected a pointer to CoralValueType, got {coral_value_ptr.type}")
    data_ptr_gep = builder.gep(coral_value_ptr, [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0), llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1)], name="data_ptr_gep")
    return builder.load(data_ptr_gep, name="data_ptr_loaded")

def _create_global_string_constant_and_gep_ptr(builder: llvmlite.ir.IRBuilder, module: llvmlite.ir.Module, py_string: str, name: str) -> llvmlite.ir.Value:
    """
    Creates a global string constant in the module if it doesn't exist,
    and returns an i8* pointer to it using the provided builder.
    """
    g_var = module.globals.get(name)
    if g_var is None:
        c_str_val = llvmlite.ir.Constant(
            llvmlite.ir.ArrayType(llvmlite.ir.IntType(8), len(py_string) + 1),
            bytearray(py_string.encode('utf-8') + b'\x00')
        )
        g_var = llvmlite.ir.GlobalVariable(module, c_str_val.type, name=name)
        g_var.linkage = 'internal'
        g_var.global_constant = True
        g_var.initializer = c_str_val

    idx_type = llvmlite.ir.IntType(32)
    # Use the provided builder to GEP to get the i8*
    return builder.gep(g_var, [llvmlite.ir.Constant(idx_type, 0), llvmlite.ir.Constant(idx_type, 0)], name=f"{name}_ptr")


def get_integer_value(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, module: llvmlite.ir.Module) -> llvmlite.ir.Value:
    """
    Extracts an i32 value from a CoralValue*.
    If type mismatch, calls coral_runtime_create_error_value and returns 0.
    """
    type_tag = get_coral_value_type_tag(builder, coral_value_ptr)
    is_correct_type = builder.icmp_unsigned('==', type_tag, TYPE_TAG_INTEGER)

    result_alloca = builder.alloca(IntegerType, name="get_int_res_alloc")

    with builder.if_else(is_correct_type) as (then_block, else_block):
        with then_block:
            data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
            int_ptr = builder.bitcast(data_op_ptr, IntegerPtrType, name="int_ptr_cast")
            loaded_int_val = builder.load(int_ptr, name="int_val_loaded")
            builder.store(loaded_int_val, result_alloca)
        with else_block:
            create_error_fn = get_runtime_function("coral_runtime_create_error_value")
            error_msg_str = "TypeError: Expected an integer value."
            msg_char_ptr = _create_global_string_constant_and_gep_ptr(builder, module, error_msg_str, ".str.err_type_int")
            # The created error CoralValue* is not directly used by this function's return path,
            # but calling this ensures the error is created in the runtime.
            builder.call(create_error_fn, [msg_char_ptr])
            builder.store(llvmlite.ir.Constant(IntegerType, 0), result_alloca) # Default/poison value

    return builder.load(result_alloca, name="final_int_value")

def get_float_value(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, module: llvmlite.ir.Module) -> llvmlite.ir.Value:
    """
    Extracts a double value from a CoralValue*.
    If type mismatch, calls coral_runtime_create_error_value and returns 0.0.
    """
    type_tag = get_coral_value_type_tag(builder, coral_value_ptr)
    is_correct_type = builder.icmp_unsigned('==', type_tag, TYPE_TAG_FLOAT)

    result_alloca = builder.alloca(FloatType, name="get_float_res_alloc")

    with builder.if_else(is_correct_type) as (then_block, else_block):
        with then_block:
            data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
            float_ptr = builder.bitcast(data_op_ptr, FloatPtrType, name="float_ptr_cast")
            loaded_val = builder.load(float_ptr, name="float_val_loaded")
            builder.store(loaded_val, result_alloca)
        with else_block:
            create_error_fn = get_runtime_function("coral_runtime_create_error_value")
            error_msg_str = "TypeError: Expected a float value."
            msg_char_ptr = _create_global_string_constant_and_gep_ptr(builder, module, error_msg_str, ".str.err_type_float")
            builder.call(create_error_fn, [msg_char_ptr])
            builder.store(llvmlite.ir.Constant(FloatType, 0.0), result_alloca)

    return builder.load(result_alloca, name="final_float_value")


def unsafe_get_boolean_value(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """
    Extracts an i1 value from a CoralValue*.
    Assumes the input coral_value_ptr is a valid boolean CoralValue.
    No type checking or error handling is performed here.
    """
    # Directly get the data pointer, cast, and load.
    data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
    bool_ptr = builder.bitcast(data_op_ptr, BooleanPtrType, name="unsafe_bool_ptr_cast")
    loaded_val = builder.load(bool_ptr, name="unsafe_bool_val_loaded")
    return loaded_val

# Recall: String data is stored in a ConcreteStringType {i32 len, i8* data_ptr}
# The OpaquePtr in CoralValue points to this ConcreteStringType instance.
ConcreteStringType = llvmlite.ir.LiteralStructType([llvmlite.ir.IntType(32), ptr_to(llvmlite.ir.IntType(8))])
ConcreteStringPtrType = ptr_to(ConcreteStringType)

def get_string_concrete_struct_ptr(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, module: llvmlite.ir.Module) -> llvmlite.ir.Value:
    """
    Gets the pointer to the ConcreteStringType from a CoralValue*.
    If type mismatch, calls coral_runtime_create_error_value and returns null pointer.
    """
    type_tag = get_coral_value_type_tag(builder, coral_value_ptr)
    is_correct_type = builder.icmp_unsigned('==', type_tag, TYPE_TAG_STRING)

    result_alloca = builder.alloca(ConcreteStringPtrType, name="get_str_struct_ptr_alloc")

    with builder.if_else(is_correct_type) as (then_block, else_block):
        with then_block:
            data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
            casted_ptr = builder.bitcast(data_op_ptr, ConcreteStringPtrType, name="concrete_str_ptr_cast")
            builder.store(casted_ptr, result_alloca)
        with else_block:
            create_error_fn = get_runtime_function("coral_runtime_create_error_value")
            error_msg_str = "TypeError: Expected a string value."
            msg_char_ptr = _create_global_string_constant_and_gep_ptr(builder, module, error_msg_str, ".str.err_type_string")
            builder.call(create_error_fn, [msg_char_ptr])
            builder.store(llvmlite.ir.Constant(ConcreteStringPtrType, None), result_alloca) # Return null pointer

    return builder.load(result_alloca, name="final_str_struct_ptr")


def get_string_data_ptr(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, module: llvmlite.ir.Module) -> llvmlite.ir.Value:
    """
    Extracts the i8* to the character data from a CoralValue* (string type).
    If not a string, error is handled by get_string_concrete_struct_ptr, and this returns null.
    """
    concrete_str_ptr = get_string_concrete_struct_ptr(builder, coral_value_ptr, module) # This now handles type check

    # If concrete_str_ptr is null due to type error, this GEP would be on null.
    # We need to check concrete_str_ptr before GEPping.
    is_valid_str_ptr = builder.icmp_unsigned('!=', concrete_str_ptr, llvmlite.ir.Constant(ConcreteStringPtrType, None))

    result_alloca = builder.alloca(ptr_to(llvmlite.ir.IntType(8)), name="get_str_data_ptr_alloc")

    with builder.if_else(is_valid_str_ptr) as (then_block, else_block):
        with then_block:
            char_data_ptr_gep = builder.gep(concrete_str_ptr,
                                            [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0),
                                             llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1)],
                                            name="str_char_data_ptr_gep")
            loaded_ptr = builder.load(char_data_ptr_gep, name="str_char_data_loaded")
            builder.store(loaded_ptr, result_alloca)
        with else_block:
            # Error already printed by get_string_concrete_struct_ptr if it was a type error
            builder.store(llvmlite.ir.Constant(ptr_to(llvmlite.ir.IntType(8)), None), result_alloca) # Return null

    return builder.load(result_alloca, name="final_str_data_ptr")


def get_string_length(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, module: llvmlite.ir.Module) -> llvmlite.ir.Value:
    """
    Extracts the length (i32) from a CoralValue* (string type).
    If not a string, error is handled by get_string_concrete_struct_ptr, and this returns 0.
    """
    concrete_str_ptr = get_string_concrete_struct_ptr(builder, coral_value_ptr, module) # Handles type check

    is_valid_str_ptr = builder.icmp_unsigned('!=', concrete_str_ptr, llvmlite.ir.Constant(ConcreteStringPtrType, None))

    result_alloca = builder.alloca(IntegerType, name="get_str_len_alloc")

    with builder.if_else(is_valid_str_ptr) as (then_block, else_block):
        with then_block:
            len_ptr_gep = builder.gep(concrete_str_ptr,
                                      [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0),
                                       llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0)],
                                      name="str_len_ptr_gep")
            loaded_len = builder.load(len_ptr_gep, name="str_len_loaded")


# --- Runtime Helper Function Declarations ---
# These functions are implemented externally (e.g., in C) and called by the generated IR.

# Stores references to declared runtime functions. Populated by declare_runtime_functions.
RUNTIME_FUNCTIONS = {}

def declare_runtime_functions(module: llvmlite.ir.Module):
    """
    Declares all necessary runtime helper functions and populates RUNTIME_FUNCTIONS.
    This should be called once per module.
    """
    char_ptr_type = ptr_to(llvmlite.ir.IntType(8))

    # Helper to declare a function if it doesn't already exist
    def declare_function(name, return_type, arg_types, var_arg=False):
        # Check if function already declared in the global RUNTIME_FUNCTIONS dict
        if name in RUNTIME_FUNCTIONS:
            # Optional: Verify existing signature if needed, though get_runtime_function will fail if it's wrong type
            return

        # Check if function already exists in the LLVM module
        existing_func = module.globals.get(name)
        if existing_func and isinstance(existing_func, llvmlite.ir.Function):
            # Function already exists in module, reuse it if signature matches
            if existing_func.function_type.return_type == return_type and \
               list(existing_func.function_type.args) == list(arg_types) and \
               existing_func.function_type.var_arg == var_arg:
                RUNTIME_FUNCTIONS[name] = existing_func
            else:
                # This case should ideally not happen if names are unique and used consistently
                raise RuntimeError(f"Function {name} already exists in module with a different signature.")
        else: # Not in module or not a function of this name, so declare anew
            fn_type = llvmlite.ir.FunctionType(return_type, arg_types, var_arg)
            RUNTIME_FUNCTIONS[name] = llvmlite.ir.Function(module, fn_type, name=name)


    # Binary operations: (CoralValue*, CoralValue*) -> CoralValue*
    binary_op_args = [CoralValuePtrType, CoralValuePtrType]
    # Map internal operator keys to actual runtime function names
    binary_op_runtime_names = {
        "add": "coral_runtime_binary_add",
        "sub": "coral_runtime_binary_sub",
        "mul": "coral_runtime_binary_mul",
        "div": "coral_runtime_binary_div",
        "mod": "coral_runtime_binary_mod",
        "eq": "coral_runtime_binary_eq",
        "ne": "coral_runtime_binary_ne",
        "lt": "coral_runtime_binary_lt",
        "le": "coral_runtime_binary_le",
        "gt": "coral_runtime_binary_gt",
        "ge": "coral_runtime_binary_ge",
        "and": "coral_runtime_binary_and",
        "or": "coral_runtime_binary_or",
    }
    for op_key, fn_name in binary_op_runtime_names.items():
        declare_function(fn_name, CoralValuePtrType, binary_op_args)
        RUNTIME_FUNCTIONS[op_key] = RUNTIME_FUNCTIONS[fn_name] # Map short key to full name for IR generator convenience

    # Unary operations: (CoralValue*) -> CoralValue*
    unary_op_args = [CoralValuePtrType]
    unary_op_runtime_names = {
        "neg": "coral_runtime_unary_neg",
        "not": "coral_runtime_unary_not",
    }
    for op_key, fn_name in unary_op_runtime_names.items():
        declare_function(fn_name, CoralValuePtrType, unary_op_args)
        RUNTIME_FUNCTIONS[op_key] = RUNTIME_FUNCTIONS[fn_name]

    # Error value creation: (char*) -> CoralValue*
    declare_function("coral_runtime_create_error_value", CoralValuePtrType, [char_ptr_type])


def get_runtime_function(name: str) -> llvmlite.ir.Function:
    """Retrieves a declared runtime function. Ensure declare_runtime_functions has been called."""
    fn = RUNTIME_FUNCTIONS.get(name)
    if fn is None:
        # Attempt to get it from the module directly if it was declared by other means
        # This is a fallback and ideally declare_runtime_functions is the source of truth for RUNTIME_FUNCTIONS
        # However, direct module check can be problematic if signature is not verified here.
        # For safety, rely on RUNTIME_FUNCTIONS being populated correctly.
        raise LookupError(f"Runtime function '{name}' not found in RUNTIME_FUNCTIONS. "
                          "Ensure 'declare_runtime_functions(module)' is called before generating IR "
                          "that relies on this function.")
    if not isinstance(fn, llvmlite.ir.Function):
         raise TypeError(f"Expected LLVM Function for '{name}', but found {type(fn)} in RUNTIME_FUNCTIONS.")
    return fn
            builder.store(loaded_len, result_alloca)
        with else_block:
            builder.store(llvmlite.ir.Constant(IntegerType, 0), result_alloca) # Return 0

    return builder.load(result_alloca, name="final_str_len")


# --- Runtime Helper Function Declarations ---
# These functions are implemented externally (e.g., in C) and called by the generated IR.

# Stores references to declared runtime functions. Populated by declare_runtime_functions.
RUNTIME_FUNCTIONS = {}

def declare_runtime_functions(module: llvmlite.ir.Module):
    """
    Declares all necessary runtime helper functions and populates RUNTIME_FUNCTIONS.
    This should be called once per module.
    """
    char_ptr_type = ptr_to(llvmlite.ir.IntType(8))

    # Helper to declare a function if it doesn't already exist
    def declare_function(name, return_type, arg_types, var_arg=False):
        if name not in RUNTIME_FUNCTIONS:
            fn_type = llvmlite.ir.FunctionType(return_type, arg_types, var_arg)
            RUNTIME_FUNCTIONS[name] = llvmlite.ir.Function(module, fn_type, name=name)

    # Binary operations: (CoralValue*, CoralValue*) -> CoralValue*
    binary_op_args = [CoralValuePtrType, CoralValuePtrType]
    binary_ops = {
        # Arithmetic
        "add": "coral_runtime_binary_add",
        "sub": "coral_runtime_binary_sub",
        "mul": "coral_runtime_binary_mul",
        "div": "coral_runtime_binary_div",
        "mod": "coral_runtime_binary_mod",
        # Comparison (name in Coral -> C function name suffix)
        # Note: AST uses 'equals' for ==, '!=' for !=, etc.
        "eq": "coral_runtime_binary_eq",    # For '==' or 'equals'
        "ne": "coral_runtime_binary_ne",    # For '!='
        "lt": "coral_runtime_binary_lt",    # For '<'
        "le": "coral_runtime_binary_le",    # For '<='
        "gt": "coral_runtime_binary_gt",    # For '>'
        "ge": "coral_runtime_binary_ge",    # For '>='
        # Logical
        "and": "coral_runtime_binary_and",
        "or": "coral_runtime_binary_or",
    }
    for op_key, fn_name in binary_ops.items():
        declare_function(fn_name, CoralValuePtrType, binary_op_args)
        RUNTIME_FUNCTIONS[op_key] = RUNTIME_FUNCTIONS[fn_name] # Map short key to full name for IR generator convenience

    # Unary operations: (CoralValue*) -> CoralValue*
    unary_op_args = [CoralValuePtrType]
    unary_ops = {
        # Arithmetic
        "neg": "coral_runtime_unary_neg",  # For '-' (unary minus)
        # Logical
        "not": "coral_runtime_unary_not",
    }
    for op_key, fn_name in unary_ops.items():
        declare_function(fn_name, CoralValuePtrType, unary_op_args)
        RUNTIME_FUNCTIONS[op_key] = RUNTIME_FUNCTIONS[fn_name]

    # Error value creation: (char*) -> CoralValue*
    declare_function("coral_runtime_create_error_value", CoralValuePtrType, [char_ptr_type])

    # Error printing: (char*) -> void
    declare_function("coral_runtime_print_error", llvmlite.ir.VoidType(), [char_ptr_type])

    # Check if a CoralValue is null: (CoralValue*) -> CoralValue* (boolean)
    declare_function("coral_runtime_is_null", CoralValuePtrType, [CoralValuePtrType])

    # List operations
    # (CoralValue** elements_array, i32 num_elements) -> CoralValue* (new list)
    declare_function("coral_runtime_create_list", CoralValuePtrType, [ptr_to(CoralValuePtrType), IntegerType])
    # (CoralValue* list, CoralValue* index) -> CoralValue* (element)
    declare_function("coral_runtime_list_get_element", CoralValuePtrType, [CoralValuePtrType, CoralValuePtrType])
    # (CoralValue* list, CoralValue* index, CoralValue* value_to_set) -> CoralValue* (usually the list itself or null/error)
    declare_function("coral_runtime_list_set_element", CoralValuePtrType, [CoralValuePtrType, CoralValuePtrType, CoralValuePtrType])

    # Map operations
    # (CoralValue** keys_array, CoralValue** values_array, i32 num_entries) -> CoralValue* (new map)
    declare_function("coral_runtime_create_map", CoralValuePtrType, [ptr_to(CoralValuePtrType), ptr_to(CoralValuePtrType), IntegerType])
    # (CoralValue* map, CoralValue* key_string) -> CoralValue* (value or null if not found)
    # Note: get_property and get_element are often used for maps as well, assuming string keys.
    # If specialized map functions are needed, they'd be here. For now, using object property access for string keys.

    # Object operations
    # (CoralValue* object, CoralValue* property_name_string) -> CoralValue* (property value or null)
    declare_function("coral_runtime_object_get_property", CoralValuePtrType, [CoralValuePtrType, CoralValuePtrType])
    # (CoralValue* object, CoralValue* property_name_string, CoralValue* value_to_set) -> CoralValue* (object or null/error)
    declare_function("coral_runtime_object_set_property", CoralValuePtrType, [CoralValuePtrType, CoralValuePtrType, CoralValuePtrType])

    # Store definition related operations
    # (CoralValue* store_name_str, CoralValue* relation_name_str) -> void
    declare_function("coral_runtime_define_relation", llvmlite.ir.VoidType(), [CoralValuePtrType, CoralValuePtrType])
    # (CoralValue* store_name_str, CoralValue* target_type_str, CoralValue* cast_func_obj) -> void
    declare_function("coral_runtime_define_cast", llvmlite.ir.VoidType(), [CoralValuePtrType, CoralValuePtrType, CoralValuePtrType])
    # (CoralValue* store_name_str, CoralValue* message_name_str, CoralValue* handler_func_obj) -> void
    declare_function("coral_runtime_define_receive_handler", llvmlite.ir.VoidType(), [CoralValuePtrType, CoralValuePtrType, CoralValuePtrType])

    # Iterator operations
    # (CoralValue* iterable) -> CoralValue* (iterator_state_object)
    declare_function("coral_runtime_iterate_start", CoralValuePtrType, [CoralValuePtrType])
    # (CoralValue* iterator_state_object) -> CoralValue* (boolean result)
    declare_function("coral_runtime_iterate_has_next", CoralValuePtrType, [CoralValuePtrType])
    # (CoralValue* iterator_state_object) -> CoralValue* (next element)
    declare_function("coral_runtime_iterate_next", CoralValuePtrType, [CoralValuePtrType])

    # Type requirement functions
    # (CoralValue* input_value) -> CoralValue* (returns input_value if boolean, else new error CoralValue)
    declare_function("coral_runtime_require_boolean", CoralValuePtrType, [CoralValuePtrType])


# Convenience accessors for commonly used runtime functions
def get_runtime_function(name: str) -> llvmlite.ir.Function:
    """Retrieves a declared runtime function. Ensure declare_runtime_functions has been called."""
    fn = RUNTIME_FUNCTIONS.get(name)
    if fn is None:
        raise LookupError(f"Runtime function '{name}' not found or not declared. "
                          "Ensure 'declare_runtime_functions(module)' is called before generating IR "
                          "that relies on this function.")
    return fn
