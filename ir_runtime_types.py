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
# Add more tags as needed (e.g., for arrays, objects, null)
TYPE_TAG_NULL = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 4)
TYPE_TAG_OBJECT = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 5)
TYPE_TAG_LIST = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 6)
TYPE_TAG_MAP = llvmlite.ir.Constant(llvmlite.ir.IntType(8), 7)


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


def get_integer_value(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, current_func: llvmlite.ir.Function) -> llvmlite.ir.Value:
    """Extracts an i32 value from a CoralValue*. Assumes type tag is already checked or includes basic check."""
    # Note: Robust error handling (e.g. branching to error block if tag mismatch) is simplified here.
    # A full implementation would involve creating error blocks and conditional branches.

    # Basic check (can be expanded with proper error handling blocks):
    # type_tag = get_coral_value_type_tag(builder, coral_value_ptr)
    # with builder.if_then(builder.icmp_unsigned('!=', type_tag, TYPE_TAG_INTEGER)):
    #     # Handle error: e.g., call a runtime error function, or jump to an error block
    #     # For now, this will just proceed, which is unsafe.
    #     pass

    data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
    # Cast the OpaquePtrType (i8*) to IntegerPtrType (i32*)
    int_ptr = builder.bitcast(data_op_ptr, IntegerPtrType, name="int_ptr_cast")
    return builder.load(int_ptr, name="int_val_loaded")

def get_float_value(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, current_func: llvmlite.ir.Function) -> llvmlite.ir.Value:
    """Extracts a double value from a CoralValue*."""
    data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
    # Cast OpaquePtrType (i8*) to FloatPtrType (double*)
    float_ptr = builder.bitcast(data_op_ptr, FloatPtrType, name="float_ptr_cast")
    return builder.load(float_ptr, name="float_val_loaded")

def get_boolean_value(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, current_func: llvmlite.ir.Function) -> llvmlite.ir.Value:
    """Extracts an i1 value from a CoralValue*."""
    data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
    # Cast OpaquePtrType (i8*) to BooleanPtrType (i1*)
    bool_ptr = builder.bitcast(data_op_ptr, BooleanPtrType, name="bool_ptr_cast")
    return builder.load(bool_ptr, name="bool_val_loaded")

# Recall: String data is stored in a ConcreteStringType {i32 len, i8* data_ptr}
# The OpaquePtr in CoralValue points to this ConcreteStringType instance.
ConcreteStringType = llvmlite.ir.LiteralStructType([llvmlite.ir.IntType(32), ptr_to(llvmlite.ir.IntType(8))])
ConcreteStringPtrType = ptr_to(ConcreteStringType)

def get_string_concrete_struct_ptr(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value) -> llvmlite.ir.Value:
    """Gets the pointer to the ConcreteStringType from a CoralValue* for a string."""
    data_op_ptr = get_coral_value_data_ptr(builder, coral_value_ptr)
    # Cast OpaquePtrType (i8*) to ConcreteStringPtrType ({i32, i8*}*)
    return builder.bitcast(data_op_ptr, ConcreteStringPtrType, name="concrete_str_ptr_cast")

def get_string_data_ptr(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, current_func: llvmlite.ir.Function) -> llvmlite.ir.Value:
    """Extracts the i8* to the character data from a CoralValue* (string type)."""
    concrete_str_ptr = get_string_concrete_struct_ptr(builder, coral_value_ptr)
    # GEP to the data field (index 1) of ConcreteStringType
    char_data_ptr_gep = builder.gep(concrete_str_ptr,
                                    [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0),
                                     llvmlite.ir.Constant(llvmlite.ir.IntType(32), 1)],
                                    name="str_char_data_ptr_gep")
    return builder.load(char_data_ptr_gep, name="str_char_data_loaded")

def get_string_length(builder: llvmlite.ir.IRBuilder, coral_value_ptr: llvmlite.ir.Value, current_func: llvmlite.ir.Function) -> llvmlite.ir.Value:
    """Extracts the length (i32) from a CoralValue* (string type)."""
    concrete_str_ptr = get_string_concrete_struct_ptr(builder, coral_value_ptr)
    # GEP to the length field (index 0) of ConcreteStringType
    len_ptr_gep = builder.gep(concrete_str_ptr,
                              [llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0),
                               llvmlite.ir.Constant(llvmlite.ir.IntType(32), 0)],
                              name="str_len_ptr_gep")
    return builder.load(len_ptr_gep, name="str_len_loaded")
