# Base Node (Optional)
class ASTNode:
    def __init__(self, location_info):
        self.location_info = location_info

    def __repr__(self):
        return f"{self.__class__.__name__}(location={self.location_info})"

# Program/Module Level
class ProgramNode(ASTNode):
    def __init__(self, body, location_info):
        super().__init__(location_info)
        self.body = body # list of top-level statements

    def __repr__(self):
        return f"ProgramNode(body_len={len(self.body)}, location={self.location_info})"

class ModuleDefinitionNode(ASTNode):
    def __init__(self, name, body, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.body = body # list of top-level statements

    def __repr__(self):
        return f"ModuleDefinitionNode(name={self.name!r}, body_len={len(self.body)}, location={self.location_info})"

# Definitions
class FunctionDefinitionNode(ASTNode):
    def __init__(self, name, params, body, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.params = params # list of ParameterNode
        self.body = body # ExpressionNode or list of StatementNodes

    def __repr__(self):
        return f"FunctionDefinitionNode(name={self.name!r}, params_len={len(self.params)}, location={self.location_info})"

class ObjectDefinitionNode(ASTNode):
    def __init__(self, name, members, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.members = members # list of FieldDefinitionNode or MethodDefinitionNode

    def __repr__(self):
        return f"ObjectDefinitionNode(name={self.name!r}, members_len={len(self.members)}, location={self.location_info})"

class StoreDefinitionNode(ASTNode):
    def __init__(self, name, is_actor, for_target, members, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.is_actor = is_actor # bool
        self.for_target = for_target # IdentifierNode or None
        self.members = members # list of various store member nodes

    def __repr__(self):
        return (f"StoreDefinitionNode(name={self.name!r}, actor={self.is_actor}, "
                f"for={self.for_target!r}, members_len={len(self.members)}, location={self.location_info})")

# Statements
class AssignmentNode(ASTNode):
    def __init__(self, target, value, location_info):
        super().__init__(location_info)
        self.target = target # AssignableTargetNode (IdentifierNode, PropertyAccessNode, etc.)
        self.value = value # ExpressionNode or MapBlockAssignmentRHSNode

    def __repr__(self):
        return f"AssignmentNode(target={self.target!r}, value={self.value!r}, location={self.location_info})"

class ExpressionStatementNode(ASTNode):
    def __init__(self, expression, location_info):
        super().__init__(location_info)
        self.expression = expression # FullExpressionNode

    def __repr__(self):
        return f"ExpressionStatementNode(expression={self.expression!r}, location={self.location_info})"

class ReturnStatementNode(ASTNode):
    def __init__(self, value, location_info):
        super().__init__(location_info)
        self.value = value # ExpressionNode or None

    def __repr__(self):
        return f"ReturnStatementNode(value={self.value!r}, location={self.location_info})"

class UseStatementNode(ASTNode):
    def __init__(self, qualified_identifier, location_info):
        super().__init__(location_info)
        self.qualified_identifier = qualified_identifier # QualifiedIdentifierNode

    def __repr__(self):
        return f"UseStatementNode(qid={self.qualified_identifier!r}, location={self.location_info})"

class EmptyStatementNode(ASTNode):
    def __init__(self, location_info):
        super().__init__(location_info)

    def __repr__(self):
        return f"EmptyStatementNode(location={self.location_info})"

# Conditional Statements
class IfThenElseStatementNode(ASTNode):
    def __init__(self, condition, if_block, else_if_clauses, else_block, location_info):
        super().__init__(location_info)
        self.condition = condition # ExpressionNode
        self.if_block = if_block # list of StatementNodes or a single StatementNode
        self.else_if_clauses = else_if_clauses # list of tuples (condition_expr, block_nodes)
        self.else_block = else_block # list of StatementNodes or a single StatementNode, or None

    def __repr__(self):
        return (f"IfThenElseStatementNode(condition={self.condition!r}, "
                f"else_if_count={len(self.else_if_clauses)}, has_else={self.else_block is not None}, "
                f"location={self.location_info})")

class UnlessStatementNode(ASTNode):
    def __init__(self, condition, block, location_info):
        super().__init__(location_info)
        self.condition = condition # ExpressionNode
        self.block = block # list of StatementNodes or a single StatementNode

    def __repr__(self):
        return f"UnlessStatementNode(condition={self.condition!r}, location={self.location_info})"

class PostfixUnlessStatementNode(ASTNode):
    def __init__(self, expression_statement, condition, location_info):
        super().__init__(location_info)
        self.expression_statement = expression_statement # ExpressionStatementNode
        self.condition = condition # ExpressionNode

    def __repr__(self):
        return (f"PostfixUnlessStatementNode(statement={self.expression_statement!r}, "
                f"condition={self.condition!r}, location={self.location_info})")

# Loop Statements
class WhileLoopNode(ASTNode):
    def __init__(self, condition, body, location_info):
        super().__init__(location_info)
        self.condition = condition # ExpressionNode
        self.body = body # list of StatementNodes or a single StatementNode

    def __repr__(self):
        return f"WhileLoopNode(condition={self.condition!r}, location={self.location_info})"

class UntilLoopNode(ASTNode):
    def __init__(self, condition, body, location_info):
        super().__init__(location_info)
        self.condition = condition # ExpressionNode
        self.body = body # list of StatementNodes or a single StatementNode

    def __repr__(self):
        return f"UntilLoopNode(condition={self.condition!r}, location={self.location_info})"

class IterateLoopNode(ASTNode):
    def __init__(self, iterable, loop_variable, body, location_info):
        super().__init__(location_info)
        self.iterable = iterable # ExpressionNode
        self.loop_variable = loop_variable # IdentifierNode or None
        self.body = body # list of StatementNodes or a single StatementNode

    def __repr__(self):
        return (f"IterateLoopNode(iterable={self.iterable!r}, var={self.loop_variable!r}, "
                f"location={self.location_info})")

# Expressions (Core)
class IdentifierNode(ASTNode):
    def __init__(self, name, location_info):
        super().__init__(location_info)
        self.name = name # str

    def __repr__(self):
        return f"IdentifierNode(name={self.name!r}, location={self.location_info})"

class LiteralNode(ASTNode):
    def __init__(self, value, literal_type, location_info):
        super().__init__(location_info)
        self.value = value # actual Python value: str, int, float, bool
        self.literal_type = literal_type # str: 'STRING', 'INTEGER', 'FLOAT', 'BOOLEAN', 'EMPTY', 'NOW'

    def __repr__(self):
        return f"LiteralNode(value={self.value!r}, type={self.literal_type!r}, location={self.location_info})"

class ListLiteralNode(ASTNode):
    def __init__(self, elements, location_info):
        super().__init__(location_info)
        self.elements = elements # list of ExpressionNode

    def __repr__(self):
        return f"ListLiteralNode(elements_len={len(self.elements)}, location={self.location_info})"

class MapLiteralNode(ASTNode):
    def __init__(self, entries, location_info):
        super().__init__(location_info)
        self.entries = entries # list of MapEntryNode

    def __repr__(self):
        return f"MapLiteralNode(entries_len={len(self.entries)}, location={self.location_info})"

class MapEntryNode(ASTNode): # Used by MapLiteralNode
    def __init__(self, key, value, location_info):
        super().__init__(location_info)
        self.key = key # IdentifierNode (or potentially other simple literals if grammar allows)
        self.value = value # ExpressionNode

    def __repr__(self):
        return f"MapEntryNode(key={self.key!r}, value={self.value!r}, location={self.location_info})"

class TernaryConditionalExpressionNode(ASTNode):
    def __init__(self, condition, true_expr, false_expr, location_info):
        super().__init__(location_info)
        self.condition = condition # ExpressionNode
        self.true_expr = true_expr # ExpressionNode
        self.false_expr = false_expr # ExpressionNode

    def __repr__(self):
        return (f"TernaryConditionalExpressionNode(condition={self.condition!r}, "
                f"true_expr={self.true_expr!r}, false_expr={self.false_expr!r}, location={self.location_info})")

class BinaryOpNode(ASTNode):
    def __init__(self, left, operator, right, location_info):
        super().__init__(location_info)
        self.left = left # ExpressionNode
        self.operator = operator # str, e.g., '+', 'or', 'equals'
        self.right = right # ExpressionNode

    def __repr__(self):
        return (f"BinaryOpNode(left={self.left!r}, op={self.operator!r}, right={self.right!r}, "
                f"location={self.location_info})")

class UnaryOpNode(ASTNode):
    def __init__(self, operator, operand, location_info):
        super().__init__(location_info)
        self.operator = operator # str, e.g., '-', 'not'
        self.operand = operand # ExpressionNode

    def __repr__(self):
        return f"UnaryOpNode(op={self.operator!r}, operand={self.operand!r}, location={self.location_info})"

class PropertyAccessNode(ASTNode):
    def __init__(self, base_expr, property_name, location_info):
        super().__init__(location_info)
        self.base_expr = base_expr # ExpressionNode
        self.property_name = property_name # IdentifierNode

    def __repr__(self):
        return (f"PropertyAccessNode(base={self.base_expr!r}, property={self.property_name!r}, "
                f"location={self.location_info})")

class ListElementAccessNode(ASTNode):
    def __init__(self, base_expr, index_expr, location_info):
        super().__init__(location_info)
        self.base_expr = base_expr # ExpressionNode
        self.index_expr = index_expr # ExpressionNode

    def __repr__(self):
        return (f"ListElementAccessNode(base={self.base_expr!r}, index={self.index_expr!r}, "
                f"location={self.location_info})")

class DollarParamNode(ASTNode):
    def __init__(self, name_or_index, location_info):
        super().__init__(location_info)
        self.name_or_index = name_or_index # str or int

    def __repr__(self):
        return f"DollarParamNode(val={self.name_or_index!r}, location={self.location_info})"

# Call Operation & Arguments
class CallOperationNode(ASTNode):
    def __init__(self, callee, arguments, call_style, location_info):
        super().__init__(location_info)
        self.callee = callee # ExpressionNode (IdentifierNode, PropertyAccessNode, etc.)
        self.arguments = arguments # list of ArgumentNode or ExpressionNode
        self.call_style = call_style # str: 'paren', 'no_paren_space', 'no_paren_comma', 'empty_indicator', 'dot_method', 'dot_across'

    def __repr__(self):
        return (f"CallOperationNode(callee={self.callee!r}, args_len={len(self.arguments)}, "
                f"style='{self.call_style}', location={self.location_info})")

class ArgumentNode(ASTNode): # For named arguments or when explicit structure is needed
    def __init__(self, value, name, location_info):
        super().__init__(location_info)
        self.value = value # ExpressionNode
        self.name = name # IdentifierNode or None for positional

    def __repr__(self):
        return f"ArgumentNode(name={self.name!r}, value={self.value!r}, location={self.location_info})"

# AssignableTargetNode - Decided against this for now, parser will create specific nodes.

# Expression Wrappers / Suffixes
class FullExpressionNode(ASTNode): # Represents an expression that can have an error handler
    def __init__(self, expression, error_handler, location_info):
        super().__init__(location_info)
        self.expression = expression # ExpressionNode (e.g. BinaryOpNode, CallOperationNode, LiteralNode etc.)
        self.error_handler = error_handler # ErrorHandlerSuffixNode or None

    def __repr__(self):
        return (f"FullExpressionNode(expression={self.expression!r}, "
                f"has_error_handler={self.error_handler is not None}, location={self.location_info})")

class ErrorHandlerSuffixNode(ASTNode):
    def __init__(self, action, location_info):
        super().__init__(location_info)
        # action can be ReturnStatementNode, ExpressionNode, or a list of StatementNodes for a block
        self.action = action

    def __repr__(self):
        return f"ErrorHandlerSuffixNode(action_type={type(self.action).__name__}, location={self.location_info})"

# Function/Object/Store Internals
class ParameterNode(ASTNode):
    def __init__(self, name, default_value, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.default_value = default_value # ExpressionNode or None

    def __repr__(self):
        return (f"ParameterNode(name={self.name!r}, has_default={self.default_value is not None}, "
                f"location={self.location_info})")

class FieldDefinitionNode(ASTNode):
    def __init__(self, name, default_value, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.default_value = default_value # ExpressionNode or None (optional default)

    def __repr__(self):
        return (f"FieldDefinitionNode(name={self.name!r}, has_default={self.default_value is not None}, "
                f"location={self.location_info})")

class MethodDefinitionNode(ASTNode): # Similar to FunctionDefinitionNode
    def __init__(self, name, params, body, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode
        self.params = params # list of ParameterNode
        self.body = body # ExpressionNode or list of StatementNodes

    def __repr__(self):
        return (f"MethodDefinitionNode(name={self.name!r}, params_len={len(self.params)}, "
                f"location={self.location_info})")

class RelationDefinitionNode(ASTNode):
    def __init__(self, name, location_info):
        super().__init__(location_info)
        self.name = name # IdentifierNode

    def __repr__(self):
        return f"RelationDefinitionNode(name={self.name!r}, location={self.location_info})"

class CastDefinitionNode(ASTNode):
    def __init__(self, cast_to_type, body, location_info):
        super().__init__(location_info)
        self.cast_to_type = cast_to_type # str: 'string', 'map', 'list'
        self.body = body # ExpressionNode or list of AssignmentNode/ExpressionNode

    def __repr__(self):
        return f"CastDefinitionNode(type='{self.cast_to_type}', location={self.location_info})"

class ReceiveHandlerNode(ASTNode):
    def __init__(self, message_name, body, location_info):
        super().__init__(location_info)
        self.message_name = message_name # IdentifierNode
        self.body = body # similar to function body (ExpressionNode or list of StatementNodes)

    def __repr__(self.location_info):
        return (f"ReceiveHandlerNode(message_name={self.message_name!r}, "
                f"location={self.location_info})")

# Map Block Specific
class MapBlockEntryNode(ASTNode): # Used by MapBlockAssignmentRHSNode
    def __init__(self, key, value, location_info):
        super().__init__(location_info)
        self.key = key # IdentifierNode
        self.value = value # ExpressionNode

    def __repr__(self):
        return f"MapBlockEntryNode(key={self.key!r}, value={self.value!r}, location={self.location_info})"

class MapBlockAssignmentRHSNode(ASTNode):
    def __init__(self, entries, location_info):
        super().__init__(location_info)
        self.entries = entries # list of MapBlockEntryNode

    def __repr__(self):
        return f"MapBlockAssignmentRHSNode(entries_len={len(self.entries)}, location={self.location_info})"

# Miscellaneous
class QualifiedIdentifierNode(ASTNode):
    def __init__(self, parts, location_info):
        super().__init__(location_info)
        self.parts = parts # list of IdentifierNode

    def __repr__(self):
        part_names = ".".join([p.name for p in self.parts])
        return f"QualifiedIdentifierNode(parts='{part_names}', location={self.location_info})"

# Placeholder for GroupedExpression if needed, parser might just handle precedence
# class GroupedExpressionNode(ASTNode):
#     def __init__(self, expression, location_info):
#         super().__init__(location_info)
#         self.expression = expression # ExpressionNode
#
#     def __repr__(self):
#         return f"GroupedExpressionNode(expression={self.expression!r}, location={self.location_info})"

# Suffix nodes for complex primary expressions (if chosen over direct construction)
# class PropertyAccessSuffixNode(ASTNode):
#     def __init__(self, property_name, location_info):
#         super().__init__(location_info)
#         self.property_name = property_name # IdentifierNode
#
#     def __repr__(self):
#         return f"PropertyAccessSuffixNode(property={self.property_name!r}, location={self.location_info})"

# class ListElementAccessSuffixNode(ASTNode):
#     def __init__(self, index_expr, location_info):
#         super().__init__(location_info)
#         self.index_expr = index_expr # ExpressionNode
#
#     def __repr__(self):
#         return f"ListElementAccessSuffixNode(index={self.index_expr!r}, location={self.location_info})"
