import copy
class Expr:
    """
    Empty base class defining an expression in propositional logic.
    Propositions and operators that build expressions will extend from this class.
    """
    def negate(self) -> "Expr":
        raise NotImplementedError("Method not implemented")

    def eval(self) -> bool:
        raise NotImplementedError("Method not implemented")

    def simplify(self) -> "Expr":
        """
        The simplify method operates in two ways:
          1. it converts "->" and "<->" to expressions containing only AND, OR, NOT operators
          2. it eliminates the NOT operator from expressions, until it reaches propositions
        """
        return self

class Proposition(Expr):
    """
    Defines a term in propositional logic
    """

    def __init__(self, name: str, value: bool):
        self.name = name
        self.value = value

    def negate(self):
        return Proposition(self.name, not self.value)

    def eval(self):
        return self.value

    def is_negation_of(self, other: "Proposition"):
        return self == other.negate()

    def __hash__(self) -> int:
        name_hash = hash(self.name)
        if self.value:
            return name_hash + 1
        else:
            return name_hash

    def __eq__(self, other):
        return self.name == other.name and self.value == other.value

    def __str__(self):
        if self.value:
            return self.name
        else:
            return "¬" + self.name

    def __repr__(self):
        return self.__str__()


class UnaryOperator(Expr):
    """
    Base class for a unary operator in propositional logic
    """
    def __init__(self, op_name: str, expression: Expr):
        self.op_name = op_name
        self.expression = expression

    def __str__(self):
        return "%s(%s)" % (self.op_name.upper(), str(self.expression))

    def __repr__(self):
        return self.__str__()


class Not(UnaryOperator):
    """
    Defines a negation operation in propositional logic
    """
    def __init__(self, expression):
        super(Not, self).__init__("NOT", expression)

    def negate(self):
        return self.expression

    def eval(self):
        return not self.expression.eval()

    def simplify(self):
        """
        Simplification for a NOT operator is pushing the negation inside the expression
        :return: The propositional logic expression resulting from pushing the negation operator inside the negated
        expression (i.e. removal of the negation operator)
        """
        return self.expression.negate()


class BinaryOperator(Expr):
    """
    Base class for a binary operator in propositional logic
    """
    def __init__(self, op_name: str, left_expr: Expr, right_expr: Expr):
        self.op_name = op_name
        self.left_expr = left_expr
        self.right_expr = right_expr

    def __str__(self):
        return "%s(%s, %s)" % (self.op_name.upper(), str(self.left_expr), str(self.right_expr))

    def __repr__(self):
        return self.__str__()


class And(BinaryOperator):
    """
    Defines an "AND" operator in propositional logic
    """
    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(And, self).__init__("AND", left_expr, right_expr)

    def eval(self):
        left_expr_eval = self.left_expr.eval()
        if not left_expr_eval:
            return False
        else:
            return self.right_expr.eval()

    def negate(self):
        return Or(self.left_expr.negate(), self.right_expr.negate())

    def simplify(self):
        return And(self.left_expr.simplify(), self.right_expr.simplify())


class Or(BinaryOperator):
    """
    Defines an "OR" operator in propositional logic
    """
    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(Or, self).__init__("OR", left_expr, right_expr)

    def eval(self):
        left_expr_eval = self.left_expr.eval()
        if left_expr_eval:
            return True
        else:
            return self.right_expr.eval()

    def negate(self):
        return And(self.left_expr.negate(), self.right_expr.negate())

    def simplify(self):
        return Or(self.left_expr.simplify(), self.right_expr.simplify())


class Implies(BinaryOperator):
    """
    Defines an "Implication" operator in propositional logic
    """
    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(Implies, self).__init__("IMPL", left_expr, right_expr)

    def eval(self):
        return self.simplify().eval()

    def negate(self):
        return And(self.left_expr, self.right_expr.negate())

    def simplify(self):
        return Or(self.left_expr.negate(), self.right_expr)


class Equivalent(BinaryOperator):
    """
    Defines an "Equivalence" operator in propositional logic
    """

    def __init__(self, left_expr: Expr, right_expr: Expr):
        super(Equivalent, self).__init__("EQ", left_expr, right_expr)

    def eval(self):
        return self.simplify().eval()

    def negate(self):
        return Or(And(self.left_expr, self.right_expr.negate()), And(self.left_expr.negate(), self.right_expr))

    def simplify(self):
        return Or(And(self.left_expr, self.right_expr), And(self.left_expr.negate(), self.right_expr.negate()))

class SemanticTablauxTree:
    def __init__(self, expr_list):
        self.expr = expr_list
        self.left = None
        self.right = None

    def eval(self):
        for i in range(0, len(self.expr) - 1):
            for j in range(i + 1, len(self.expr)):
                if (type(self.expr[i].simplify()) == Proposition and type(self.expr[j].simplify()) == Proposition and self.expr[i].simplify() == self.expr[j].negate().simplify()):
                    raise Exception('contradiction found in expr')
    def expand(self, expand_index = -1):
        self.eval()
        expand_node = self.expr[expand_index]
        if (expand_index == -1):
            next_expand_index = 0
        else:
            next_expand_index = expand_index + 1
        simplified_node = expand_node.simplify()

        if (isinstance(simplified_node, Or)):
            left_node = simplified_node.left_expr
            right_node = simplified_node.right_expr

            self.left = copy.deepcopy(self)
            self.right = copy.deepcopy(self)

            self.left.expr[expand_index] = left_node
            self.right.expr[expand_index] = right_node


        elif (isinstance(simplified_node, And)):
            node_1,node_2 = simplified_node.left_expr, simplified_node.right_expr
            self.left = copy.deepcopy(self)
            self.left.expr[expand_index] = node_1
            self.left.expr.insert(expand_index, node_2)

        else:
            # base case, we reached a base proposition
            return

        print(f'simplified_node: {simplified_node}')
        print(self.expr)
        if (self.left is not None):
            print(self.left.expr[expand_index])
        if (self.right is not None):
            print(self.right.expr[expand_index])

        if ((self.left is not None) and next_expand_index < len(self.left.expr)):
            self.left.expand(next_expand_index)
        if ((self.right is not None) and next_expand_index < len(self.right.expr)):
            self.right.expand(next_expand_index)

if __name__ == "__main__":
    a1 = Proposition("A1", True)
    neg_a1 = Proposition("A1", False)

    a2 = Proposition("A2", True)
    a3 = Proposition("A3", True)

    assert a1.is_negation_of(neg_a1)
    assert neg_a1.is_negation_of(a1)

    assert a2 != a3
    assert a1 == neg_a1.negate()

    expr1 = Implies(a1, a2)
    expr2 = Or(a3, Not(Or(a3, neg_a1)))

    assert expr1.eval(), "a1->a2 must be true, since a1 and a2 are true"
    assert expr2.eval(), "a3 v ~(a3 v ~a1) must be true, since a3 is true"

    expr = Not(And(expr1, expr2))
    #    print("Truth value of %s is %r" % (str(expr), expr.eval()))

    #   print("Simplified version of expression \n\t%s is \n\t%s" % (str(expr), str(expr.simplify())))

    #   print("Negation of expression \n\t%s is \n\t%s" % (str(expr), str(expr.negate())))

    #   tablaux = SemanticTablauxTree([a1, Not(a1)])

    ## example 1:

    weather = Proposition("weather", True)
    park = Proposition("park", True)
    walk = Proposition("walk", True)

    p1 = Implies(weather, park)
    p2 = Implies(park, walk)
    p3 = Implies(walk, park)
    p4 = And(weather, Not(park))

    tablaux = SemanticTablauxTree([p1, p2, p3, p4])
    try:
        tablaux.expand()
        print('***** Example 1 is sound *****')
    except:
        print('contradiction found in expr')
    print("##########################")

    ## example 2:
    paullikes = Proposition("paul likes", True)
    paulbuys = Proposition("paul buys", True)

    wendylikes = Proposition("wendy likes", True)
    wendybuys = Proposition("wendy buys", True)

    susanlikes = Proposition("susan likes", True)
    susanbuys = Proposition("susan buys", True)

    basketfull = Proposition("basket has apples", True)

    p5 = Implies(paullikes, paulbuys)
    p6 = Implies(wendylikes, wendybuys)
    p7 = Implies(susanlikes, susanbuys)
    p8 = Implies(wendybuys, basketfull)
    p9 = Or(Or(paullikes, susanlikes), wendylikes)
    tablaux = SemanticTablauxTree([p5, p6, p7, p8, p9, basketfull])
    try:
        tablaux.expand()
        print('***** Example 2 is sound *****')
    except:
        print('contradiction found in expr')
    print("##########################")
    ## example 3:
    yuehblackmailed = Proposition("blackmailed yueh", True)
    yuehpactshark = Proposition("yueh pacts hark", True)
    yuehloyal = Proposition("yueh loyal", True)
    dukerewards = Proposition("duke rewards yueh", True)

    p10 = Implies(yuehblackmailed, yuehpactshark)
    p11 = Implies(yuehblackmailed, yuehloyal.negate())
    p12 = Equivalent(dukerewards, yuehloyal)
    p13 = And(yuehblackmailed, dukerewards)
    tablaux = SemanticTablauxTree([p10, p11, p12, p13])
    try:
        tablaux.expand()
        print('***** Example 3 is sound *****')
    except:
        print('contradiction found in expr')
    print("##########################")

    ## example 4:
    alfredcar = Proposition("alfted takes car", True)
    alfredbus = Proposition("alfred takes bus", True)
    cargoeswork = Proposition("car goes work", True)
    carhasgas = Proposition("car has gas", True)
    alfredgoeswork = Proposition("alfred goes work", True)
    busgoeswork = Proposition("bus goes work", True)
    cityhastraffic = Proposition("citytraffic", True)

    p14 = Or(alfredbus, alfredcar)
    p15 = Equivalent(cargoeswork, carhasgas)
    p16 = Equivalent(alfredgoeswork, And(alfredcar, cargoeswork))
    p17 = Equivalent(alfredgoeswork, And(alfredbus, busgoeswork))
    p18 = Equivalent(alfredbus, carhasgas.negate())
    p19 = Equivalent(alfredcar, carhasgas)
    p20 = Equivalent(busgoeswork, cityhastraffic.negate())
    p21 = carhasgas.negate()
    p22 = cityhastraffic
    p23 = alfredgoeswork
    tablaux = SemanticTablauxTree([p14, p15, p16, p17, p18, p19, p20, p21, p22, p23])
    try:
        tablaux.expand()
        print('***** Example 4 is sound *****')
    except:
        print('contradiction found in expr')

    p30 = Proposition("whatever", True)
    p31 = p30.negate()
    tablaux = SemanticTablauxTree([p30, p31])
    try:
        tablaux.expand()
        print('***** Example 5 is sound *****')
    except:
        print('contradiction found in expr')
