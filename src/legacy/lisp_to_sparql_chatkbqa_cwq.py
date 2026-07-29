import re
import sys
from typing import List


FB_NS = "http://rdf.freebase.com/ns/"

_OPERATOR_CONVERT_MAP = {
    '( greater equal': '( ge',
    '( greater than':  '( gt',
    '( less equal':    '( le',
    '( less than':     '( lt',
}


def normalize_operators(sexpr: str) -> str:
    for k, v in _OPERATOR_CONVERT_MAP.items():
        sexpr = sexpr.replace(k, v)
        sexpr = sexpr.replace(k.upper(), v)
    return sexpr


def fix_glued_reverse_marker(sexpr: str) -> str:
    return re.sub(r'\(R(?=[^\s)])', '(R ', sexpr)

_YEAR_RE = r"\d{4}"
_YEAR_MONTH_RE = r"\d{4}-\d{2}"
_YEAR_MONTH_DATE_RE = r"\d{4}-\d{2}-\d{2}"
_BARE_DATE_TOKEN_RE = re.compile(
    r"(?<![\w.^])(" + _YEAR_MONTH_DATE_RE + "|" + _YEAR_MONTH_RE + "|" + _YEAR_RE + r")"
    r"(?!\^\^)(?=[\s)]|$)"
)


def type_tag_bare_date_tokens(sexpr: str) -> str:
    def _tag(m: "re.Match") -> str:
        token = m.group(1)
        if len(token) == 4 and token.isdigit() and int(token) >= 3000:
            return token
        return token + "^^http://www.w3.org/2001/XMLSchema#dateTime"

    return _BARE_DATE_TOKEN_RE.sub(_tag, sexpr)


def strip_freebase_prefixes(sexpr: str) -> str:
    # Convert comparison operators
    sexpr = normalize_operators(sexpr)

    sexpr = fix_glued_reverse_marker(sexpr)
    # Strip full URIs
    sexpr = re.sub(r'<http://rdf\.freebase\.com/ns/([^>]+)>', r'\1', sexpr)
    # Strip fb:/fbp: prefixes
    sexpr = re.sub(r'\bfbp?:([^\s()]+)', r'\1', sexpr)
    # Tag bare year/date literals with ^^xsd:dateTime
    sexpr = type_tag_bare_date_tokens(sexpr)
    # lisp_to_nested_expression expects (JOIN not ( JOIN
    sexpr = re.sub(r'\(\s+', '(', sexpr)
    sexpr = re.sub(r'\s+\)', ')', sexpr)
    return sexpr


# ---------------------------------------------------------------------------
# Copied from ChatKBQA logic_form_util_cwq
def lisp_to_nested_expression(lisp_string: str) -> list:
    """
    Parses a lisp s-expression string into a nested Python list.
    e.g. "(JOIN (R foo.bar) m.123)" → ['JOIN', ['R', 'foo.bar'], 'm.123']
    """
    stack: List = []
    current_expression: List = []
    tokens = lisp_string.split()
    for token in tokens:
        while token[0] == '(':
            nested_expression: List = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]
    return current_expression[0]


def _linearize_lisp_expression(expression: list, sub_formula_id: list) -> list:
    sub_formulas = []
    for i, e in enumerate(expression):
        if isinstance(e, list) and e[0] != 'R':
            sub_formulas.extend(_linearize_lisp_expression(e, sub_formula_id))
            expression[i] = '#' + str(sub_formula_id[0] - 1)
    sub_formulas.append(expression)
    sub_formula_id[0] += 1
    return sub_formulas


def lisp_to_sparql(lisp_program: str) -> str:
    clauses = []
    order_clauses = []
    entities = set()
    identical_variables_r = {}
    expression = lisp_to_nested_expression(lisp_program)
    superlative = False

    if expression[0] in ['ARGMAX', 'ARGMIN']:
        superlative = True
        if isinstance(expression[2], list):
            def retrieve_relations(exp: list):
                rtn = []
                for element in exp:
                    if element == 'JOIN':
                        continue
                    elif isinstance(element, str):
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'R':
                        rtn.append(element)
                    elif isinstance(element, list) and element[0] == 'JOIN':
                        rtn.extend(retrieve_relations(element))
                return rtn
            relations = retrieve_relations(expression[2])
            expression = expression[:2]
            expression.extend(relations)

    sub_programs = _linearize_lisp_expression(expression, [0])
    question_var = len(sub_programs) - 1
    count = False

    def get_root(var: int) -> int:
        while var in identical_variables_r:
            if var == identical_variables_r[var]:
                break
            var = identical_variables_r[var]
        return var

    for i, subp in enumerate(sub_programs):
        i = str(i)
        if subp[0] == 'JOIN':
            if isinstance(subp[1], list):  # (R relation)
                if subp[2][:2] in ["m.", "g."]:
                    clauses.append(f"ns:{subp[2]} ns:{subp[1][1]} ?x{i} .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':
                    clauses.append(f"?x{subp[2][1:]} ns:{subp[1][1]} ?x{i} .")
                else:
                    if subp[2].__contains__('^^'):
                        data_type = subp[2].split("^^")[1].split("#")[1]
                        if data_type not in ['integer', 'float', 'dateTime']:
                            subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                        else:
                            subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
                    clauses.append(f"{subp[2]} ns:{subp[1][1]} ?x{i} .")
            else:
                if subp[2][:2] in ["m.", "g."]:
                    clauses.append(f"?x{i} ns:{subp[1]} ns:{subp[2]} .")
                    entities.add(subp[2])
                elif subp[2][0] == '#':
                    clauses.append(f"?x{i} ns:{subp[1]} ?x{subp[2][1:]} .")
                else:
                    if re.match(r'[\w_]*\.[\w_]*\.[\w_]*', subp[2]):
                        pass  # 2-hop relation placeholder
                    elif re.match(r"[a-zA-Z_]*\.[a-zA-Z_]*", subp[2]):
                        subp[2] = 'ns:' + subp[2]
                        clauses.append(f"?x{i} ns:{subp[1]} {subp[2]} .")
                    elif len(subp) > 3:
                        subp[2] = " ".join(subp[2:])
                        clauses.append(f"?x{i} ns:{subp[1]} {subp[2]} .")
                    else:
                        if subp[2].__contains__('^^'):
                            literal_value = subp[2].split("^^")[0].strip('"')
                        else:
                            literal_value = subp[2].strip('"')
                        clauses.append(f"?x{i} ns:{subp[1]} ?st{i} .")
                        clauses.append(
                            f'FILTER (SUBSTR(STR(?st{i}), 1, STRLEN("{literal_value}")) = "{literal_value}")'
                        )

        elif subp[0] == 'AND':
            var1 = int(subp[2][1:])
            rooti = get_root(int(i))
            root1 = get_root(var1)
            if rooti > root1:
                identical_variables_r[rooti] = root1
            else:
                identical_variables_r[root1] = rooti
                root1 = rooti
            if subp[1][0] == "#":
                var2 = int(subp[1][1:])
                root2 = get_root(var2)
                if root1 > root2:
                    identical_variables_r[root1] = root2
                else:
                    identical_variables_r[root2] = root1
            else:
                clauses.append(f"?x{i} ns:type.object.type ns:{subp[1]} .")

        elif subp[0] in ['le', 'lt', 'ge', 'gt']:
            if subp[1].startswith('#'):
                line_num = int(subp[1].replace('#', ''))
                first_relation = sub_programs[line_num][1]
                second_relation = sub_programs[line_num][2]
                if isinstance(first_relation, list):
                    clauses.append(f"?cvt ns:{first_relation[1]} ?x{i} .")
                else:
                    clauses.append(f"?x{i} ns:{first_relation} ?cvt .")
                if isinstance(second_relation, list):
                    clauses.append(f"?y{i} ns:{second_relation[1]} ?cvt .")
                else:
                    clauses.append(f"?cvt ns:{second_relation} ?y{i} .")
            else:
                clauses.append(f"?x{i} ns:{subp[1]} ?y{i} .")
            op = {'le': '<=', 'lt': '<', 'ge': '>=', 'gt': '>'}[subp[0]]
            if subp[2].__contains__('^^'):
                data_type = subp[2].split("^^")[1].split("#")[1]
                if data_type not in ['integer', 'float', 'dateTime']:
                    subp[2] = f'"{subp[2].split("^^")[0] + "-08:00"}"^^<{subp[2].split("^^")[1]}>'
                else:
                    subp[2] = f'"{subp[2].split("^^")[0]}"^^<{subp[2].split("^^")[1]}>'
            if re.match(r'\d+', subp[2]) or re.match(r'"\d+"^^xsd:integer', subp[2]):
                clauses.append(f"FILTER (xsd:integer(?y{i}) {op} {subp[2]})")
            else:
                clauses.append(f"FILTER (?y{i} {op} {subp[2]})")

        elif subp[0] == 'TC':
            var = int(subp[1][1:])
            rooti = get_root(int(i))
            root_var = get_root(var)
            if rooti > root_var:
                identical_variables_r[rooti] = root_var
            else:
                identical_variables_r[root_var] = rooti
            year = subp[3]
            if year.lower() == 'now':
                from_para = '"2015-08-10"^^xsd:dateTime'
                to_para   = '"2015-08-10"^^xsd:dateTime'
            else:
                if "^^" in year:
                    year = year.split("^^")[0]
                from_para = f'"{year}-12-31"^^xsd:dateTime'
                to_para   = f'"{year}-01-01"^^xsd:dateTime'
            rel_from_property = subp[2].split('.')[-1]
            if rel_from_property == 'from':
                rel_to_property = 'to'
            elif rel_from_property == 'end_date':
                subp[2] = subp[2].replace('end_date', 'start_date')
                rel_to_property = 'end_date'
            else:
                rel_to_property = 'to_date'
            opposite_rel = subp[2].replace(rel_from_property, rel_to_property)
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{subp[2]} ?sk0}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{subp[2]} ?sk1 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk1) <= {from_para}) }})')
            clauses.append(f'FILTER(NOT EXISTS {{?x{i} ns:{opposite_rel} ?sk2}} || ')
            clauses.append(f'EXISTS {{?x{i} ns:{opposite_rel} ?sk3 . ')
            clauses.append(f'FILTER(xsd:datetime(?sk3) >= {to_para}) }})')

        elif subp[0] in ['ARGMIN', 'ARGMAX']:
            superlative = True
            if subp[1][0] == '#':
                var = int(subp[1][1:])
                rooti = get_root(int(i))
                root_var = get_root(var)
                if rooti > root_var:
                    identical_variables_r[rooti] = root_var
                else:
                    identical_variables_r[root_var] = rooti
            else:
                clauses.append(f'?x{i} ns:type.object.type ns:{subp[1]} .')
            if len(subp) == 3:
                clauses.append(f'?x{i} ns:{subp[2]} ?arg0 .')
            elif len(subp) > 3:
                for j, relation in enumerate(subp[2:-1]):
                    var0 = f'x{i}' if j == 0 else f'c{j-1}'
                    var1 = f'c{j}'
                    if isinstance(relation, list) and relation[0] == 'R':
                        clauses.append(f'?{var1} ns:{relation[1]} ?{var0} .')
                    else:
                        clauses.append(f'?{var0} ns:{relation} ?{var1} .')
                clauses.append(f'?c{j} ns:{subp[-1]} ?arg0 .')
            order_clauses.append("ORDER BY ?arg0" if subp[0] == 'ARGMIN' else "ORDER BY DESC(?arg0)")
            order_clauses.append("LIMIT 1")

        elif subp[0] == 'COUNT':
            var = int(subp[1][1:])
            root_var = get_root(var)
            identical_variables_r[int(i)] = root_var
            count = True

    # Merge identical variables
    for i in range(len(clauses)):
        for k in identical_variables_r:
            clauses[i] = clauses[i].replace(f'?x{k} ', f'?x{get_root(k)} ')

    question_var = get_root(question_var)
    for i in range(len(clauses)):
        clauses[i] = clauses[i].replace(f'?x{question_var} ', '?x ')

    filter_variables = []
    for clause in clauses:
        for var in re.findall(r'\?\w+', clause):
            var = var.strip()
            if var not in filter_variables and var != '?x' and not var.startswith('?sk'):
                filter_variables.append(var)

    ifent = True
    for var in filter_variables:
        clauses.append(f'FILTER (?x != {var})')
        ifent = False
    if ifent:
        for entity in entities:
            clauses.append(f'FILTER (?x != ns:{entity})')

    clauses.insert(0, "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))")
    clauses.insert(0, "WHERE {")
    clauses.insert(0, "SELECT COUNT DISTINCT ?x" if count else "SELECT DISTINCT ?x")
    clauses.insert(0, "PREFIX ns: <http://rdf.freebase.com/ns/>")
    clauses.append('}')
    clauses.extend(order_clauses)

    return '\n'.join(clauses)


def sexpr_to_sparql(sexpr: str) -> str:
    return lisp_to_sparql(strip_freebase_prefixes(sexpr))
