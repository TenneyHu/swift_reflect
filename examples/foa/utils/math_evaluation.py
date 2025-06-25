import re

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def normalize_answer(answer):
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\{(?P<text>.+?)\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    string = string.replace("[.", "[0.")
    string = string.replace("(.", "(0.")

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
                sympy_parser.standard_transformations
                + (sympy_parser.implicit_multiplication_application,)
        ),
    )


def _parse_latex(expr: str) -> str:
    """Attempts to parse latex to an expression sympy can read."""
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")  # Play nice with mixed numbers.
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    # Replace the specific characters that this parser uses.
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  ## implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        'percent',
        'point',
        'percentage'
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "mph",
        "kilometer",
        "millimeter",
        "mm",
        "ounce",
        "dozen",
        "pound",
        "ton",
        "gallon",
        "liter",
        "quart",
        "pint",
        "cup",
        "tablespoon",
        "teaspoon",
        "gram",
        "kilogram",
        "kg",
        "milligram",
        "mg",
        "hour",
        "kwh",
        "calorie",
        "cal",
        "joule",
        "watt",
        "kilowatt",
        "ampere",
        "volt",
        "ohm",
        "hertz",
        "bolts",
        "screw",
        "nail",
        "point",
        "call"
        "cent",
        "euro",
        "dollar",
        "yen",

    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub(f"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))

    # if "\\" in expr:
    if re.search(r"\\(?!\d)", expr):
        try:
            expr = _parse_latex(expr)
        except:
            pass

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    # Remove "\\" if they are still present
    expr = expr.replace("\\", "")

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
            len(expr) > 2
            and expr[0] in TUPLE_CHARS
            and expr[-1] in TUPLE_CHARS
            and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        # elems = [elem.strip() for elem in expr[1:-1].split(",")]
        elems = [elem.strip() for elem in re.split(',|;', expr[1:-1])]
    else:
        elems = [expr]
    return elems


def grade_answer(predicted_answer: str, ground_truth: str) -> bool:
    """
    Args:
    predicted_answer: The answer given by the model
    ground_truth: The correct answer

    Returns:
    bool: Whether the given answer is correct

    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if predicted_answer is None:
        return False

    if str(predicted_answer) == str(ground_truth):
        return True

    ground_truth_normalized_mathd = normalize_answer(ground_truth)
    predicted_answer_normalized_mathd = normalize_answer(predicted_answer)

    # be at least as lenient as mathd
    if ground_truth_normalized_mathd == predicted_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(predicted_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
            ground_truth_normalized[0] != given_normalized[0]
            or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


if __name__ == "__main__":
    # Test the answer checker
    assert grade_answer("0.1", ".1")
    assert grade_answer("-2", "-2")
    assert grade_answer("-2", "-2.0")
    assert grade_answer("-2", "-2.000")
    assert grade_answer("-2", "-2.")
    assert grade_answer("x ∈ [-2,5]", "x \in [-2,5]")
    assert grade_answer("x ∈ [-2,5]", "x \in [-2, 5]")
    assert grade_answer("x ∈ [-0.2,0.5]", "x \in [-0.2, 0.5]")
    assert grade_answer("x ∈ [0.2,-0.5]", "x \in [0.2, -0.5]")
    assert not grade_answer("x ∈ [-0.2,0.5]", "Y \in [-0.2, 0.5]")
    assert not grade_answer("x \in [-0.2,0.5]", "x \in [-0.2, 0.6]")

    assert grade_answer("-1/2", "-0.5")
    assert grade_answer("-1/3", "-3/9")
    assert grade_answer("-1/3", "-2/6")
    assert grade_answer("-1/5", "-0.2")
    assert grade_answer("-1/5", "-\\frac{1}{5}")
    assert grade_answer("-1/5", "-\\frac{2}{10}")

    assert not grade_answer("0.1", ".2")
    assert not grade_answer("-2", "-3")
    assert not grade_answer("-2", "-2.1")
    assert grade_answer("-2", "-\\frac{4}{2}")
    assert not grade_answer("-2", "\\frac{1}{2}")

    assert grade_answer("(1,2,3)", "(1.0,2.0,3.0)")
    assert not grade_answer("(1,2,3)", "(1.0,2.0,3.1)")
    assert grade_answer("(1,2,3)", "(1;2;3)")
    assert grade_answer("4,6,14,15", "4, 6, 14, 15")
    assert not grade_answer("4,6,14,15", "4, 6, 14, 15.5")

    assert not grade_answer("\\tfrac{1}{2} + \\frac1{72}", "\\\\frac{1}{2} + 2/3")
    assert not grade_answer("10, 4, -2", "4, 10, -2")
    assert not grade_answer("10, 4, 2", "4, 12, 2")
    assert grade_answer("10\\text{ units}", "10 ")
    assert not grade_answer("10\\text{ units}", "100 ")
    assert grade_answer("10", "\\$10")
    assert grade_answer("\\left(x-2\\right)\\left(x+2\\right)", "(x-2)(x+2)")
    assert not grade_answer("0.1, 4, 2", "4, .1, 2")
    assert grade_answer("10\\%", "10")
    assert grade_answer("10\\sqrt{3} + \\sqrt4", "10\\sqrt3 + \\sqrt{4}")
    assert grade_answer("\\frac34i", "\\frac{3}{4}i")
    assert grade_answer("\\tfrac83", "\\frac{8}{3}")
    assert not grade_answer("5x - 7y + 11z + 4 = 0", "x + y - z + 2 = 0")
    assert grade_answer("5x - 7y + 11z + 4 = 0", "5x - 7y + 11z + 4 = 0")

    assert grade_answer("50 %", "50 percent")
    assert grade_answer("50%", "50")

    assert grade_answer("\\frac{\\sin x}{\\cos x}", "\\frac{\\sin x}{\\cos x}")
    assert not grade_answer("\\frac{\\sin x}{\\cos x}", "\\frac{\\sin x}{\\sin x}")

    assert grade_answer("$\\begin{pmatrix} 1 \\\\ -2 \\\\ -1 \\end{pmatrix}}$",
                        "\\begin{pmatrix} 1 \\\\ -2 \\\\ -1 \\end{pmatrix}")
    assert grade_answer("$\\begin{pmatrix} 1 \\\\ -2 \\\\ -1 \\end{pmatrix}}$",
                        "\\begin{pmatrix} 1.0 \\\\ -2.0 \\\\ -1.0 \\end{pmatrix}")
    assert not grade_answer("$\\begin{pmatrix} 1 \\\\ -2 \\\\ -1 \\end{pmatrix}}$",
                            "\\begin{pmatrix} 1 \\\\ -2 \\\\ -3 \\end{pmatrix}")
