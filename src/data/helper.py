import re


def remove_docstrings(code_string):
    pattern_double = r'"""[\s\S]*?"""'
    pattern_single = r"'''[\s\S]*?'''"

    code_string = re.sub(pattern_double, '', code_string)
    code_string = re.sub(pattern_single, '', code_string)
    return code_string


def remove_comments(code_string):
    pattern = r'#.*'
    return re.sub(pattern, '', code_string)


def clean_whitespace(text):
    return ' '.join(text.split())


def preprocess_code(text, is_code=True):
    if not text:
        return ""

    if is_code:
        text = remove_docstrings(text)
        text = remove_comments(text)
        text = clean_whitespace(text)
    else:
        text = clean_whitespace(text)

    return text.strip()