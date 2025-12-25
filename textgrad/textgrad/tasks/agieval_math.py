from .base import Dataset
import platformdirs
import json
import re
import textgrad as tg
from pathlib import Path

def extract_last_line(string):
    lines = string.split('\n')
    for item in lines[::-1]:
        if item.strip() != '':
            string = item
            break
    return string

def remove_few_shot_prefix(string: str):
    prefix_list = ['The answer is therefore', '答案是']
    for prefix in prefix_list:
        if string.startswith(prefix):
            string = string[len(prefix):].strip()
        elif prefix in string:
            index = string.rfind(prefix)
            if index >= 0:
                string = string[index + len(prefix):].strip()
    return string

def parse_math_answer(setting_name, raw_string):
    if setting_name == 'few-shot-CoT':
        raw_string = extract_last_line(raw_string)
    if setting_name == 'few-shot-CoT' or setting_name == 'few-shot':
        raw_string = remove_few_shot_prefix(raw_string)
        return raw_string

    def remove_boxed(s):
        left = '\\boxed{'
        try:
            assert s[:len(left)] == left
            assert s[-1] == '}'
            answer = s[len(left):-1]
            if '=' in answer:
                answer = answer.split('=')[-1].lstrip(' ')
            return answer
        except:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind('\\boxed')
        if idx < 0:
            idx = string.rfind('\\fbox')
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == '{':
                num_left_braces_open += 1
            if string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(s):
        first_pattern = '\$(.*)\$'
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if '=' in last_match:
                last_match = last_match.split('=')[-1].lstrip(' ')
        return last_match

    def get_answer_without_dollar_sign(s):
        last_match = None
        if '=' in s:
            last_match = s.split('=')[-1].lstrip(' ').rstrip('.')
            if '\\n' in last_match:
                last_match = last_match.split('\\n')[0]
        else:
            pattern = '(?:\\$)?\d+(?:\.\d+)?(?![\w\d])'
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match

    raw_string = remove_few_shot_prefix(raw_string)
    if '\\boxed' in raw_string:
        answer = remove_boxed(last_boxed_only_string(raw_string))
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    return answer

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def _strip_string(string):
    # linebreaks
    string = string.replace('\n', '')
    # print(string)

    # remove inverse spaces
    string = string.replace('\\!', '')
    # print(string)

    # replace \\ with \
    string = string.replace('\\\\', '\\')
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')
    # print(string)

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')
    # print(string)

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

# code from https://github.com/hendrycks/math/blob/main/modeling/math_equivalence.py
def _fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string

def _fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string

def _fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except:
        return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print('WARNING: Both None')
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2

def agieval_math_eval_fn(prediction: tg.Variable, ground_truth_answer: tg.Variable):
    pred = parse_math_answer("", str(prediction.value))
    ref = str(ground_truth_answer.value)
    return int(is_equiv(pred, ref))

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

class AGIEvalMath(Dataset):
    def __init__(self, root:str=None, split:str="train"):
        assert split in ["train", "val", "test"]
        train_path = Path(root) / "AGIEvalMath" / "math_train.jsonl"
        val_path = Path(root) / "AGIEvalMath" / "math_validation.jsonl"
        test_path = Path(root) / "AGIEvalMath" / "math_test.jsonl"
        if split == "train":
            self.data = load_jsonl(train_path)
        elif split == "val":
            self.data = load_jsonl(val_path)
        elif split == "test":
            self.data = load_jsonl(test_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data[index]
        question = row['question']
        answer = row['answer']
        return question, answer
    
    def get_task_description(self):
        return """Let's think step by step."""
