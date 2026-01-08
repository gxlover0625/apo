import re

def strip_latex(response: str) -> str:
  if response.startswith("$") and response.endswith("$"):
    response = response[1:-1]
  if "boxed{" in response and response.endswith("}"):
    response = response[0:-1].split("boxed{")[1]
  if "text{" in response and response.endswith("}"):
    response = response[0:-1].split("text{")[1]
  if "texttt{" in response and response.endswith("}"):
    response = response[0:-1].split("texttt{")[1]
  return response

def extract_answer(sample: str) -> str:
  """Extracts the final answer from the sample."""
  answer_prefixes = [
      "The answer is:",
      "The final answer is ",
      "The final answer is: ",
      "The answer is "
  ]
  answer = sample
  for answer_prefix in answer_prefixes:
    if answer_prefix in answer:
      answer = answer.split(answer_prefix)[-1].strip()
  if answer.endswith("."):
    answer = answer[:-1]
  return strip_latex(answer)

def extract_answer(sample: str) -> str:
  """Extracts the final answer from the sample."""
  answer_prefixes = [
      "The answer is:",
      "The final answer is ",
      "The final answer is: ",
      "The answer is "
  ]
  answer = sample
  for answer_prefix in answer_prefixes:
    if answer_prefix in answer:
      answer = answer.split(answer_prefix)[-1].strip()
  if answer.endswith("."):
    answer = answer[:-1]
  return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
  """Fuzzy match function for BigBench Extra Hard."""
  if prediction == reference:
    return True

  # (a) vs a
  if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
    return prediction[1] == reference
  if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
    return reference[1] == prediction

  # Numbers
  try:
    if float(prediction) == float(reference):
      return True
  except ValueError:
    pass

  # quote issues
  if prediction.replace("'", "") == reference.replace("'", ""):
    return True

  # Bracket issues
  if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
    return True

  # Question mark issues
  if prediction.endswith("?") and prediction[:-1] == reference:
    return True

  return False

def preprocess_sample(sample: str) -> str:
  prediction = extract_answer(sample.strip()).lower()
  prediction = prediction.replace(", ", ",").replace("**", "")
  prediction = prediction.split("\n")[0]
  prediction = prediction[0:-1] if prediction.endswith(".") else prediction
  return prediction

def preprocess_reference(reference: str) -> str:
  reference = reference.strip().lower()
  reference = reference.replace(", ", ",")
  return reference

def bbeh_mcq_eval_fn(prediction: str, ground_truth_answer: str):
    pred = preprocess_sample(prediction)
    ref = preprocess_reference(ground_truth_answer)
    return fuzzy_match(pred, ref)

def gpqa_process_pred(answer):
    patterns = [r'answer is \((.)\)', r'Answer: \((.)\)', r'answer: \((.)\)', r'answer \((.)\)', r'\((.)\)']
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match and match.group(1) in ['A', 'B', 'C', 'D', 'E']:
            return match.group(1)
    return None
  
def gpqa_eval_fn(prediction: str, ground_truth_answer: str):
    pred = gpqa_process_pred(prediction)
    ref = ground_truth_answer
    return pred == ref

def get_eval_fn(dataset_name:str):
    if dataset_name in ["Geo_Group", "BBH_geometric_shapes", "bbeh_geometric_shapes", "Logical_Group", "BBH_logical_deduction_seven_objects", "bbeh_boardgame_qa"]:
        return bbeh_mcq_eval_fn
    elif dataset_name in ["gpqa"]:
        return gpqa_eval_fn
    elif dataset_name in ["agieval_aqua", "agieval_gaokao_math", "agieval_sat"]:
        return gpqa_eval_fn

