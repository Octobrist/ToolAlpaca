import re
from utils import parse_json_string
from sentence_transformers import SentenceTransformer, util
sentence_model = SentenceTransformer(model_name_or_path='/home/huan/projects/llm/paraphrase-MiniLM-L3-v2',
                            cache_folder='/home/huan/projects/llm/paraphrase-MiniLM-L3-v2')

ERROR_DETAILS = {
    'NO_API_CALL': 0,
    'API_NAME_MISMATCH': {'0':0, '1':0, '2':0, '3':0, '4':0},
    'INVALID_PARAMETERS': {'0':0, '1':0, '2':0, '3':0, '4':0},
    'INPUT_MISMATCH': {'0':0, '1':0, '2':0,},
}
def match_score(keywords1, keywords2):
    kw_emb1 = sentence_model.encode(keywords1)
    kw_emb2 = sentence_model.encode(keywords2)
    score = util.cos_sim(kw_emb1, kw_emb2).item()
    return score


def judge_api_name(api_name, golden_answer, api_docs):
    apis_name_list = list(api_docs.keys())
    gt_name = golden_answer['Action']
    if api_name != gt_name:
        if api_name in apis_name_list:
            return 1, api_name
        for name in apis_name_list:
            if re.sub(r'[^a-zA-Z]', '', api_name).lower() == re.sub(r'[^a-zA-Z]', '', name).lower():
                return 2, [name]
        for name in apis_name_list:
            if match_score(api_name, name) >= 0.5:
                return 3, [name]
        else:
            return 4, apis_name_list
    return 0, None

def judge_invalid_key(keys, golden_answer, api_docs):
    key_list = []
    for api_name, key_docs in api_docs.items():
        for key_name in list(key_docs.keys()):
            key_list.append(key_name)

    golden_answer_input = eval(golden_answer['Action_Input'])
    assert isinstance(golden_answer_input, dict)
    golden_answer_keys = list(api_docs[golden_answer['Action']].keys())
    # if golden_answer['Action'] == 'result_get':
    #     print(1)
    for key_name in keys:
        if key_name in golden_answer_keys:
            continue
        if key_name in key_list:
            return 1, (key_name, golden_answer_keys)
        gt_raw_keys = [(kkk, re.sub(r'[^a-zA-Z]', '', kkk).lower()) for kkk in key_list]
        for gt_raw_key in gt_raw_keys:
            if re.sub(r'[^a-zA-Z]', '', key_name).lower() == gt_raw_key[1]:
                return 2, (key_name, [gt_raw_key[0]])
            if match_score(key_name.strip().lower(), gt_raw_key[0].strip().lower()) >= 0.5:
                return 3, (key_name, [gt_raw_key[0]])
        return 4, (key_name, golden_answer_keys)
    return 0, (None, None)

TYPE_DICT = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict
    }
def judge_input_mismatch(answer, answer_input, golden_answer, api_docs):
    golden_answer_input = eval(golden_answer['Action_Input'])
    assert isinstance(golden_answer_input, dict)
    for key, gt_value in golden_answer_input.items():
        value_desc = api_docs[key]
        value_type = None
        for pre_type in TYPE_DICT.keys():
            if f'{pre_type}.' in value_desc.lower() or f'{pre_type}[string].' in value_desc.lower():
                value_type = TYPE_DICT[pre_type]
                break
        assert value_type is not None
        if key not in answer_input.keys():
            continue
        value = answer_input[key]
        if not isinstance(value, value_type):
            return 1, (key, value, value_desc)
        elif str(value).lower() == str(gt_value).lower() or str(gt_value) in str(value) or (match_score(str(gt_value), str(value))>=0.3 and len(str(gt_value).split(' ')) > 2):
            # or len(str(gt_value).split(',')) >= 2
                # (answer == 'current_get' and key == 'access_key') \
                # or (answer == 'result_get' and key == 'appid') \
                # or (key == 'api_key') or \
            continue
        else:
            return 2, (key, value, value_desc)
    return 0, (None, None, None)


def static_feedback(answer, golden_answer, api_docs):
    regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    if 'error' in answer.keys():
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    if 'Could not parse LLM output:' in answer['output']:
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    if 'intermediate_steps' not in answer.keys():
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    if 'intermediate_steps' in answer.keys() and len(answer['intermediate_steps']) == 0:
        # match = re.search(regex, answer['output'], re.DOTALL)
        # if not match:
        #     ERROR_DETAILS['NO_API_CALL'] += 1
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
        # action = match.group(1).strip()
        # action_input = match.group(2).strip(" ").strip('"')
    else:
        action = answer['intermediate_steps'][-1][0][0]
        action_input = answer['intermediate_steps'][-1][0][1]

    try:
        action_input = parse_json_string(action_input)
        assert isinstance(action_input, dict)
    except:
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, action, action_input

    # print(action, action_input)
    error_type, detail = judge_api_name(action, golden_answer, api_docs)
    ERROR_DETAILS['API_NAME_MISMATCH'][str(error_type)] += 1
    if error_type != 0:
        return 'api_name_mismatch', error_type, detail, action, action_input
    assert action == golden_answer['Action']
    error_type, detail = judge_invalid_key(list(action_input.keys()), golden_answer, api_docs)
    ERROR_DETAILS['INVALID_PARAMETERS'][str(error_type)] += 1
    if error_type != 0:
        return 'invalid_parameters', error_type, detail, action, action_input
    error_type, detail = judge_input_mismatch(action_input, action_input, golden_answer, api_docs[action])
    ERROR_DETAILS['INPUT_MISMATCH'][str(error_type)] += 1
    if error_type != 0:
        return 'input_mismatch', error_type, detail, action, action_input
    return None, None, None, action, action_input


def mutil_static_feedback(answer, golden_answer, cur_time, api_docs):
    regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    if 'error' in answer.keys():
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    if 'Could not parse LLM output:' in answer['output']:
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    if 'intermediate_steps' not in answer.keys():
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    if 'intermediate_steps' in answer.keys() and len(answer['intermediate_steps']) == 0:
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, None, None
    else:
        action = answer['intermediate_steps'][cur_time][0][0]
        action_input = answer['intermediate_steps'][cur_time][0][1]

    try:
        action_input = parse_json_string(action_input)
        assert isinstance(action_input, dict)
    except:
        ERROR_DETAILS['NO_API_CALL'] += 1
        return 'no_api_call', None, None, action, action_input

    # print(action, action_input)
    error_type, detail = judge_api_name(action, golden_answer, api_docs)
    ERROR_DETAILS['API_NAME_MISMATCH'][str(error_type)] += 1
    if error_type != 0:
        return 'api_name_mismatch', error_type, detail, action, action_input
    assert action == golden_answer['Action']
    error_type, detail = judge_invalid_key(list(action_input.keys()), golden_answer, api_docs)
    ERROR_DETAILS['INVALID_PARAMETERS'][str(error_type)] += 1
    if error_type != 0:
        return 'invalid_parameters', error_type, detail, action, action_input
    error_type, detail = judge_input_mismatch(action_input, action_input, golden_answer, api_docs[action])
    ERROR_DETAILS['INPUT_MISMATCH'][str(error_type)] += 1
    if error_type != 0:
        return 'input_mismatch', error_type, detail, action, action_input
    return None, None, None, action, action_input

def get_error_details():
    return ERROR_DETAILS