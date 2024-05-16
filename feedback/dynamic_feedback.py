from utils import parse_json_string


def get_incorrect_samples(eval_info):
    incorrect_samples = []
    for key, values in eval_info.items():
        if key == 'statistics':
            continue
        for value_idx, value in enumerate(values):
            if 'response_correctness' not in value.keys() or value['response_correctness'] == 'No' or value['response_correctness'] == 'Uncertain':
                incorrect_samples.append((key, value['id']))
    assert len(incorrect_samples) == eval_info['statistics']['response']['No'] + eval_info['statistics']['error_num'] + eval_info['statistics']['response']['Uncertain']
    return incorrect_samples

def get_cur_details_from_last_feedbacks(last_feedbacks):
    regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
    action, action_input, idx = None, None, 0
    for idx, feedback in enumerate(last_feedbacks[::-1]):
        action = feedback[0][0]
        action_input = feedback[0][1]
        if 'Could not parse my generation' in feedback[1] and action_input == "None": # 得到有意义的上次调用
            continue
        if action is not None and action_input != 'None':
            break
    return action, action_input, idx

def get_cur_details(answer):
        regex = r"ASSISTANT\s*Action\s*\d*\s*:(.*?)\nASSISTANT\s*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        if 'error' in answer.keys():
            return None, None, None, answer['error']
        if 'Could not parse LLM output:' in answer['output']:
            return None, None, None, answer['output']
        if 'intermediate_steps' not in answer.keys():
            return None, None, None, answer['output']
        if 'intermediate_steps' in answer.keys() and len(answer['intermediate_steps']) == 0:
            # match = re.search(regex, answer['output'], re.DOTALL)
            # if not match:
            #     ERROR_DETAILS['NO_API_CALL'] += 1
            return None, None, None, answer['output']
            # action = match.group(1).strip()
            # action_input = match.group(2).strip(" ").strip('"')
        else:
            action = answer['intermediate_steps'][-1][0][0]
            action_input = answer['intermediate_steps'][-1][0][1]
            action_observation = answer['intermediate_steps'][-1][1]
        try:
            action_input = parse_json_string(action_input)
            assert isinstance(action_input, dict)
        except:
            return action, action_input, action_observation, answer['output']
        return action, action_input, action_observation, answer['output']

def dynamic_feedback(answer, golden_answer, api_docs, gpt4):
    return