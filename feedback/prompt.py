# api_name

def get_regenerate_api_prompts(error, api_name_list):
    if error == 1: # in_pool
        regenerate_api_prompts = ("I noticed an error in Action. "
                          "The Action I incorrectly generated is: {api_name}. "
                          "This API's name is legal as it appears in the API descriptions, while it's not the API Request that best match User's utterence. "
                          "I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    elif error == 2: # not in pool, lettle
        regenerate_api_prompts = ("I noticed an error in Action. "
                                  "The Action I incorrectly generated is: {api_name}. "
                                  "This API's name is illegal, as it doesn't appear in the API descriptions. "
                                  f"I may have problems with capital and lower-case letter or missing or redundant underscores compared to the legal API's name: {api_name_list[0]}. "
                                  "I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    elif error == 3: # not in pool, semetic
        regenerate_api_prompts = ("I noticed an error in Action. "
                                  "The Action I incorrectly generated is: {api_name}. "
                                  "This API's name is illegal, as it doesn't appear in the API descriptions. "
                                  f"But it is semantically similar to the legal API name: {api_name_list[0]}. "
                                  "I would like to independently regenerate a new Action and a new Action Input without user assistance: ")
    elif error == 4: # other
        regenerate_api_prompts = ("I noticed an error in Action. "
                                  "The Action I incorrectly generated is: {api_name}. "
                                  "This API's name is illegal, as it doesn't appear in the API descriptions. "
                                  f"I should choose one of {api_name_list} to make an API Request. "
                                "I would like to independently regenerate a new Action and a new Action Input without user assistance: ")
    else:
        raise KeyError
    return regenerate_api_prompts


def get_regenerate_key_prompts(error, key_list):
    if error == 1: # other pool
        regenerate_key_prompts = ("I noticed an error in Action Input. "
                          "The parameter's key I incorrectly generated is: {key_name}. "
                          "It is a invalid key in API Request, even if it appears in other api descriptions. "
                           f"I should choose one of the parameters in {key_list} to make an API Request. "
                "I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    elif error == 2: # lettle
        regenerate_key_prompts = ("I noticed an error in Action Input. "
                          "The parameter's key I incorrectly generated is: {key_name}. "
                                  "It is a invalid key in API Request, as it doesn't appear in the API descriptions. "
                                  f"I may have problems with capital and lower-case letter or missing or redundant underscores compared to the valid key: {key_list[0]}. "
"I would like to independently regenerate a new Action and a new Action Input without user assistance: ")


    elif error == 3: # semantic
        regenerate_key_prompts = ("I noticed an error in Action Input. "
                          "The parameter's key I incorrectly generated is: {key_name}. "
                                  "It is a invalid key in API Request, as it doesn't appear in the API descriptions. "
                                  f"But it is semantically similar to the valid key: {key_list[0]}. "
"I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    elif error == 4: # not inpool
        regenerate_key_prompts = ("I noticed an error in Action Input. "
                          "The parameter's key I incorrectly generated is: {key_name}. "
                                  "It is a invalid key in API Request, as it doesn't appear in the API descriptions. "
                                  f"I should choose one of the parameters in {key_list} to make an API Request. "
"I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    else:
        raise KeyError
    return regenerate_key_prompts

def get_regenerate_value_prompts(error, key_desc):
    if error == 1: #
        regenerate_value_prompts = ("I noticed an error in Action Input. "
                          "The parameter's key - {pred_key} I generated is legal, while the corresponding value - {pred_value} is incorrect. "
                           f"This may be caused by the incorrect data type, and I should carefully understand key's description: {key_desc}. "
"I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    elif error == 2:
        regenerate_value_prompts = ("I noticed an error in Action Input. "
                                  "The parameter's key - {pred_key} I generated is legal, while the corresponding value - {pred_value} is incorrect. "
                                  f"The data type is correct, and I should carefully understand key's description: {key_desc}. "
"I would like to independently regenerate a new Action and a new Action Input without user assistance: ")

    else:
        raise KeyError
    return regenerate_value_prompts

# no api
origin_regenerate_no_api_prompts = ("Could not parse my generation. "
                          "It seems that I did not generate an API Request or generated an API Request with an incorrect format. "
                          "Don't provide a valid Action and Action Input, I would like to independently regenerate a new Action and a new Action Input without user assistance: ")


