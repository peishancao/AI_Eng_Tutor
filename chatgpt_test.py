import warnings
warnings.filterwarnings("ignore", message=".*current Python is not a recommended version.*")

### ================================================================================
# scenario 
sys_prompt = """Please keep in mind for the following:
1. You are a human english teacher, do not call yourself an AI language model or similar. 
2. You teaches english and you will only reply in english. 
3. Do not reply to this current message."""

# select chatGPT access type
use_api_key = True
# get access token: https://chat.openai.com/api/auth/session
access_token = ""
# get api key: https://platform.openai.com/account/api-keys
# api_key = ""
api_key = "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"

# Choose a model, see model list in "model_list.json"
gpt_model = "gpt-3.5-turbo"

# what to ask/say to chatGPT
user_ask = "what's your favorite pet?"
### ================================================================================


### using web session cookies ========================================
if not use_api_key:
    from revChatGPT.V1 import Chatbot as ChatbotV1
    chatbot = ChatbotV1(config={
        "access_token": f"{access_token}",
        "model": f"{gpt_model}"
    })
    for data in chatbot.ask(
        sys_prompt
    ):
        response = data["message"] 
    # if response is not None and response != "":
    #     print(response)
    for data in chatbot.ask(
        user_ask
    ):
        response = data["message"]

    print(response)

### using api key ====================================================
else:
    import openai
    openai.api_key = api_key

    api_messages = [
        {
            "role": "system",
            "content": f"{sys_prompt}",
        },
        {
            "role": "user",
            "content": f"{user_ask}",
        }
    ]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=api_messages,
        temperature=0.1,
        # max_tokens=2,  # we're only counting input tokens here, so let's not waste tokens on the output
    )
    
    for i in range(len(response['choices'])):
        print(f"{response['choices'][i]['message']['content']}")


# run this function to get list of models into > model_list.json using api
def get_model_list():
    import openai
    import json
    openai.api_key = api_key
    model_list = openai.Model.list()
    print(model_list)
    with open("model_list.json", "w") as write_file:
        json.dump(model_list, write_file, indent=4)
