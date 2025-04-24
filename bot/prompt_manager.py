from utils import prompts


class PromptManager:
    def __init__(self, prompts_config: dict):
        self.prompts = prompts_config
        self.base_prompt = prompts_config.get('__base__', '')
        
    def get_full_prompt(self, prompt_name: str) -> str:
        if prompt_name not in self.prompts:
            prompt_name = self.prompts.get('__active__', 'default')
        
        specific_prompt = self.prompts.get(prompt_name, '')
        return f"{specific_prompt}\n\n{self.base_prompt}".strip()
    


prompt = PromptManager(prompts)

# assistant_prompt = prompt.get_full_prompt('__active__')

# def get_prompt(a):
#     return a

# system_prompt = get_prompt(assistant_prompt)
# print(assistant_prompt)
# print(system_prompt)