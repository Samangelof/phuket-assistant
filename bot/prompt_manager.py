from utils import prompts


class PromptManager:
    def __init__(self, prompts_config: dict):
        self.prompts = prompts_config
        self.base_prompt = prompts_config.get('__base__', '')
        
    def get_full_prompt(self, prompt_name: str) -> str:
        if prompt_name not in self.prompts:
            prompt_name = self.prompts.get('__active__', 'default')
        
        specific_prompt = self.prompts.get(prompt_name, '')
        return f"{self.base_prompt}\n\n{specific_prompt}".strip()
    


prompt = PromptManager(prompts)