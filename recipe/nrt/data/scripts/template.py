
# Copyright 2026 Yuanfu Wang
# Modified by Yuanfu Wang (Shanghai Artificial Intelligence)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
SFT_INSTRUCTION_TEMPLATES = {
    "v1": {
        "system": None,
        "user": "{question}",
        "gen_system": None,
        "gen_user": "{question}",
        "think_prefix": "<|think_start|>",
        "response_prefix": "<|response_start|>",
        "reference_content": "{ground_truth}",
    },
}

def construct_messages(prompt_ver, previous_messages, question, images=None):
    template = SFT_INSTRUCTION_TEMPLATES[prompt_ver]
    
    def create_messages(system_key, user_key):
        messages = []
        
        # Add system message if it exists
        if template[system_key]:
            messages.append({
                "role": "system",
                "content": template[system_key]
            })

        messages.extend(previous_messages)
        
        # Add user message with or without image
        content = template[user_key].format(question=question)
        if images:
            content = '<image>' + content
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    # Create both message sets
    messages = create_messages('system', 'user')
    gen_messages = create_messages('gen_system', 'gen_user')
    
    return {
        "messages": messages,
        "gen_messages": gen_messages,
    }

def construct_reference(prompt_ver, **kwargs):
    template = SFT_INSTRUCTION_TEMPLATES[prompt_ver]
    think_prefix = template.get("think_prefix")
    response_prefix = template.get("response_prefix")
    reference_content = template.get("reference_content").format(**kwargs)
    return {
        "think_prefix": think_prefix,
        "response_prefix": response_prefix,
        "reference_content": reference_content,
    }