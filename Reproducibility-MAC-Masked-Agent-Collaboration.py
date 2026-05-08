
import json
import random
import time
import asyncio
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.agents import ChatAgent
import logging
logging.disable(logging.INFO)
from rapidfuzz import fuzz
import os
import gc
import torch

def list_to_string(list_input):
    if isinstance(list_input, list):
        return ' '.join(str(item) for item in list_input).strip('[]')
    return str(list_input)

def exact_string_match(corr_results, words):
    str1 = str(corr_results)
    str2 = "(" + str(corr_results) + ")"
    str3 = str(corr_results) + "."
    str4 = "*" + str(corr_results) + "."
    str5 = "**" + str(corr_results) + "."
    str6 = "(" + str(corr_results) + ")."
    str7 = str(corr_results) + ","

    return str1 in words or str2 in words or str3 in words or str4 in words or str5 in words or str6 in words or str7 in words

def clean_final_answer(text):
    """Clean special characters in the final answer."""
    if isinstance(text, str):
        text = text.replace('\n', ' ')
        text = text.replace('"', '').replace("'", '')
        text = text.replace('[', '').replace(']', '')

        return text
    elif isinstance(text, list):
        return [clean_final_answer(item) for item in text]
    else:
        return text

dataset_name = "NEJMQA"
SOTA_name = "Our"
eprm_state = "test_v1"
file_out = open('./R_output/'+dataset_name+'_'+SOTA_name+'_'+eprm_state+'.out', 'a', encoding='utf-8')


file_json='./R_output/'+dataset_name+'_'+SOTA_name+'_'+eprm_state+'_detailed_model_results.json'
os.makedirs(os.path.dirname(file_json), exist_ok=True)

model_medllama2_7b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="medllama2:7b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_med42_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="thewindmom/llama3-med42-8b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_huatuogpt_o1_8b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="ZimaBlueAI/HuatuoGPT-o1-8B:latest",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048*4,
                          "stream": True},
)
model_phi4_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="phi4:14b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_qwen25_14b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:14b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True
                       },
)
model_qwen25_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen2.5:32b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True
                       },
)
model_qwq_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwq:32b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_openthinker_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="openthinker:32b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_deepseek_r1_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="deepseek-r1:32b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_llama3_70b_instruct = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="llama3:70b-instruct",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                            "max_tokens": 2048*4,
                            "stream": True
                            },
)
model_qwen15_72b_chat = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen:72b-chat-v1.5-q6_K",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True
                       },
)
model_qwen15_110b_chat = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen:110b-chat-v1.5-q4_0",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048*4,
                          "stream": True
                          },
)
model_dbrx_132b_instruct = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="dbrx:instruct",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048*4,
                          "stream": True
                          },
)

model_Mixtral_141b_v01 = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="mixtral:v0.1",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048*4,
                          "stream": True
                          },
)
model_wizardlm2_141b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="wizardlm2:8x22b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                          "max_tokens": 2048*4,
                          "stream": True
                          },
)

model_qwen3_30b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen3:30b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)
model_qwen3_32b = ModelFactory.create(
    model_platform=ModelPlatformType.OLLAMA,
    model_type="qwen3:32b",
    url="http://localhost:11434/v1",
    model_config_dict={"temperature": 0.7,
                       "max_tokens": 2048*4,
                       "stream": True},
)


LLM_sys_msg = '''
You are an AI designed to answer multiple-choice questions. For each question, select exactly one answer option. Do NOT provide explanations or commentary unless explicitly requested. Base your selection solely on the information given in the question and answer choices. If uncertain, choose the most likely correct answer based on the available information.
Do NOT repeat the answer option in your answer.
Think step by step, enclose your reasoning process in <think>...</think> tags. Finish your answer with {"The answer is (A)."} inside <answer>...</answer> tags, where X is the correct letter choice. Example: {Question:\nWhich of the following represents an accurate statement concerning arthropods?\nOptions:\nA. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.\nAnswer: <think> Let\'s think step by step. Peptidoglycan is known to comprise the plasma membrane of most bacteria, rather than the exoskeleton of arthropods, which is made of chitin, which rules out (A). The answer (C) is false because arthropods are a highly successful phylum. Likewise, arthropods have paired, jointed appendages, which rules out (D). The only remaining option is (B), as arthropods have an open circulatory system with a dorsal tubular heart.</think>. <answer> The answer is (B). </answer>}

Question:
{question}
Options:
(A) {option_1}
(B) {option_2}
(C) {option_3}
(D) {option_4}
Correct answer: {correct_answer}
'''
MOA_sys_msg = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.Finish your answer with {"The answer is (A)."} inside <answer>...</answer> tags, where X is the correct letter choice.

Responses from models:"""

class ScenarioMedQA:
    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict
        self.answer = scenario_dict["answer"]
        self.question = scenario_dict["question"]
        self.department = scenario_dict["csv_filename"]
    def patient_information(self) -> dict:
        return self.question
    def diagnosis_information(self) -> dict:
        return self.answer
    def department_information(self) -> dict:
        return self.department

class ScenarioLoaderMedQA:
    def __init__(self) -> None:
        with open("./NEJMQA/NEJMQA-655multi_questions.json", "r") as f:
            self.scenario_strs = json.load(f)
        self.scenarios = [ScenarioMedQA(_str) for _str in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return self.scenarios[random.randint(0, len(self.scenarios)-1)]

    def get_scenario(self, id):
        if id is None: return self.sample_scenario()
        return self.scenarios[id]

class DoctorTalker(ChatAgent):
    def __init__(self,
        scenario=None,
        max_infs=10,
        bias=None,
        model = None,
        sys_msg = None,
        memory = None,
        message_window_size = None
    ):
        self.infs = 0
        self.MAX_INFS = max_infs
        self.scenario = scenario
        self.bias = bias

        super().__init__(
            system_message=sys_msg,
            model=model,
            memory=memory,
            message_window_size=message_window_size
        )

    def inference_doctor(self, question) -> str:
        q_prompt = question
        answer = self.step(q_prompt).msgs[0].content
        self.infs += 1
        return answer

def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )

async def run_llm(in_model, patient_response, prev_response=None):

    print("🪪 model_name:", in_model.model_type, file=file_out)
    print("🪪 model_name:", in_model.model_type)

    for sleep_time in [1, 2, 4]:
        try:
            if prev_response:
                assistant_sys_msg = getFinalSystemPrompt(MOA_sys_msg, prev_response)
                Doctor = DoctorTalker(model=in_model, sys_msg=assistant_sys_msg)
                result = Doctor.inference_doctor(question=patient_response)
            else:
                Doctor = DoctorTalker(model=in_model, sys_msg=LLM_sys_msg)
                result = Doctor.inference_doctor(question=patient_response)
            return result
        except Exception as e:

            error_str = str(e)

            print(e)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if "CUDA error: out of memory" in error_str or "out of memory" in error_str:
                wait_time = 2 * (2 ** sleep_time)
                print(f"CUDA out of memory, retrying in {wait_time}s...", file=file_out)
                print(f"CUDA out of memory, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                wait_time = 2 * (sleep_time + 1)
                await asyncio.sleep(wait_time)

    print(f"All attempts failed for model {in_model.model_type}, returning empty result", file=file_out)
    return ""

def calculate_min_fuzzy_score_indices(results):
    """Return indices of the lowest and second-lowest fuzzy scores."""
    base_response = str(results[0])
    fuzzy_scores = []

    for i in range(1, len(results)):
        score = fuzz.ratio(base_response, str(results[i])) / 100.0
        fuzzy_scores.append((i, score))

    sorted_scores = sorted(fuzzy_scores, key=lambda x: x[1])

    if len(sorted_scores) >= 2:
        min_index = sorted_scores[0][0]
        second_min_index = sorted_scores[1][0]
        return min_index, second_min_index, fuzzy_scores
    elif len(sorted_scores) == 1:
        return sorted_scores[0][0], None, fuzzy_scores
    else:
        return None, None, None

async def main():
    scenario_loader = ScenarioLoaderMedQA()

    for _scenario_id in [67, 100]:

        print("🚇 ======================== ======================== ======================== ", file=file_out)
        print("🆔 scenario_id:", _scenario_id, file=file_out)
        print("🆔 scenario_id:", _scenario_id)

        detailed_results = {}

        scenario = scenario_loader.get_scenario(id=_scenario_id)

        print("🥼 department:", scenario.department_information(), file=file_out)

        patient_response = scenario.patient_information()

        print("😶‍🌫️ patient_response:", patient_response, file=file_out)
        layers = 5

        reference_models = [model_openthinker_32b, model_deepseek_r1_32b, model_qwq_32b, model_qwen25_32b, model_phi4_14b, model_qwen25_14b]

        results = []

        detailed_results = {}

        top_models = reference_models

        results = await asyncio.gather(*[run_llm(model, patient_response) for model in top_models])

        removed_models = []
        sorted_scores = []
        min_index = []
        second_min_index = []

        for _ in range(1, layers-1):

            print("🪜 Layer:", _, file=file_out)
            print("🪜 Layer:", _)

            detailed_results = {
                "scenario_id": _scenario_id,
                "question": patient_response,
                "correct_answer": scenario.diagnosis_information(),
                "department": scenario.department_information(),
                "results": results,
                "removed_models": removed_models,
                "sorted_scores": sorted_scores,
                "min_index": min_index,
                "second_min_index": second_min_index,
            }

            with open(file_json, "a", encoding="utf-8") as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)

            if len(top_models) > 2:
                min_index, second_min_index, sorted_scores = calculate_min_fuzzy_score_indices(results)
                print("min_index, second_min_index", min_index, second_min_index)
                print("min_index, second_min_index", min_index, second_min_index, file=file_out)
                if min_index is not None:
                    removed_model = top_models.pop(min_index)
                    print(f"🍕 Removed model: {removed_model.model_type}")
                    removed_models.append(removed_model.model_type)
                    print(f"🍕 Removed model: {removed_model.model_type}", file=file_out)
                if second_min_index is not None:
                    if second_min_index > min_index:
                        second_min_index -= 1
                    removed_model = top_models.pop(second_min_index)
                    print(f"🍕 Removed model: {removed_model.model_type}")
                    removed_models.append(removed_model.model_type)
                    print(f"🍕 Removed model: {removed_model.model_type}", file=file_out)
            elif len(top_models) > 1:
                removed_model = top_models.pop()
                print(f"🍔 Removed model: {removed_model.model_type}")
                print(f"🍔 Removed model: {removed_model.model_type}", file=file_out)
                removed_models.append(removed_model.model_type)
                print(f"🍔 Removed model: {removed_models}")
            else:
                break
            for model in top_models:
                print("current model", model.model_type)

            results = await asyncio.gather(
                *[run_llm(in_model=model, patient_response=patient_response, prev_response=results) for model in top_models]
            )

        results = list_to_string(results)

        final_result = clean_final_answer(results)
        print("🎯Final answer: ", final_result,"🎯", file=file_out)
        print("🎯Final answer: ", final_result,"🎯")

        corr_results = scenario.diagnosis_information()
        print("🚩Correct answer:", corr_results,"🚩", file=file_out)
        print("🚩Correct answer:", corr_results,"🚩")

        fuzzy_matching = fuzz.partial_ratio(str(final_result), str(corr_results))
        if fuzzy_matching < 60:
            print("🤔 Fuzzy matching score: ", fuzzy_matching, "🤔", file=file_out)
        else:
            words = str(results).split()
            print("correction:", corr_results, file=file_out)
            if exact_string_match(corr_results, words):
                print("😁 Fuzzy matching score: ", fuzzy_matching, "😁", file=file_out)
            else:
                print("🤔 Fuzzy matching score: ", fuzzy_matching, "🤔", file=file_out)

time_start = time.time()
try:
    asyncio.run(main())
finally:
    print("...")
time_end = time.time()
print("⏰ Time taken:", time_end - time_start, "seconds", file=file_out)