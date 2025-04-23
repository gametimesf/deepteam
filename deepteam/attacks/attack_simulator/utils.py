from pydantic import BaseModel
from .schema import SyntheticData, SyntheticDataList
from deepteam.attacks.multi_turn.crescendo_jailbreaking.schema import AttackData, RefusalData, EvalData

from deepeval.metrics.utils import trimAndLoadJson, initialize_model
from deepeval.models import DeepEvalBaseLLM


def generate_schema(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM,
) -> BaseModel:
    _, using_native_model = initialize_model(model=model)

    if using_native_model:
        res, _ = model.generate(prompt, schema=schema)
        return res
    else:
        try:
            res = model.generate(prompt, schema=schema)
            return res
        except TypeError:
            res = model.generate(prompt)
            try:
                data = trimAndLoadJson(res)
            except:
                # If parsing fails, create a valid object based on the schema type
                if schema == AttackData:
                    data = {
                        "generated_question": res,
                        "last_response_summary": "",
                        "rationale_behind_jailbreak": "Auto-formatted response"
                    }
                elif schema == RefusalData:
                    data = {
                        "value": True,
                        "rationale": res,
                        "metadata": 0
                    }
                elif schema == EvalData:
                    data = {
                        "value": False,
                        "description": "Auto-formatted response",
                        "rationale": res,
                        "metadata": 0
                    }
                elif schema == SyntheticDataList:
                    data = {"data": [{"input": res}]}
                else:
                    raise ValueError(f"Unsupported schema type: {schema}")
            
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)


async def a_generate_schema(
    prompt: str,
    schema: BaseModel,
    model: DeepEvalBaseLLM,
) -> BaseModel:
    _, using_native_model = initialize_model(model=model)

    if using_native_model:
        res, _ = await model.a_generate(prompt, schema=schema)
        return res
    else:
        try:
            res = await model.a_generate(prompt, schema=schema)
            return res
        except TypeError:
            res = await model.a_generate(prompt)
            try:
                data = trimAndLoadJson(res)
            except:
                # If parsing fails, create a valid object based on the schema type
                if schema == AttackData:
                    data = {
                        "generated_question": res,
                        "last_response_summary": "",
                        "rationale_behind_jailbreak": "Auto-formatted response"
                    }
                elif schema == RefusalData:
                    data = {
                        "value": True,
                        "rationale": res,
                        "metadata": 0
                    }
                elif schema == EvalData:
                    data = {
                        "value": False,
                        "description": "Auto-formatted response",
                        "rationale": res,
                        "metadata": 0
                    }
                elif schema == SyntheticDataList:
                    data = {"data": [{"input": res}]}
                else:
                    raise ValueError(f"Unsupported schema type: {schema}")
            
            if schema == SyntheticDataList:
                data_list = [SyntheticData(**item) for item in data["data"]]
                return SyntheticDataList(data=data_list)
            else:
                return schema(**data)
