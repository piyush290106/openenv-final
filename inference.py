import json
import os
from openai import OpenAI

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://piyush290106-openenv-ecommerce-api.hf.space/v1"
)

MODEL_NAME = os.getenv("MODEL_NAME", "openenv-agent")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def b(x):
    return "true" if x else "false"

def main():
    rewards = []
    steps = 0
    success = False

    print(f"[START] task=inventory_control env=openenv-ecommerce model={MODEL_NAME}", flush=True)

    try:
        done = False

        while not done and steps < 3:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": "Choose the next best action and execute one step."
                }],
                temperature=0
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)

            action = data.get("action")
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))

            rewards.append(reward)
            steps += 1

            action_str = json.dumps(action, separators=(",", ":"))

            print(
                f"[STEP] step={steps} action={action_str} reward={reward:.2f} done={b(done)} error=null",
                flush=True
            )

        success = done

    except Exception as e:
        print(
            f"[STEP] step={steps+1} action=null reward=0.00 done=false error={json.dumps(str(e))}",
            flush=True
        )

    finally:
        score = sum(rewards) if rewards else 0.0
        rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"

        print(
            f"[END] success={b(success)} steps={steps} score={score:.2f} rewards={rewards_str}",
            flush=True
        )

if __name__ == "__main__":
    main()
