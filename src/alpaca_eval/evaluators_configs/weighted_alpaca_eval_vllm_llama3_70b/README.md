# Annotator Served by vLLM

This config demonstrates how to utilize an annotator served by vLLM. This brings some advantages: 
- Allow users to use "weighted"-style annotator, similar to `weighted_alpaca_eval_gpt4_turbo`;
- One vLLM server can support multiple nodes in a cluster environment;
- Easy setup using vLLM's OpenAI-compatible APIs. 

## Setup
1. Start the vLLM Server:

    ```bash
    vllm serve /home/shared/Meta-Llama-3-70B-Instruct --dtype auto --api-key token-abc123
    ```

2. Create the client config `local_configs.yaml` in `client_configs` folder:

    ```bash 
    default:
        - api_key: "token-abc123"
        base_url: "http://localhost:8000/v1"
    ```

3. Run evaluation: 

    ```bash
    export OPENAI_CLIENT_CONFIG_PATH=<path to local_configs.yaml>
    alpaca_eval evaluate --model_outputs 'example/outputs.json' --annotators_config weighted_alpaca_eval_vllm_llama3_70b
    ```