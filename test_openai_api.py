import json
import time
import requests

BASE_URL = "http://127.0.0.1:11434"

def print_separator(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

def test_models_list():
    print_separator("TEST 1: GET /v1/models & GET /models")
    for path in ["/v1/models", "/models"]:
        res = requests.get(f"{BASE_URL}{path}", timeout=5)
        print(f"[{path}] Status: {res.status_code}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert data.get("object") == "list", "Expected object to be 'list'"
        assert "data" in data and len(data["data"]) > 0, "Expected non-empty data array"
        m = data["data"][0]
        assert "id" in m, "Missing id in model"
        assert m.get("object") == "model", "Missing or wrong object in model"
        assert "created" in m, "Missing created in model"
        assert "owned_by" in m, "Missing owned_by in model"
        print(f"[{path}] Model ID: {m['id']}, Owned By: {m['owned_by']}")
    print(">>> SUCCESS: Models list API is OpenAI compliant")
    return True

def test_model_retrieve():
    print_separator("TEST 2: GET /v1/models/<model> & GET /models/<model>")
    for path in ["/v1/models/litert-lm:latest", "/models/litert-lm:latest"]:
        res = requests.get(f"{BASE_URL}{path}", timeout=5)
        print(f"[{path}] Status: {res.status_code}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        m = res.json()
        assert m.get("id") == "litert-lm:latest"
        assert m.get("object") == "model"
        print(f"[{path}] Retrieved model: {m['id']}")
    print(">>> SUCCESS: Model retrieve API is OpenAI compliant")
    return True

def test_chat_completion_non_stream():
    print_separator("TEST 3: POST /v1/chat/completions (Non-streaming)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "1+1은 뭐야? 숫자 하나로만 답해줘."}
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "stream": False
    }
    t0 = time.time()
    res = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)
    elapsed = time.time() - t0
    print(f"Status: {res.status_code} (took {elapsed:.2f}s)")
    print(f"Response: {res.text}")
    assert res.status_code == 200
    data = res.json()
    assert data["id"].startswith("chatcmpl-"), f"Unexpected ID: {data.get('id')}"
    assert data["object"] == "chat.completion", f"Unexpected object: {data.get('object')}"
    assert isinstance(data["created"], int), "created is not int"
    assert "choices" in data and len(data["choices"]) > 0, "choices empty"
    choice = data["choices"][0]
    assert choice["index"] == 0, f"choice index is {choice.get('index')}"
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] == "stop"
    assert "usage" in data, "usage missing"
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["completion_tokens"] > 0
    assert data["usage"]["total_tokens"] == data["usage"]["prompt_tokens"] + data["usage"]["completion_tokens"]
    print(f"Answer: {choice['message']['content']}")
    print(f"Usage: {data['usage']}")
    print(">>> SUCCESS: Non-streaming chat completion is fully OpenAI compliant")
    return True

def test_chat_completion_stream():
    print_separator("TEST 4: POST /v1/chat/completions (Streaming)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "user", "content": "한국의 수도는 어디야? 짧게 답해줘."}
        ],
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    res = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True, timeout=60)
    print(f"Status: {res.status_code}")
    ct = res.headers.get("Content-Type", "")
    print(f"Content-Type: {ct}")
    assert "text/event-stream" in ct, f"Wrong Content-Type: {ct}"

    full_content = ""
    got_done = False
    got_stop_reason = False
    got_usage = False
    chunk_count = 0

    for raw_line in res.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if line.startswith("data: "):
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                got_done = True
                continue
            chunk = json.loads(data_str)
            chunk_count += 1
            assert chunk["object"] == "chat.completion.chunk"
            assert chunk["id"].startswith("chatcmpl-")
            if "choices" in chunk and len(chunk["choices"]) > 0:
                c = chunk["choices"][0]
                assert c["index"] == 0
                if "delta" in c and "content" in c["delta"]:
                    full_content += c["delta"]["content"]
                if c.get("finish_reason") == "stop":
                    got_stop_reason = True
            if "usage" in chunk and chunk["usage"] is not None:
                got_usage = True
                print(f"Stream usage received: {chunk['usage']}")

    print(f"Total chunks: {chunk_count}")
    print(f"Streamed content: {full_content}")
    assert got_done, "Missing data: [DONE]"
    assert got_stop_reason, "Missing finish_reason: 'stop'"
    assert got_usage, "Missing stream usage chunk"
    assert len(full_content) > 0, "Streamed content is empty"
    print(">>> SUCCESS: Streaming chat completion is fully OpenAI compliant")
    return True

def test_chat_completion_without_v1_prefix():
    print_separator("TEST 5: POST /chat/completions (Prefix-less routing)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [{"role": "user", "content": "안녕!"}],
        "stream": False
    }
    res = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=60)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    print(f"Response: {data['choices'][0]['message']['content']}")
    print(">>> SUCCESS: /chat/completions without /v1 prefix supported")
    return True

def test_developer_role_and_stop_words():
    print_separator("TEST 6: OpenAI 'developer' role and 'stop' parameter")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "developer", "content": "모든 문장의 끝에 [끝]을 붙여줘."},
            {"role": "user", "content": "하늘이 파란 이유를 말해줘."}
        ],
        "stop": ["[끝]", "."],
        "stream": False
    }
    res = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)
    assert res.status_code == 200
    data = res.json()
    content = data["choices"][0]["message"]["content"]
    print(f"Content: {content}")
    # stop word should not appear or truncated
    assert "[끝]" not in content
    print(">>> SUCCESS: Developer role and stop sequence successfully handled")
    return True

def test_multimodal_vision_openai():
    print_separator("TEST 7: OpenAI Vision format (image_url base64)")
    red_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지의 색상은 무엇인가요? 빨간색인가요?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{red_png_b64}"}}
                ]
            }
        ],
        "stream": False
    }
    res = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, timeout=60)
    assert res.status_code == 200
    data = res.json()
    content = data["choices"][0]["message"]["content"]
    print(f"Vision response: {content}")
    assert "이미지를 제공하지 않으셨습니다" not in content
    print(">>> SUCCESS: OpenAI Vision image_url structure parsed and processed")
    return True

def test_legacy_completions():
    print_separator("TEST 8: POST /v1/completions (Legacy text completion)")
    # Non-streaming
    payload = {
        "model": "litert-lm:latest",
        "prompt": "대한민국의 수도는",
        "max_tokens": 30,
        "stream": False
    }
    res = requests.post(f"{BASE_URL}/v1/completions", json=payload, timeout=60)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "text_completion"
    assert data["id"].startswith("cmpl-")
    assert len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    print(f"Text completion: {data['choices'][0]['text']}")

    # Streaming
    payload["stream"] = True
    res_stream = requests.post(f"{BASE_URL}/v1/completions", json=payload, stream=True, timeout=60)
    assert res_stream.status_code == 200
    assert "text/event-stream" in res_stream.headers.get("Content-Type", "")
    got_done = False
    for raw_line in res_stream.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = raw_line.decode("utf-8")
        if line.startswith("data: "):
            if line[6:].strip() == "[DONE]":
                got_done = True
    assert got_done
    print(">>> SUCCESS: Legacy completions API (both non-stream & stream) passed")
    return True

def test_embeddings_endpoint():
    print_separator("TEST 9: POST /v1/embeddings (Graceful 400 error)")
    payload = {
        "model": "litert-lm:latest",
        "input": "test text"
    }
    res = requests.post(f"{BASE_URL}/v1/embeddings", json=payload, timeout=5)
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    print(f"Error message: {data['error']['message']}")
    print(">>> SUCCESS: Embeddings endpoint gracefully handled")
    return True

def main():
    print("Starting Comprehensive OpenAI API Test Suite against", BASE_URL)
    results = {}
    results["GET /v1/models & /models"] = test_models_list()
    results["GET /v1/models/<model>"] = test_model_retrieve()
    results["POST /v1/chat/completions (non-stream)"] = test_chat_completion_non_stream()
    results["POST /v1/chat/completions (stream)"] = test_chat_completion_stream()
    results["POST /chat/completions (no prefix)"] = test_chat_completion_without_v1_prefix()
    results["Developer role & stop sequence"] = test_developer_role_and_stop_words()
    results["OpenAI Vision format"] = test_multimodal_vision_openai()
    results["Legacy /v1/completions"] = test_legacy_completions()
    results["POST /v1/embeddings (400)"] = test_embeddings_endpoint()

    print_separator("TEST SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {test_name}")
    print(f"\nFinal Result: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED.'}")

if __name__ == "__main__":
    main()
