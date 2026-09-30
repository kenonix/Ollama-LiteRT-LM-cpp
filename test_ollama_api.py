import json
import time
import requests

BASE_URL = "http://127.0.0.1:11434"

def print_separator(title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

def test_root():
    print_separator("TEST 1: GET / (Root Health Check)")
    try:
        res = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Status: {res.status_code}")
        print(f"Headers: Content-Type={res.headers.get('Content-Type')}")
        print(f"Body: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "Ollama is running" in res.text, f"Unexpected body: {res.text}"
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_tags():
    print_separator("TEST 2: GET /api/tags (Model List)")
    try:
        res = requests.get(f"{BASE_URL}/api/tags", timeout=5)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert "models" in data, "No 'models' field"
        assert len(data["models"]) > 0, "Empty models list"
        model = data["models"][0]
        assert "name" in model, "No 'name' field"
        assert "model" in model, "No 'model' field"
        print(f"Found model: {model['name']}")
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_version():
    print_separator("TEST 3: GET /api/version (Version)")
    try:
        res = requests.get(f"{BASE_URL}/api/version", timeout=5)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert "version" in data, "Missing 'version' field"
        print(f"Version: {data['version']}")
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_show():
    print_separator("TEST 4: POST /api/show (Model Info)")
    try:
        res = requests.post(f"{BASE_URL}/api/show", json={"name": "litert-lm:latest"}, timeout=5)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert "details" in data, "Missing 'details' field"
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_chat_non_stream():
    print_separator("TEST 5: POST /api/chat (Non-streaming, stream=false)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "user", "content": "1+1은 뭐야? 한 단어로만 대답해줘."}
        ],
        "stream": False
    }
    t0 = time.time()
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
        elapsed = time.time() - t0
        print(f"Status: {res.status_code} (took {elapsed:.2f}s)")
        print(f"Response: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert "message" in data, "Missing 'message' field"
        assert data["message"].get("role") == "assistant"
        assert "content" in data["message"]
        assert data.get("done") is True
        assert "created_at" in data, "Missing 'created_at'"
        print(f"Generated text: {data['message']['content']}")
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_chat_stream():
    print_separator("TEST 6: POST /api/chat (Streaming, stream=true)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "user", "content": "안녕? 자기소개 한 문장으로 해줘."}
        ],
        "stream": True
    }
    t0 = time.time()
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, stream=True, timeout=60)
        print(f"Status: {res.status_code}")
        print(f"Content-Type: {res.headers.get('Content-Type')}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        assert "application/x-ndjson" in res.headers.get("Content-Type", "")
        
        chunks = []
        full_content = ""
        done_found = False
        
        for line in res.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunks.append(line)
            j = json.loads(line)
            if "message" in j and "content" in j["message"]:
                full_content += j["message"]["content"]
            if j.get("done") is True:
                done_found = True
                assert j.get("done_reason") == "stop"
                
        elapsed = time.time() - t0
        print(f"Received {len(chunks)} chunks in {elapsed:.2f}s")
        print(f"Full content: {full_content}")
        assert done_found, "Never received done: true"
        assert len(full_content) > 0, "No content generated"
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_chat_stream_default():
    print_separator("TEST 7: POST /api/chat (stream omitted - Ollama default is stream: true)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "user", "content": "사과의 색깔은?"}
        ]
    }
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, stream=True, timeout=60)
        print(f"Status: {res.status_code}")
        ct = res.headers.get('Content-Type', '')
        print(f"Content-Type: {ct}")
        assert "application/x-ndjson" in ct, f"Expected application/x-ndjson, got {ct}"
        
        count = 0
        for line in res.iter_lines(decode_unicode=True):
            if line:
                count += 1
        print(f"Successfully streamed {count} chunks by default!")
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_chat_multi_turn():
    print_separator("TEST 8: POST /api/chat (Multi-turn conversation)")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "system", "content": "너는 유용한 AI 어시스턴트야."},
            {"role": "user", "content": "내 이름은 영희야."},
            {"role": "assistant", "content": "안녕하세요 영희님! 반갑습니다."},
            {"role": "user", "content": "내 이름이 뭐라고 했지? 이름만 말해줘."}
        ],
        "stream": False
    }
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        content = data.get("message", {}).get("content", "")
        print(f"AI response: {content}")
        assert "영희" in content, "Model forgot earlier context"
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_api_generate_non_stream():
    print_separator("TEST 9: POST /api/generate (stream=false)")
    payload = {
        "model": "litert-lm:latest",
        "prompt": "지구는 둥글다. 맞으면 'O', 틀리면 'X'로만 답해줘.",
        "stream": False
    }
    try:
        res = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=60)
        print(f"Status: {res.status_code}")
        print(f"Response: {res.text}")
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"
        data = res.json()
        assert "response" in data, "Missing 'response' field"
        assert data.get("done") is True
        assert data.get("done_reason") == "stop"
        print(f"Generated: {data['response']}")
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_api_generate_stream():
    print_separator("TEST 10: POST /api/generate (streaming default)")
    payload = {
        "model": "litert-lm:latest",
        "prompt": "봄, 여름, 가을, 겨울 중 가장 좋아하는 계절을 하나 고르고 이유를 한 문장으로 말해줘."
    }
    try:
        res = requests.post(f"{BASE_URL}/api/generate", json=payload, stream=True, timeout=60)
        print(f"Status: {res.status_code}")
        ct = res.headers.get("Content-Type", "")
        print(f"Content-Type: {ct}")
        assert "application/x-ndjson" in ct
        
        chunks = []
        full_text = ""
        done_found = False
        for line in res.iter_lines(decode_unicode=True):
            if not line:
                continue
            chunks.append(line)
            j = json.loads(line)
            full_text += j.get("response", "")
            if j.get("done") is True:
                done_found = True
        print(f"Received {len(chunks)} chunks")
        print(f"Full text: {full_text}")
        assert done_found, "Missing done: true"
        assert len(full_text) > 0, "Empty response"
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def test_cors_options():
    print_separator("TEST 11: OPTIONS /api/generate & /api/chat (CORS Preflight)")
    try:
        res = requests.options(f"{BASE_URL}/api/generate", timeout=5)
        assert res.status_code == 204
        assert res.headers.get("Access-Control-Allow-Origin") == "*"
        print(">>> SUCCESS")
        return True
    except Exception as e:
        print(f">>> FAILED: {e}")
        return False

def main():
    print("Starting Patched Ollama API Test Suite against", BASE_URL)
    results = {}
    results["GET /"] = test_root()
    results["GET /api/tags"] = test_tags()
    results["GET /api/version"] = test_version()
    results["POST /api/show"] = test_show()
    results["OPTIONS /api/generate (CORS)"] = test_cors_options()
    results["POST /api/chat (stream=False)"] = test_chat_non_stream()
    results["POST /api/chat (stream=True)"] = test_chat_stream()
    results["POST /api/chat (stream omitted default)"] = test_chat_stream_default()
    results["POST /api/chat (multi-turn)"] = test_chat_multi_turn()
    results["POST /api/generate (stream=False)"] = test_api_generate_non_stream()
    results["POST /api/generate (streaming default)"] = test_api_generate_stream()

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
