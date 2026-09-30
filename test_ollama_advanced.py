import requests
import json

BASE_URL = "http://127.0.0.1:11434"

def test_api_show():
    print("\n--- TEST: POST /api/show ---")
    res = requests.post(f"{BASE_URL}/api/show", json={"name": "litert-lm:latest"}, timeout=5)
    print(f"Status: {res.status_code}")
    print(f"Body: {res.text[:200]}")
    assert res.status_code == 200
    print(">>> SUCCESS")
    return True

def test_ollama_client_simulation():
    """Simulate official Ollama Python SDK request pattern"""
    print("\n--- TEST: Official Ollama SDK Pattern Simulation ---")
    res = requests.post(f"{BASE_URL}/api/chat", json={
        "model": "litert-lm:latest",
        "messages": [{"role": "user", "content": "hello"}]
        # stream omitted -> SDK expects stream: True by default!
    }, stream=True, timeout=30)
    
    ct = res.headers.get("Content-Type", "")
    print(f"Content-Type: {ct}")
    assert "application/x-ndjson" in ct, f"Expected ndjson, got {ct}"
    first_chunk = res.raw.readline().decode('utf-8')
    print(f"First chunk: {first_chunk.strip()}")
    j = json.loads(first_chunk)
    assert "message" in j or "done" in j
    print(">>> SUCCESS")
    return True

def test_multimodal_base64():
    print("\n--- TEST: Multimodal base64 image in /api/chat ---")
    # 1x1 Red PNG base64
    red_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {
                "role": "user",
                "content": "이 이미지의 주요 색상은 무엇인가요? 빨간색인가요?",
                "images": [red_png_b64]
            }
        ],
        "stream": False
    }
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
        print(f"Status: {res.status_code}")
        print(f"Body: {res.text}")
        assert res.status_code == 200
        data = res.json()
        content = data.get("message", {}).get("content", "")
        print(f"Vision response: {content}")
        # Make sure it didn't say "이미지를 제공하지 않으셨습니다"
        assert "이미지를 제공하지 않으셨습니다" not in content, "Failed: Model did not receive image!"
        print(">>> SUCCESS: Model received and processed the base64 image!")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_client_abort_streaming():
    print("\n--- TEST: Client disconnection during streaming ---")
    payload = {
        "model": "litert-lm:latest",
        "messages": [
            {"role": "user", "content": "긴 동화를 하나 지어줘. 1000자 이상으로 길게 써줘."}
        ],
        "stream": True
    }
    try:
        res = requests.post(f"{BASE_URL}/api/chat", json=payload, stream=True, timeout=30)
        print(f"Connected, reading 3 chunks...")
        count = 0
        for line in res.iter_lines():
            if line:
                count += 1
                if count >= 3:
                    print("Closing connection prematurely...")
                    res.close()
                    break
        print("Closed connection successfully. Now sending another request to check server health...")
        res2 = requests.post(f"{BASE_URL}/api/chat", json={
            "model": "litert-lm:latest",
            "messages": [{"role": "user", "content": "1+1="}],
            "stream": False
        }, timeout=30)
        print(f"Health check status: {res2.status_code}")
        print(f"Health check response: {res2.text}")
        assert res2.status_code == 200
        print(">>> SUCCESS: Server survived abort!")
        return True
    except Exception as e:
        print(f">>> Failed abort test: {e}")
        return False

if __name__ == "__main__":
    test_api_show()
    test_ollama_client_simulation()
    test_multimodal_base64()
    test_client_abort_streaming()
