# litert-lm 을 ollama api로 서빙해보기

cpp폴더에서 

빌드 : ./bazelisk build //:multimodal_cli

그래픽 실행 : ./bazel-bin/multimodal_cli [모델의 로컬주소(litert-lm 형식만 지원됨)] --ui

서버 실행 : ./bazel-bin/multimodal_cli [모델의 로컬주소(litert-lm 형식만 지원됨)] --server
